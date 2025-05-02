"""Tests for the Chain implementation and its components."""

import unittest
from unittest.mock import Mock

from sifaka.chain import (
    Chain,
    ChainResult,
    ChainOrchestrator,
    ChainExecutor,
    FeedbackFormatter,
    RetryManager,
)
from sifaka.critics import CriticCore
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models.base import ModelProvider
from sifaka.rules import Rule, RuleResult
from sifaka.validation import Validator, ValidationResult


class TestChain(unittest.TestCase):
    """Tests for the Chain implementation and its components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock model provider
        self.model = Mock(spec=ModelProvider)
        self.model.generate.return_value = "Test output"
        self.model.model_name = "test_model"

        # Create mock rules
        self.passing_rule = Mock(spec=Rule)
        self.passing_rule.validate.return_value = RuleResult(passed=True, message="Passed")

        self.failing_rule = Mock(spec=Rule)
        self.failing_rule.validate.return_value = RuleResult(passed=False, message="Failed")

        # Create mock critic
        self.critic = Mock(spec=CriticCore)
        self.critic.critique.return_value = {"feedback": "Needs improvement", "score": 0.5}

    def test_chain_with_passing_rules(self):
        """Test chain with rules that pass validation."""
        # Create chain with only passing rules
        chain = Chain(model=self.model, rules=[self.passing_rule], critic=None, max_attempts=3)

        # Run chain
        result = chain.run("Test prompt")

        # Verify results
        self.assertEqual(result.output, "Test output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)
        self.model.generate.assert_called_once_with("Test prompt")
        self.passing_rule.validate.assert_called_once_with("Test output")

    def test_chain_with_failing_rules_no_critic(self):
        """Test chain with rules that fail validation and no critic."""
        # Create chain with failing rules and no critic
        chain = Chain(model=self.model, rules=[self.failing_rule], critic=None, max_attempts=3)

        # Run chain - should raise ValueError
        with self.assertRaises(ValueError):
            chain.run("Test prompt")

        # Verify model and rule were called
        self.model.generate.assert_called_once_with("Test prompt")
        self.failing_rule.validate.assert_called_once_with("Test output")

    def test_chain_with_failing_rules_and_critic(self):
        """Test chain with rules that fail validation and a critic."""
        # Create mock improver
        improver = Mock(spec=Improver)
        improver.improve.return_value = ImprovementResult(
            output="Test output", critique_details={"feedback": "Needs improvement"}, improved=True
        )
        improver.get_feedback.return_value = "Needs improvement"

        # Create chain with failing rules and a critic
        chain = Chain(
            model=self.model, rules=[self.failing_rule], critic=self.critic, max_attempts=3
        )

        # Replace the improver with our mock
        chain.improver = improver

        # Mock validator to make test pass on second attempt
        # First attempt: validation fails
        # Second attempt: validation passes
        chain.validator.validate = Mock()
        chain.validator.validate.side_effect = [
            ValidationResult(
                output="Test output",
                rule_results=[RuleResult(passed=False, message="Failed first time")],
                all_passed=False,
            ),
            ValidationResult(
                output="Test output",
                rule_results=[RuleResult(passed=True, message="Passed second time")],
                all_passed=True,
            ),
        ]

        # Run chain
        result = chain.run("Test prompt")

        # Verify results
        self.assertEqual(result.output, "Test output")  # Second attempt output
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)  # Second attempt passed

        # Verify model was called twice with different prompts
        self.assertEqual(self.model.generate.call_count, 2)
        self.model.generate.assert_any_call("Test prompt")  # First call
        # Second call should include feedback
        self.model.generate.assert_any_call(
            "Test prompt\n\nPrevious attempt feedback:\nNeeds improvement"
        )

    def test_max_attempts(self):
        """Test chain respects max_attempts setting."""
        # Create mock improver
        improver = Mock(spec=Improver)
        improver.improve.return_value = ImprovementResult(
            output="Test output", critique_details={"feedback": "Needs improvement"}, improved=True
        )
        improver.get_feedback.return_value = "Needs improvement"

        # Create chain with fewer max_attempts
        chain = Chain(
            model=self.model, rules=[self.failing_rule], critic=self.critic, max_attempts=2
        )

        # Replace the improver with our mock
        chain.improver = improver

        # Always fail validation
        chain.validator.validate = Mock()
        chain.validator.validate.return_value = ValidationResult(
            output="Test output",
            rule_results=[RuleResult(passed=False, message="Always failing")],
            all_passed=False,
        )

        # Run chain - should raise ValueError after max_attempts
        with self.assertRaises(ValueError) as context:
            chain.run("Test prompt")

        # Verify error message includes attempt count
        self.assertIn("Validation failed after 2 attempts", str(context.exception))

        # Verify model was called twice
        self.assertEqual(self.model.generate.call_count, 2)

    def test_feedback_formatter(self):
        """Test FeedbackFormatter class."""
        formatter = FeedbackFormatter()

        # Test format_feedback with valid input
        feedback = formatter.format_feedback({"feedback": "Test feedback"})
        self.assertEqual(feedback, "Test feedback")

        # Test format_feedback with missing feedback
        feedback = formatter.format_feedback({"other": "value"})
        self.assertEqual(feedback, "")

        # Test create_prompt_with_feedback
        prompt = formatter.create_prompt_with_feedback("Original prompt", "Test feedback")
        self.assertEqual(prompt, "Original prompt\n\nPrevious attempt feedback:\nTest feedback")

    def test_chain_executor(self):
        """Test ChainExecutor class."""
        # Create mock components
        generator = Mock(spec=Generator)
        generator.generate.return_value = "Test output"

        validator = Mock(spec=Validator)
        validator.validate.return_value = ValidationResult(
            output="Test output",
            rule_results=[RuleResult(passed=True, message="Passed")],
            all_passed=True,
        )

        improver = Mock(spec=Improver)

        # Create executor
        executor = ChainExecutor(generator=generator, validator=validator, improver=improver)

        # Test execute with passing validation
        output, validation_result, critique_details = executor.execute("Test prompt")

        # Verify results
        self.assertEqual(output, "Test output")
        self.assertTrue(validation_result.all_passed)
        self.assertIsNone(critique_details)

        # Verify components were called
        generator.generate.assert_called_once_with("Test prompt")
        validator.validate.assert_called_once_with("Test output")
        improver.improve.assert_not_called()

    def test_retry_manager(self):
        """Test RetryManager class."""
        # Create mock executor
        executor = Mock(spec=ChainExecutor)
        executor.execute.return_value = (
            "Test output",
            ValidationResult(
                output="Test output",
                rule_results=[RuleResult(passed=True, message="Passed")],
                all_passed=True,
            ),
            None,
        )

        # Create retry manager
        retry_manager = RetryManager(max_attempts=3)

        # Test run_with_retries
        result = retry_manager.run_with_retries(executor, "Test prompt")

        # Verify results
        self.assertEqual(result.output, "Test output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)

        # Verify executor was called
        executor.execute.assert_called_once_with("Test prompt")

    def test_chain_orchestrator(self):
        """Test ChainOrchestrator class."""
        # Create mock components
        generator = Mock(spec=Generator)
        validator = Mock(spec=Validator)
        improver = Mock(spec=Improver)
        retry_manager = Mock(spec=RetryManager)
        retry_manager.run_with_retries.return_value = ChainResult(
            output="Test output", rule_results=[RuleResult(passed=True, message="Passed")]
        )

        # Create orchestrator
        orchestrator = ChainOrchestrator(
            generator=generator, validator=validator, improver=improver, retry_manager=retry_manager
        )

        # Test run
        result = orchestrator.run("Test prompt")

        # Verify results
        self.assertEqual(result.output, "Test output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)

        # Verify retry_manager was called with correct executor
        retry_manager.run_with_retries.assert_called_once()

        # Get the executor that was passed to run_with_retries
        executor = retry_manager.run_with_retries.call_args[0][0]

        # Verify executor has correct components
        self.assertEqual(executor.generator, generator)
        self.assertEqual(executor.validator, validator)
        self.assertEqual(executor.improver, improver)


if __name__ == "__main__":
    unittest.main()
