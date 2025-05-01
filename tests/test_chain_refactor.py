"""Tests for the refactored Chain implementation."""

import unittest
from unittest.mock import Mock, MagicMock

from sifaka.chain import Chain, ChainResult
from sifaka.critics import PromptCritic
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models.base import ModelProvider
from sifaka.rules import Rule, RuleResult
from sifaka.validation import Validator, ValidationResult


class TestChainRefactor(unittest.TestCase):
    """Tests for the refactored Chain implementation."""

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
        self.critic = Mock(spec=PromptCritic)
        self.critic.critique.return_value = {"feedback": "Needs improvement", "score": 0.5}

    def test_chain_with_passing_rules(self):
        """Test chain with rules that pass validation."""
        # Create chain with only passing rules
        chain = Chain(
            model=self.model,
            rules=[self.passing_rule],
            critic=None,
            max_attempts=3
        )

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
        chain = Chain(
            model=self.model,
            rules=[self.failing_rule],
            critic=None,
            max_attempts=3
        )

        # Run chain - should raise ValueError
        with self.assertRaises(ValueError):
            chain.run("Test prompt")

        # Verify model and rule were called
        self.model.generate.assert_called_once_with("Test prompt")
        self.failing_rule.validate.assert_called_once_with("Test output")

    def test_chain_with_failing_rules_and_critic(self):
        """Test chain with rules that fail validation and a critic."""
        # Create chain with failing rules and a critic
        chain = Chain(
            model=self.model,
            rules=[self.failing_rule],
            critic=self.critic,
            max_attempts=3
        )

        # Mock chain components to make test pass on second attempt
        # First attempt: validation fails
        # Second attempt: validation passes
        self.failing_rule.validate.side_effect = [
            RuleResult(passed=False, message="Failed first time"),
            RuleResult(passed=True, message="Passed second time")
        ]

        # Run chain
        result = chain.run("Test prompt")

        # Verify results
        self.assertEqual(result.output, "Test output")  # Second attempt output
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)  # Second attempt passed

        # Verify critic was called
        self.critic.critique.assert_called_once_with("Test output")

        # Verify model was called twice with different prompts
        self.assertEqual(self.model.generate.call_count, 2)
        self.model.generate.assert_any_call("Test prompt")  # First call
        # Second call should include feedback
        self.model.generate.assert_any_call("Test prompt\n\nPrevious attempt feedback:\nNeeds improvement")

    def test_max_attempts(self):
        """Test chain respects max_attempts setting."""
        # Create chain with fewer max_attempts
        chain = Chain(
            model=self.model,
            rules=[self.failing_rule],
            critic=self.critic,
            max_attempts=2
        )

        # Always fail validation
        self.failing_rule.validate.return_value = RuleResult(passed=False, message="Always failing")

        # Run chain - should raise ValueError after max_attempts
        with self.assertRaises(ValueError) as context:
            chain.run("Test prompt")

        # Verify error message includes attempt count
        self.assertIn("Validation failed after 2 attempts", str(context.exception))

        # Verify model was called twice
        self.assertEqual(self.model.generate.call_count, 2)

        # Verify critic was called
        self.critic.critique.assert_called_once_with("Test output")


if __name__ == "__main__":
    unittest.main()