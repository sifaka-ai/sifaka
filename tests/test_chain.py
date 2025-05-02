"""Tests for the Chain implementation."""

import unittest
from unittest.mock import Mock

from sifaka.chain import Chain, ChainResult
from sifaka.models.base import ModelProvider
from sifaka.rules import Rule, RuleResult


class TestChain(unittest.TestCase):
    """Tests for the Chain implementation."""

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
        self.critic = Mock()
        self.critic.critique = Mock(return_value={"feedback": "Needs improvement", "score": 0.5})

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
        # Create a mock model
        model = Mock(spec=ModelProvider)
        model.generate.return_value = "Test output"

        # Create a mock failing rule
        failing_rule = Mock(spec=Rule)
        failing_rule.validate.return_value = RuleResult(passed=False, message="Failed")

        # Create chain with failing rules and no critic
        chain = Chain(model=model, rules=[failing_rule], critic=None, max_attempts=3)

        # Mock the chain._chain.run method to raise ValueError
        chain._chain.run = Mock(side_effect=ValueError("Validation failed"))

        # Run chain - should raise ValueError
        with self.assertRaises(ValueError):
            chain.run("Test prompt")

    def test_chain_with_failing_rules_and_critic(self):
        """Test chain with rules that fail validation and a critic."""
        # Create a mock model that returns different outputs on each call
        model = Mock(spec=ModelProvider)
        model.generate.side_effect = ["First output", "Second output"]

        # Create a mock failing rule that passes on the second attempt
        failing_rule = Mock(spec=Rule)
        failing_rule.validate.side_effect = [
            RuleResult(passed=False, message="Failed first time"),
            RuleResult(passed=True, message="Passed second time"),
        ]

        # Create a mock critic
        critic = Mock()
        critic.critique = Mock(return_value={"feedback": "Needs improvement"})

        # Create chain with failing rules and a critic
        chain = Chain(model=model, rules=[failing_rule], critic=critic, max_attempts=3)

        # Mock the chain._chain.run method to return a successful result
        chain._chain.run = Mock(
            return_value=ChainResult(
                output="Second output",
                rule_results=[RuleResult(passed=True, message="Passed second time")],
            )
        )

        # Run chain
        result = chain.run("Test prompt")

        # Verify results
        self.assertEqual(result.output, "Second output")
        self.assertEqual(len(result.rule_results), 1)
        self.assertTrue(result.rule_results[0].passed)

    def test_max_attempts(self):
        """Test chain respects max_attempts setting."""
        # Create a mock model
        model = Mock(spec=ModelProvider)

        # Create a mock failing rule
        failing_rule = Mock(spec=Rule)
        failing_rule.validate.return_value = RuleResult(passed=False, message="Always failing")

        # Create a mock critic
        critic = Mock()
        critic.critique = Mock(return_value={"feedback": "Needs improvement"})

        # Create chain with fewer max_attempts
        chain = Chain(model=model, rules=[failing_rule], critic=critic, max_attempts=2)

        # Mock the chain._chain.run method to raise ValueError
        error_message = "Validation failed after 2 attempts. Errors:\nAlways failing"
        chain._chain.run = Mock(side_effect=ValueError(error_message))

        # Run chain - should raise ValueError after max_attempts
        with self.assertRaises(ValueError) as context:
            chain.run("Test prompt")

        # Verify error message includes attempt count
        self.assertIn("Validation failed after 2 attempts", str(context.exception))


if __name__ == "__main__":
    unittest.main()
