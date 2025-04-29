"""
Tests for the Chain module of Sifaka.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.chain import Chain, ChainResult
from sifaka.rules import Rule, RuleResult
from sifaka.critics.prompt import CriticMetadata


class TestChain:
    """Test suite for Chain class."""

    def test_chain_initialization(self, mock_model_provider, mock_rule):
        """Test Chain initialization."""
        chain = Chain(model=mock_model_provider, rules=[mock_rule])
        assert chain.model == mock_model_provider
        assert chain.rules == [mock_rule]
        assert chain.critic is None
        assert chain.max_attempts == 3

    def test_chain_run_all_rules_pass(self, mock_model_provider, mock_rule):
        """Test Chain run with all rules passing."""
        chain = Chain(model=mock_model_provider, rules=[mock_rule])

        # Configure mock
        mock_model_provider.generate.return_value = "Generated output"
        mock_rule.validate.return_value = RuleResult(passed=True, message="Rule passed")

        # Run chain
        result = chain.run("Input prompt")

        # Verify model was called
        mock_model_provider.generate.assert_called_once_with("Input prompt")

        # Verify rule was called
        mock_rule.validate.assert_called_once_with("Generated output")

        # Verify result
        assert isinstance(result, ChainResult)
        assert result.output == "Generated output"
        assert len(result.rule_results) == 1
        assert result.rule_results[0].passed is True
        assert result.critique_details is None

    def test_chain_run_rule_fails_no_critic(self, mock_model_provider, mock_failing_rule):
        """Test Chain run with failing rule and no critic."""
        chain = Chain(model=mock_model_provider, rules=[mock_failing_rule])

        # Configure mock
        mock_model_provider.generate.return_value = "Generated output"

        # Run chain and expect ValueError
        with pytest.raises(ValueError) as excinfo:
            chain.run("Input prompt")

        # Verify error message
        assert "Validation failed" in str(excinfo.value)
        assert "Validation failed" in str(excinfo.value)

        # Verify model was called
        mock_model_provider.generate.assert_called_once_with("Input prompt")

        # Verify rule was called
        mock_failing_rule.validate.assert_called_once_with("Generated output")

    def test_chain_run_with_critic_improvement(self, mock_model_provider, mock_failing_rule):
        """Test Chain run with failing rule and critic improving the output."""
        # Create mock critic
        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticMetadata(
            feedback="This output needs improvement",
            improved_output="Improved output",
            improvement_score=0.8,
        )

        chain = Chain(
            model=mock_model_provider, rules=[mock_failing_rule], critic=mock_critic, max_attempts=2
        )

        # Configure mocks
        # First attempt fails, second succeeds
        mock_model_provider.generate.side_effect = ["Generated output", "Improved output"]
        mock_failing_rule.validate.side_effect = [
            RuleResult(passed=False, message="Rule failed"),
            RuleResult(passed=True, message="Rule passed"),
        ]

        # Run chain
        result = chain.run("Input prompt")

        # Verify model was called twice
        assert mock_model_provider.generate.call_count == 2
        mock_model_provider.generate.assert_any_call("Input prompt")
        mock_model_provider.generate.assert_any_call(
            "Input prompt\n\nPrevious attempt feedback:\nThis output needs improvement"
        )

        # Verify rule was called twice
        assert mock_failing_rule.validate.call_count == 2

        # Verify critic was called
        mock_critic.critique.assert_called_once_with("Generated output")

        # Verify result
        assert isinstance(result, ChainResult)
        assert result.output == "Improved output"
        assert len(result.rule_results) == 1
        assert result.rule_results[0].passed is True
        assert result.critique_details is not None

    def test_chain_run_max_attempts_exceeded(self, mock_model_provider, mock_failing_rule):
        """Test Chain run with max attempts exceeded."""
        # Create mock critic
        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticMetadata(
            feedback="This output needs improvement",
            improved_output="Improved output",
            improvement_score=0.8,
        )

        chain = Chain(
            model=mock_model_provider, rules=[mock_failing_rule], critic=mock_critic, max_attempts=2
        )

        # Configure mocks - all attempts fail
        mock_model_provider.generate.return_value = "Generated output"
        mock_failing_rule.validate.return_value = RuleResult(passed=False, message="Rule failed")

        # Run chain and expect ValueError
        with pytest.raises(ValueError) as excinfo:
            chain.run("Input prompt")

        # Verify error message
        assert "Validation failed after 2 attempts" in str(excinfo.value)

        # Verify model was called twice
        assert mock_model_provider.generate.call_count == 2

        # Verify rule was called twice
        assert mock_failing_rule.validate.call_count == 2

        # Verify critic was called once
        mock_critic.critique.assert_called_once()

    def test_chain_run_multiple_rules(self, mock_model_provider):
        """Test Chain run with multiple rules."""
        # Create two mock rules
        rule1 = MagicMock(spec=Rule)
        rule1.validate.return_value = RuleResult(passed=True, message="Rule 1 passed")

        rule2 = MagicMock(spec=Rule)
        rule2.validate.return_value = RuleResult(passed=True, message="Rule 2 passed")

        chain = Chain(model=mock_model_provider, rules=[rule1, rule2])

        # Configure model mock
        mock_model_provider.generate.return_value = "Generated output"

        # Run chain
        result = chain.run("Input prompt")

        # Verify model was called
        mock_model_provider.generate.assert_called_once_with("Input prompt")

        # Verify rules were called
        rule1.validate.assert_called_once_with("Generated output")
        rule2.validate.assert_called_once_with("Generated output")

        # Verify result
        assert isinstance(result, ChainResult)
        assert result.output == "Generated output"
        assert len(result.rule_results) == 2
        assert result.rule_results[0].passed is True
        assert result.rule_results[1].passed is True

    def test_chain_run_mixed_rule_results(self, mock_model_provider):
        """Test Chain run with mixed rule results (one passes, one fails)."""
        # Create two mock rules
        rule1 = MagicMock(spec=Rule)
        rule1.validate.return_value = RuleResult(passed=True, message="Rule 1 passed")

        rule2 = MagicMock(spec=Rule)
        rule2.validate.return_value = RuleResult(passed=False, message="Rule 2 failed")

        # Create mock critic
        mock_critic = MagicMock()
        mock_critic.critique.return_value = {"feedback": "This output needs improvement"}

        chain = Chain(model=mock_model_provider, rules=[rule1, rule2], critic=mock_critic)

        # Configure model mock - second attempt succeeds
        mock_model_provider.generate.side_effect = ["Generated output", "Improved output"]

        # Configure rule2 mock - second attempt passes
        rule2.validate.side_effect = [
            RuleResult(passed=False, message="Rule 2 failed"),
            RuleResult(passed=True, message="Rule 2 passed"),
        ]

        # Run chain
        result = chain.run("Input prompt")

        # Verify model was called twice
        assert mock_model_provider.generate.call_count == 2

        # Verify rules were called
        assert rule1.validate.call_count == 2
        assert rule2.validate.call_count == 2

        # Verify critic was called
        mock_critic.critique.assert_called_once_with("Generated output")

        # Verify result
        assert isinstance(result, ChainResult)
        assert result.output == "Improved output"
        assert len(result.rule_results) == 2
        assert result.rule_results[0].passed is True
        assert result.rule_results[1].passed is True
