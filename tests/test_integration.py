"""
Integration tests for Sifaka components working together.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.chain import Chain
from sifaka.rules import LengthRule, ProhibitedContentRule, FormatRule, RuleConfig, RuleResult
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig, CriticMetadata


class TestSifakaIntegration:
    """Integration tests for Sifaka components."""

    @patch("sifaka.critics.prompt.PromptCritic")
    def test_chain_with_multiple_rules(self, mock_critic_class, mock_model_provider):
        """Test a chain with multiple rule types."""
        # Create rules
        length_rule = LengthRule(
            name="length_check",
            config=RuleConfig(params={"min_length": 10, "max_length": 1000, "unit": "characters"}),
        )

        content_rule = ProhibitedContentRule(
            name="content_filter",
            config=RuleConfig(params={"terms": ["bad", "inappropriate"], "case_sensitive": False}),
        )

        # Configure mocks
        mock_model_provider.generate.return_value = (
            "This is a good test message that passes all rules."
        )

        # Create chain
        chain = Chain(model=mock_model_provider, rules=[length_rule, content_rule])

        # Run chain
        result = chain.run("Generate a test message")

        # Verify
        assert result.output == "This is a good test message that passes all rules."
        assert len(result.rule_results) == 2
        assert all(r.passed for r in result.rule_results)

    @patch("sifaka.critics.prompt.PromptCritic")
    def test_chain_with_critic_improvement(
        self, mock_critic_class, mock_model_provider, mock_failing_rule
    ):
        """Test a chain with a critic that improves content."""
        # Configure mock critic
        mock_critic = MagicMock()
        # Use CriticMetadata instead of dict
        mock_critic.critique.return_value = CriticMetadata(
            feedback="The content needs to be longer.",
            improved_output="This is an improved version of the content that is much longer and meets the requirements.",
            improvement_score=0.8,
        )
        mock_critic_class.return_value = mock_critic

        # Create critic config
        critic_config = PromptCriticConfig(
            name="test_critic",
            description="Test critic for improving content",
            system_prompt="You are a helpful critic that improves content.",
        )

        # First generate a short message that fails the length rule
        # Then generate an improved message that passes
        mock_model_provider.generate.side_effect = [
            "Short msg.",
            "This is an improved version of the content that is much longer and meets the requirements.",
        ]

        # Initial validation fails, then passes after improvement
        mock_failing_rule.validate.side_effect = [
            RuleResult(passed=False, message="Content is too short", metadata={}),
            RuleResult(passed=True, message="Content is long enough", metadata={}),
        ]

        # Create chain
        chain = Chain(
            model=mock_model_provider, rules=[mock_failing_rule], critic=mock_critic, max_attempts=2
        )

        # Run chain
        result = chain.run("Generate content")

        # Verify
        assert "improved version" in result.output
        assert result.critique_details is not None

    def test_full_pipeline_with_multiple_rules(self, mock_model_provider):
        """Test a complete pipeline with multiple rules of different types."""
        # Create rules
        length_rule = LengthRule(
            name="length_check",
            config=RuleConfig(params={"min_length": 10, "max_length": 1000, "unit": "characters"}),
        )

        content_rule = ProhibitedContentRule(
            name="content_filter",
            config=RuleConfig(
                params={
                    "terms": ["bad", "inappropriate", "offensive"],
                    "case_sensitive": False,
                }
            ),
        )

        format_rule = FormatRule(
            name="format_check",
            format_type="markdown",
            config=RuleConfig(
                params={
                    "required_elements": ["#", "-", "*"],
                    "min_elements": 1,
                }
            ),
        )

        # Configure mock model to return different responses based on input
        def mock_generate(prompt):
            if "feedback" in prompt:
                # Second attempt after feedback
                return """# Improved Response

This is a much better response that follows all the rules
and guidelines. It has proper length, markdown formatting,
and no prohibited content.

- Point 1
- Point 2
                """
            else:
                # First attempt (too short)
                return "Short bad response."

        mock_model_provider.generate.side_effect = mock_generate

        # Create mock critic
        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticMetadata(
            feedback="The content is too short and contains prohibited terms.",
            improved_output="",
            improvement_score=0.3,
        )

        # Create chain with all rules
        chain = Chain(
            model=mock_model_provider,
            rules=[length_rule, content_rule, format_rule],
            critic=mock_critic,
            max_attempts=2,
        )

        # Run chain
        result = chain.run("Generate a test message")

        # Verify final output meets requirements
        assert "Improved Response" in result.output
        assert "markdown formatting" in result.output
        assert len(result.rule_results) == 3
        assert all(r.passed for r in result.rule_results)

    def test_rule_combination_with_different_priorities(self, mock_model_provider):
        """Test rules with different priorities in a chain."""
        # Create rules with different priorities
        high_priority_rule = LengthRule(
            name="critical_length",
            config=RuleConfig(params={"min_length": 20, "max_length": 100}, priority="HIGH"),
        )

        medium_priority_rule = ProhibitedContentRule(
            name="content_check",
            config=RuleConfig(params={"terms": ["test"]}, priority="MEDIUM"),
        )

        low_priority_rule = FormatRule(
            name="format_check",
            format_type="plain_text",
            config=RuleConfig(params={"min_length": 10}, priority="LOW"),
        )

        # Configure mock model
        mock_model_provider.generate.return_value = (
            "This is a valid response without prohibited terms."
        )

        # Create chain with rules in mixed priority order
        chain = Chain(
            model=mock_model_provider,
            rules=[low_priority_rule, high_priority_rule, medium_priority_rule],
        )

        # Run chain
        result = chain.run("Generate a test message")

        # Verify all rules passed
        assert result.output == "This is a valid response without prohibited terms."
        assert len(result.rule_results) == 3
        assert all(r.passed for r in result.rule_results)
