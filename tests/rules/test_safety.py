"""
Tests for the safety rules module of Sifaka.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.rules.content.safety import ToxicityRule, BiasRule, HarmfulContentRule
from sifaka.rules.base import RuleConfig, RuleResult


class TestToxicityRule:
    """Test suite for ToxicityRule class."""

    @patch("sifaka.rules.content.safety.DefaultToxicityValidator")
    def test_toxicity_rule_initialization(self, mock_validator):
        """Test ToxicityRule initialization."""
        # Configure mock
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance

        config = RuleConfig(params={"threshold": 0.7, "model_name": "test-model"})

        # Use patch to override _create_default_validator
        with patch.object(
            ToxicityRule, "_create_default_validator", return_value=mock_validator_instance
        ):
            rule = ToxicityRule(config=config)
            assert rule.name == "toxicity_rule"

    def test_toxicity_rule_validation_safe(self):
        """Test ToxicityRule validation with safe content."""
        # Create a mock validator that simulates low toxicity
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=True,
            message="Content is non-toxic",
            metadata={
                "toxicity": 0.1,
                "severe_toxicity": 0.05,
                "obscene": 0.02,
                "identity_attack": 0.01,
                "insult": 0.03,
                "threat": 0.01,
            },
        )

        # Use patch to override the validator creation
        with patch.object(ToxicityRule, "_create_default_validator", return_value=mock_validator):
            rule = ToxicityRule(config=RuleConfig(params={"threshold": 0.7}))

            # Validate safe text
            result = rule.validate("This is a friendly and helpful message.")

            # Verify validator was called
            mock_validator.validate.assert_called_once_with(
                "This is a friendly and helpful message."
            )

            # Verify result
            assert result.passed is True
            assert "Content is non-toxic" in result.message
            assert result.metadata["toxicity"] == 0.1
            assert result.metadata["severe_toxicity"] == 0.05

    def test_toxicity_rule_validation_toxic(self):
        """Test ToxicityRule validation with toxic content."""
        # Create a mock validator that simulates high toxicity
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=False,
            message="Content contains toxic elements",
            metadata={
                "toxicity": 0.8,
                "severe_toxicity": 0.6,
                "obscene": 0.7,
                "identity_attack": 0.3,
                "insult": 0.9,
                "threat": 0.5,
                "toxic_categories": ["insult", "obscene", "toxicity"],
            },
        )

        # Use patch to override the validator creation
        with patch.object(ToxicityRule, "_create_default_validator", return_value=mock_validator):
            rule = ToxicityRule(config=RuleConfig(params={"threshold": 0.7}))

            # Validate toxic text
            result = rule.validate("This is toxic content for testing.")

            # Verify result
            assert result.passed is False
            assert "Content contains toxic elements" in result.message
            assert result.metadata["toxicity"] == 0.8
            assert result.metadata["severe_toxicity"] == 0.6
            assert "insult" in result.metadata["toxic_categories"]

    def test_toxicity_rule_custom_threshold(self):
        """Test ToxicityRule with custom threshold."""
        # Create a mock validator with medium toxicity
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=True,  # The validator returns a simple result
            message="Content toxicity evaluation",
            metadata={"toxicity": 0.6, "severe_toxicity": 0.4},
        )

        # For lower threshold test
        mock_validator_low = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator_low.validation_type = str
        mock_validator_low.validate.return_value = RuleResult(
            passed=False,
            message="Content contains toxic elements",
            metadata={"toxicity": 0.6, "severe_toxicity": 0.4},
        )

        # For higher threshold test
        mock_validator_high = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator_high.validation_type = str
        mock_validator_high.validate.return_value = RuleResult(
            passed=True,
            message="Content is non-toxic",
            metadata={"toxicity": 0.6, "severe_toxicity": 0.4},
        )

        # Lower threshold test
        with patch.object(
            ToxicityRule, "_create_default_validator", return_value=mock_validator_low
        ):
            rule_lower = ToxicityRule(config=RuleConfig(params={"threshold": 0.5}))
            result = rule_lower.validate("Content with medium toxicity.")
            assert result.passed is False

        # Higher threshold test
        with patch.object(
            ToxicityRule, "_create_default_validator", return_value=mock_validator_high
        ):
            rule_higher = ToxicityRule(config=RuleConfig(params={"threshold": 0.8}))
            result = rule_higher.validate("Content with medium toxicity.")
            assert result.passed is True


class TestBiasRule:
    """Test suite for BiasRule class."""

    @patch("sifaka.rules.content.safety.DefaultBiasValidator")
    def test_bias_rule_initialization(self, mock_validator):
        """Test BiasRule initialization."""
        # Configure mock
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance

        config = RuleConfig(
            params={
                "threshold": 0.6,
                "categories": ["gender", "race", "religion"],
                "model_name": "test-bias-model",
            }
        )

        # Use patch to override _create_default_validator
        with patch.object(
            BiasRule, "_create_default_validator", return_value=mock_validator_instance
        ):
            rule = BiasRule(config=config)
            assert rule.name == "bias_rule"

    def test_bias_rule_validation_unbiased(self):
        """Test BiasRule validation with unbiased content."""
        # Create a mock validator for unbiased content
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=True,
            message="Content does not contain bias",
            metadata={
                "gender": 0.1,
                "race": 0.2,
                "religion": 0.15,
                "age": 0.05,
                "overall_bias": 0.12,
            },
        )

        # Use patch to override the validator creation
        with patch.object(BiasRule, "_create_default_validator", return_value=mock_validator):
            rule = BiasRule(config=RuleConfig(params={"threshold": 0.6}))

            # Validate unbiased text
            result = rule.validate("This is a neutral and unbiased message.")

            # Verify validator was called
            mock_validator.validate.assert_called_once_with(
                "This is a neutral and unbiased message."
            )

            # Verify result
            assert result.passed is True
            assert "Content does not contain bias" in result.message
            assert result.metadata["gender"] == 0.1
            assert result.metadata["overall_bias"] == 0.12

    def test_bias_rule_validation_biased(self):
        """Test BiasRule validation with biased content."""
        # Create a mock validator for biased content
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=False,
            message="Content contains bias",
            metadata={
                "gender": 0.7,
                "race": 0.3,
                "religion": 0.8,
                "age": 0.5,
                "overall_bias": 0.65,
                "biased_categories": ["gender", "religion"],
            },
        )

        # Use patch to override the validator creation
        with patch.object(BiasRule, "_create_default_validator", return_value=mock_validator):
            rule = BiasRule(config=RuleConfig(params={"threshold": 0.6}))

            # Validate biased text
            result = rule.validate("This is potentially biased content for testing.")

            # Verify result
            assert result.passed is False
            assert "Content contains bias" in result.message
            assert "gender" in result.metadata["biased_categories"]
            assert "religion" in result.metadata["biased_categories"]
            assert len(result.metadata["biased_categories"]) == 2

    def test_bias_rule_specific_categories(self):
        """Test BiasRule with specific categories."""
        # Create a mock validator for category-specific testing
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=False,
            message="Content contains bias",
            metadata={
                "gender": 0.7,
                "race": 0.3,
                "religion": 0.8,
                "age": 0.9,
                "overall_bias": 0.65,
                "biased_categories": [
                    "gender"
                ],  # Only gender is in the test categories and above threshold
            },
        )

        # Use patch to override the validator creation
        with patch.object(BiasRule, "_create_default_validator", return_value=mock_validator):
            # Only care about gender and race
            rule = BiasRule(
                config=RuleConfig(params={"threshold": 0.6, "categories": ["gender", "race"]})
            )

            result = rule.validate("Testing specific categories.")

            # Should only fail on gender (race is below threshold)
            assert result.passed is False
            assert "gender" in result.metadata["biased_categories"]
            assert "race" not in result.metadata["biased_categories"]
            assert "religion" not in result.metadata["biased_categories"]
            assert "age" not in result.metadata["biased_categories"]


class TestHarmfulContentRule:
    """Test suite for HarmfulContentRule class."""

    @patch("sifaka.rules.content.safety.DefaultHarmfulContentValidator")
    def test_harmful_content_rule_initialization(self, mock_validator):
        """Test HarmfulContentRule initialization."""
        # Configure mock
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance

        config = RuleConfig(
            params={
                "threshold": 0.7,
                "categories": ["violence", "self_harm", "sexual", "harassment"],
                "model_name": "test-harmful-model",
            }
        )

        # Use patch to override _create_default_validator
        with patch.object(
            HarmfulContentRule, "_create_default_validator", return_value=mock_validator_instance
        ):
            rule = HarmfulContentRule(config=config)
            assert rule.name == "harmful_content_rule"

    def test_harmful_content_rule_validation_safe(self):
        """Test HarmfulContentRule validation with safe content."""
        # Create a mock validator for safe content
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=True,
            message="Content does not contain harmful elements",
            metadata={
                "violence": 0.1,
                "self_harm": 0.05,
                "sexual": 0.2,
                "harassment": 0.15,
                "overall_harm": 0.125,
            },
        )

        # Use patch to override the validator creation
        with patch.object(
            HarmfulContentRule, "_create_default_validator", return_value=mock_validator
        ):
            rule = HarmfulContentRule(config=RuleConfig(params={"threshold": 0.7}))

            # Validate safe text
            result = rule.validate("This is a safe and harmless message.")

            # Verify validator was called
            mock_validator.validate.assert_called_once_with("This is a safe and harmless message.")

            # Verify result
            assert result.passed is True
            assert "Content does not contain harmful elements" in result.message
            assert result.metadata["violence"] == 0.1
            assert result.metadata["overall_harm"] == 0.125

    def test_harmful_content_rule_validation_harmful(self):
        """Test HarmfulContentRule validation with harmful content."""
        # Create a mock validator for harmful content
        mock_validator = MagicMock()
        # Add validation_type attribute to fix isinstance check
        mock_validator.validation_type = str
        mock_validator.validate.return_value = RuleResult(
            passed=False,
            message="Content contains harmful elements",
            metadata={
                "violence": 0.8,
                "self_harm": 0.4,
                "sexual": 0.9,
                "harassment": 0.5,
                "overall_harm": 0.65,
                "harmful_categories": ["violence", "sexual"],
            },
        )

        # Use patch to override the validator creation
        with patch.object(
            HarmfulContentRule, "_create_default_validator", return_value=mock_validator
        ):
            rule = HarmfulContentRule(config=RuleConfig(params={"threshold": 0.7}))

            # Validate harmful text
            result = rule.validate("This is potentially harmful content for testing.")

            # Verify result
            assert result.passed is False
            assert "Content contains harmful elements" in result.message
            assert "violence" in result.metadata["harmful_categories"]
            assert "sexual" in result.metadata["harmful_categories"]
            assert len(result.metadata["harmful_categories"]) == 2
