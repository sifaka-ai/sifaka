"""
Tests for the ToxicityValidator.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from sifaka.validators import ToxicityValidator
from sifaka.types import ValidationResult


# Mock the Detoxify class
class MockDetoxify:
    def __init__(self, model_name="original"):
        self.model_name = model_name

    def predict(self, text):
        # Return mock toxicity scores based on keywords in the text
        scores = {
            "toxicity": 0.1,
            "severe_toxicity": 0.05,
            "obscene": 0.1,
            "threat": 0.05,
            "insult": 0.1,
            "identity_attack": 0.05,
        }

        # Increase scores based on keywords
        if "hate" in text.lower():
            scores["toxicity"] = 0.7
            scores["severe_toxicity"] = 0.3
            scores["insult"] = 0.6

        if "kill" in text.lower() or "die" in text.lower():
            scores["toxicity"] = 0.8
            scores["severe_toxicity"] = 0.6
            scores["threat"] = 0.8

        if "stupid" in text.lower() or "idiot" in text.lower():
            scores["toxicity"] = 0.6
            scores["insult"] = 0.7

        # Special case for the test_check_all_categories test
        # This should only trigger identity_attack but keep toxicity low
        if "race and gender" in text.lower():
            scores["identity_attack"] = 0.7
            # Make sure primary categories stay low
            scores["toxicity"] = 0.2
            scores["severe_toxicity"] = 0.1
            scores["threat"] = 0.1
        elif any(word in text.lower() for word in ["race", "gender", "religion"]):
            scores["identity_attack"] = 0.7
            scores["toxicity"] = 0.7

        return scores


# Mock the detoxify import
@pytest.fixture(autouse=True)
def mock_detoxify():
    # Create a mock module
    mock_module = MagicMock()
    mock_module.Detoxify = MockDetoxify

    # Add the mock to sys.modules
    with patch.dict(sys.modules, {"detoxify": mock_module}):
        yield


def test_init():
    """Test initialization of ToxicityValidator."""
    validator = ToxicityValidator(
        threshold=0.4,
        severe_toxic_threshold=0.6,
        threat_threshold=0.5,
        check_all_categories=True,
        category_thresholds={"insult": 0.3},
        model_name="unbiased",
    )

    assert validator.threshold == 0.4
    assert validator.severe_toxic_threshold == 0.6
    assert validator.threat_threshold == 0.5
    assert validator.check_all_categories is True
    assert validator.category_thresholds == {"insult": 0.3}
    assert validator.model_name == "unbiased"


def test_get_threshold_for_category():
    """Test the get_threshold_for_category method."""
    validator = ToxicityValidator(
        threshold=0.5,
        severe_toxic_threshold=0.7,
        threat_threshold=0.6,
        category_thresholds={"insult": 0.4},
    )

    # Test custom threshold
    assert validator.get_threshold_for_category("insult") == 0.4

    # Test special thresholds
    assert validator.get_threshold_for_category("severe_toxicity") == 0.7
    assert validator.get_threshold_for_category("threat") == 0.6

    # Test default threshold
    assert validator.get_threshold_for_category("toxicity") == 0.5
    assert validator.get_threshold_for_category("obscene") == 0.5
    assert validator.get_threshold_for_category("identity_attack") == 0.5


def test_validate_non_toxic():
    """Test validation of non-toxic text."""
    validator = ToxicityValidator()

    result = validator.validate("This is a friendly and helpful message.")

    assert result.passed is True
    assert result.score > 0.8  # Should have a high score
    assert "passed toxicity validation" in result.message.lower()
    assert len(result.issues) == 0
    assert len(result.suggestions) == 0
    assert "scores" in result.metadata


def test_validate_toxic():
    """Test validation of toxic text."""
    validator = ToxicityValidator()

    result = validator.validate("I hate you and wish you would die.")

    assert result.passed is False
    assert result.score < 0.5  # Should have a low score
    assert "toxic content" in result.message.lower()
    assert len(result.issues) > 0
    assert len(result.suggestions) > 0
    assert "scores" in result.metadata


def test_validate_empty():
    """Test validation of empty text."""
    validator = ToxicityValidator()

    result = validator.validate("")

    assert result.passed is True
    assert result.score == 1.0
    assert "empty text" in result.message.lower()
    assert len(result.issues) == 0
    assert len(result.suggestions) == 0
    assert result.metadata == {"scores": {}}


def test_check_all_categories():
    """Test validation with check_all_categories."""
    # With check_all_categories=True, it should fail if any category fails
    validator_strict = ToxicityValidator(threshold=0.5, check_all_categories=True)

    # With check_all_categories=False (default), it should only fail for primary categories
    validator_lenient = ToxicityValidator(threshold=0.5, check_all_categories=False)

    # This text has high identity_attack but low on primary categories
    result_strict = validator_strict.validate("Let's discuss race and gender issues.")
    result_lenient = validator_lenient.validate("Let's discuss race and gender issues.")

    # Strict validator should fail
    assert result_strict.passed is False

    # Lenient validator should pass (as identity_attack isn't a primary category)
    assert result_lenient.passed is True


def test_custom_thresholds():
    """Test validation with custom thresholds."""
    # Stricter validator
    validator_strict = ToxicityValidator(threshold=0.3)

    # More lenient validator
    validator_lenient = ToxicityValidator(threshold=0.7)

    # Mildly toxic text
    text = "This is a bit stupid."

    # Should fail with strict threshold
    result_strict = validator_strict.validate(text)
    assert result_strict.passed is False

    # Should pass with lenient threshold
    result_lenient = validator_lenient.validate(text)
    assert result_lenient.passed is True


def test_exception_handling():
    """Test handling of exceptions during validation."""
    validator = ToxicityValidator()

    # Mock the model.predict method to raise an exception
    with patch.object(validator.model, "predict", side_effect=Exception("Test error")):
        result = validator.validate("Test text")

        assert result.passed is False
        assert "error" in result.message.lower()
        assert len(result.issues) == 1
        assert "error occurred" in result.issues[0].lower()
        assert len(result.suggestions) == 1
        assert "error" in result.metadata
