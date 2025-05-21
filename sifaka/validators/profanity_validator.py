"""
Profanity validator for Sifaka.

This module provides a validator that checks if text contains profanity
using the ProfanityClassifier.
"""

import logging
from typing import Any, Dict, List, Optional

from sifaka.classifiers.profanity_classifier import ProfanityClassifier
from sifaka.validators.base_validator import BaseValidator

# Configure logger
logger = logging.getLogger(__name__)


class ProfanityValidator(BaseValidator):
    """
    Validator that checks if text contains profanity.

    This validator uses the ProfanityClassifier to detect profanity in text.
    It can be configured with custom words and a confidence threshold.
    """

    def __init__(
        self,
        custom_words: Optional[List[str]] = None,
        threshold: float = 0.5,
        name: Optional[str] = None,
        **options: Any,
    ):
        """
        Initialize the profanity validator.

        Args:
            custom_words: Optional list of additional profane words to detect.
            threshold: Confidence threshold for considering text profane.
            name: Optional name for the validator.
            **options: Additional options for the validator.
        """
        super().__init__(name=name or "ProfanityValidator", **options)

        # Create classifier
        self.classifier = ProfanityClassifier(custom_words=custom_words)
        self.threshold = threshold

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with threshold={threshold}, "
            f"custom_words={len(custom_words) if custom_words else 0}"
        )

    def _validate(self, text: str) -> Dict[str, Any]:
        """
        Validate text against profanity.

        Args:
            text: The text to validate.

        Returns:
            A dictionary with validation results.
        """
        # Classify the text
        result = self.classifier.classify(text)

        # Check if the text is profane
        is_profane = result.label == "profane" and result.confidence >= self.threshold

        if is_profane:
            # Extract profane words from metadata
            profane_words = result.metadata.get("profane_words", []) if result.metadata else []
            profane_word_count = (
                result.metadata.get("profane_word_count", 0) if result.metadata else 0
            )

            # Create issues and suggestions with clearer instructions
            issues = [f"Text contains profanity: {', '.join(profane_words)}"]
            suggestions = [
                f"Remove or rephrase the following profane words: {', '.join(profane_words)}",
                "Consider using more appropriate language",
            ]

            # Create a more explicit message for the model
            message = (
                f"Text contains profanity. Please completely remove or replace the following "
                f"profane words from your response: {', '.join(profane_words)}. "
                f"Generate a new version without any profanity."
            )

            # Calculate score based on confidence
            score = max(0.0, 1.0 - result.confidence)

            logger.debug(
                f"{self.name}: Validation failed, found {profane_word_count} profane words "
                f"with confidence {result.confidence:.2f}"
            )

            return {
                "passed": False,
                "message": message,  # Use the more explicit message
                "details": {
                    "profane_words": profane_words,
                    "profane_word_count": profane_word_count,
                    "confidence": result.confidence,
                    "threshold": self.threshold,
                    "classifier_metadata": result.metadata,
                },
                "score": score,
                "issues": issues,
                "suggestions": suggestions,
            }

        # No profanity found or below threshold
        logger.debug(f"{self.name}: Validation passed with confidence {result.confidence:.2f}")

        return {
            "passed": True,
            "message": "Text contains no profanity",
            "details": {
                "confidence": result.confidence,
                "threshold": self.threshold,
                "classifier_metadata": result.metadata,
            },
            "score": 1.0,
            "issues": [],
            "suggestions": [],
        }


def create_profanity_validator(
    custom_words: Optional[List[str]] = None,
    threshold: float = 0.5,
    name: Optional[str] = None,
    **options: Any,
) -> ProfanityValidator:
    """
    Create a profanity validator.

    This is a convenience function for creating a ProfanityValidator.

    Args:
        custom_words: Optional list of additional profane words to detect.
        threshold: Confidence threshold for considering text profane.
        name: Optional name for the validator.
        **options: Additional options for the validator.

    Returns:
        A ProfanityValidator instance.
    """
    return ProfanityValidator(
        custom_words=custom_words,
        threshold=threshold,
        name=name,
        **options,
    )
