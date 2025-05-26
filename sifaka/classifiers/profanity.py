"""Profanity classifier for detecting inappropriate language in text.

This module provides a classifier for detecting profanity and inappropriate
language using the better_profanity library with fallback to rule-based detection.
"""

import importlib
from typing import Any, List, Optional, Set

from sifaka.classifiers.base import ClassificationResult, ClassifierError, TextClassifier
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Configure logger
logger = get_logger(__name__)

# Basic profanity words for fallback (partial list)
BASIC_PROFANITY_WORDS = {
    "damn",
    "hell",
    "crap",
    "shit",
    "fuck",
    "bitch",
    "ass",
    "bastard",
    "piss",
    "bloody",
    "stupid",
    "idiot",
    "moron",
    "dumb",
    "hate",
}


class ProfanityClassifier(TextClassifier):
    """Classifier for detecting profanity and inappropriate language.

    This classifier uses the better_profanity library when available,
    with fallback to simple rule-based detection. It can detect profanity,
    censor inappropriate words, and provide detailed statistics.

    Attributes:
        custom_words: Additional words to consider profane
        censor_char: Character to use for censoring
        profanity_filter: The profanity detection library instance
    """

    def __init__(
        self,
        custom_words: Optional[List[str]] = None,
        censor_char: str = "*",
        name: str = "ProfanityClassifier",
        description: str = "Detects profanity and inappropriate language",
    ):
        """Initialize the profanity classifier.

        Args:
            custom_words: Additional words to consider profane
            censor_char: Character to use for censoring
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.custom_words = custom_words or []
        self.censor_char = censor_char
        self.profanity_filter: Optional[Any] = None
        self._initialize_filter()

    def _initialize_filter(self) -> None:
        """Initialize the profanity filter."""
        try:
            # Try to use better_profanity
            profanity_module = importlib.import_module("better_profanity")

            # Configure profanity filter
            profanity_module.profanity.load_censor_words()

            # Add custom words if provided
            if self.custom_words:
                profanity_module.profanity.add_censor_words(self.custom_words)

            # Set censor character if the method exists
            if hasattr(profanity_module.profanity, "set_censor_char"):
                profanity_module.profanity.set_censor_char(self.censor_char)

            self.profanity_filter = profanity_module.profanity
            logger.debug("Initialized profanity classifier with better_profanity")

        except ImportError:
            logger.warning(
                "better_profanity not available. Using rule-based detection. "
                "Install better_profanity for better accuracy: pip install better_profanity"
            )
            self.profanity_filter = None

    def classify(self, text: str) -> ClassificationResult:
        """Classify text for profanity.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with profanity prediction

        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label="clean", confidence=1.0, metadata={"reason": "empty_text", "input_length": 0}
            )

        try:
            if self.profanity_filter is not None:
                return self._classify_with_library(text)
            else:
                return self._classify_with_rules(text)

        except Exception as e:
            logger.error(f"Profanity classification failed: {e}")
            raise ClassifierError(
                message=f"Failed to classify text for profanity: {str(e)}",
                component="ProfanityClassifier",
                operation="classification",
            )

    def _classify_with_library(self, text: str) -> ClassificationResult:
        """Classify using better_profanity library."""
        if self.profanity_filter is None:
            raise ClassifierError(
                message="Profanity filter is not available",
                component="ProfanityClassifier",
                operation="library_classification",
            )

        # Check if text contains profanity
        contains_profanity = self.profanity_filter.contains_profanity(text)

        # Censor the text
        censored_text = self.profanity_filter.censor(text)

        # Get profane words by comparing original and censored
        profane_words = self._extract_profane_words(text, censored_text)
        profane_word_count = len(profane_words)

        # Calculate profanity score
        total_words = len(text.split())
        profanity_score = profane_word_count / max(1, total_words)

        # Determine label and confidence
        if contains_profanity:
            label = "profane"
            confidence = 0.5 + (profanity_score * 0.5)  # Scale confidence
        else:
            label = "clean"
            confidence = 1.0 - profanity_score

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "better_profanity",
                "censored_text": censored_text,
                "is_censored": censored_text != text,
                "profane_word_count": profane_word_count,
                "profane_words": list(profane_words),
                "profanity_score": profanity_score,
                "input_length": len(text),
            },
        )

    def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using simple rule-based approach."""
        text_lower = text.lower()
        words = text_lower.split()

        # Combine basic profanity words with custom words
        all_profanity = BASIC_PROFANITY_WORDS.union(set(word.lower() for word in self.custom_words))

        # Find profane words
        profane_words = [word for word in words if word in all_profanity]
        profane_word_count = len(profane_words)

        # Calculate profanity score
        total_words = len(words)
        profanity_score = profane_word_count / max(1, total_words)

        # Create censored text
        censored_words = []
        for word in words:
            if word in all_profanity:
                censored_words.append(self.censor_char * len(word))
            else:
                censored_words.append(word)
        censored_text = " ".join(censored_words)

        # Determine label and confidence
        if profane_word_count > 0:
            label = "profane"
            confidence = 0.6 + (profanity_score * 0.3)  # Conservative confidence
        else:
            label = "clean"
            confidence = 0.8  # Conservative confidence for rule-based

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "censored_text": censored_text,
                "is_censored": censored_text != text,
                "profane_word_count": profane_word_count,
                "profane_words": profane_words,
                "profanity_score": profanity_score,
                "input_length": len(text),
            },
        )

    def _extract_profane_words(self, original: str, censored: str) -> Set[str]:
        """Extract profane words by comparing original and censored text."""
        original_words = original.split()
        censored_words = censored.split()

        profane_words = set()
        for orig, cens in zip(original_words, censored_words, strict=False):
            if orig != cens and self.censor_char in cens:
                profane_words.add(orig.lower())

        return profane_words

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["clean", "profane"]


def create_profanity_validator(
    custom_words: Optional[List[str]] = None, name: str = "ProfanityValidator"
) -> ClassifierValidator:
    """Create a validator that detects profanity in text.

    Args:
        custom_words: Additional words to consider profane
        name: Name of the validator

    Returns:
        A ClassifierValidator configured for profanity detection
    """
    classifier = ProfanityClassifier(custom_words=custom_words)

    return ClassifierValidator(
        classifier=classifier,
        threshold=0.5,
        invalid_labels=["profane"],  # Profane text is invalid
        name=name,
    )
