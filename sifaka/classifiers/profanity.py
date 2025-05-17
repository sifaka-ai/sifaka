"""
Profanity classifier for Sifaka.

This module provides a classifier that detects profanity and inappropriate language
in text using the better_profanity library, which is a fast, flexible profanity
filter with good accuracy.
"""

import importlib
from typing import List, Optional, Set, Any

from sifaka.classifiers import ClassificationResult


class ProfanityClassifier:
    """
    A profanity classifier that detects inappropriate language in text.

    This classifier uses the better_profanity library to detect profanity and
    inappropriate language in text. It supports custom word lists and censoring,
    and provides detailed metadata about detected profanity.

    Attributes:
        custom_words: Optional list of additional profane words to detect.
        censor_char: Character to use for censoring profane words.
        name: The name of the classifier.
        description: The description of the classifier.
    """

    def __init__(
        self,
        custom_words: Optional[List[str]] = None,
        censor_char: str = "*",
        name: str = "profanity_classifier",
        description: str = "Detects profanity and inappropriate language in text",
    ):
        """
        Initialize the profanity classifier.

        Args:
            custom_words: Optional list of additional profane words to detect.
            censor_char: Character to use for censoring profane words.
            name: The name of the classifier.
            description: The description of the classifier.
        """
        self._name = name
        self._description = description
        self._custom_words = custom_words or []
        self._censor_char = censor_char
        self._profanity = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def _load_profanity(self) -> Any:
        """
        Load the better_profanity library and configure it.

        Returns:
            The configured better_profanity module.

        Raises:
            ImportError: If better_profanity is not installed.
            RuntimeError: If initialization fails.
        """
        try:
            profanity = importlib.import_module("better_profanity")

            # Configure profanity filter
            profanity.profanity.load_censor_words()

            # Add custom words if provided
            if self._custom_words:
                profanity.profanity.add_censor_words(self._custom_words)

            # Set censor character
            profanity.profanity.set_censor_char(self._censor_char)

            return profanity.profanity
        except ImportError:
            raise ImportError(
                "better_profanity package is required for ProfanityClassifier. "
                "Install it with: pip install better_profanity"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize better_profanity: {e}")

    def _initialize(self) -> None:
        """Initialize the profanity filter if needed."""
        if not self._initialized:
            self._profanity = self._load_profanity()
            self._initialized = True

    def _get_profane_words(self, text: str) -> Set[str]:
        """
        Get the profane words in text.

        Args:
            text: The text to check for profane words.

        Returns:
            A set of profane words found in the text.
        """
        # This is a workaround since better_profanity doesn't expose the matched words directly
        # We'll check each word in the text against the profanity filter
        words = text.split()
        profane_words: Set[str] = set()

        if self._profanity is None:
            return profane_words

        for word in words:
            if self._profanity.contains_profanity(word):
                profane_words.add(word)

        return profane_words

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as profane or clean.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the profanity label and confidence score.
        """
        # Initialize profanity filter if needed
        self._initialize()

        # Handle empty text
        if not text or not text.strip():
            return ClassificationResult(
                label="clean",
                confidence=1.0,
                metadata={
                    "input_length": 0,
                    "reason": "empty_text",
                    "censored_text": "",
                    "is_censored": False,
                    "profane_word_count": 0,
                },
            )

        try:
            # Ensure profanity filter is initialized
            if self._profanity is None:
                self._initialize()

            # Check again after initialization
            if self._profanity is None:
                raise RuntimeError("Failed to initialize profanity filter")

            # Check if text contains profanity
            contains_profanity = self._profanity.contains_profanity(text)

            # Censor the text
            censored_text = self._profanity.censor(text)

            # Get profane words
            profane_words = self._get_profane_words(text)
            profane_word_count = len(profane_words)

            # Calculate profanity score
            # Simple scoring: number of profane words / total words
            total_words = len(text.split())
            profanity_score = profane_word_count / max(1, total_words)

            # Determine label and confidence
            if contains_profanity:
                label = "profane"
                # Scale confidence based on profanity score
                confidence = 0.5 + (profanity_score * 0.5)
            else:
                label = "clean"
                confidence = 1.0 - profanity_score

            return ClassificationResult(
                label=label,
                confidence=confidence,
                metadata={
                    "input_length": len(text),
                    "censored_text": censored_text,
                    "is_censored": censored_text != text,
                    "profane_word_count": profane_word_count,
                    "profane_words": list(profane_words),
                    "profanity_score": profanity_score,
                },
            )

        except Exception as e:
            # Handle errors
            return ClassificationResult(
                label="clean",
                confidence=0.5,
                metadata={
                    "error": str(e),
                    "reason": "classification_error",
                    "input_length": len(text),
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: The list of texts to classify.

        Returns:
            A list of ClassificationResults.
        """
        return [self.classify(text) for text in texts]
