"""
Profanity classifier for Sifaka.

This module provides a classifier that detects profanity and inappropriate language
in text using the better_profanity library, which is a fast, flexible profanity
filter with good accuracy.
"""

import importlib
import logging
from typing import Any, List, Optional, Set

from sifaka.classifiers import BaseClassifier, ClassificationResult

# Configure logger
logger = logging.getLogger(__name__)


class ProfanityClassifier(BaseClassifier):
    """
    A profanity classifier that detects inappropriate language in text.

    This classifier uses the better_profanity library to detect profanity and
    inappropriate language in text. It supports custom word lists and censoring,
    and provides detailed metadata about detected profanity.
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
        super().__init__(name=name, description=description)
        self._custom_words = custom_words or []
        self._censor_char = censor_char
        self._profanity = None
        self._initialized = False

        # Log initialization
        logger.debug(
            f"Initialized {self.name} with {len(self._custom_words)} custom words, "
            f"censor_char='{self._censor_char}'"
        )

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

            # Set censor character if the method exists
            # Some versions of better_profanity might not have this method
            if hasattr(profanity.profanity, "set_censor_char"):
                profanity.profanity.set_censor_char(self._censor_char)

            return profanity.profanity
        except ImportError:
            logger.error("better_profanity package is required for ProfanityClassifier")
            raise ImportError(
                "better_profanity package is required for ProfanityClassifier. "
                "Install it with: pip install better_profanity"
            )
        except Exception as e:
            logger.error(f"Failed to initialize better_profanity: {str(e)}")
            raise RuntimeError(f"Failed to initialize better_profanity: {e}")

    def _initialize(self) -> None:
        """Initialize the profanity filter if needed."""
        if not self._initialized:
            self._profanity = self._load_profanity()
            self._initialized = True
            logger.debug(f"{self.name}: Initialized profanity filter")

    def _get_profane_words(self, text: str) -> Set[str]:
        """
        Get the profane words in text.

        Args:
            text: The text to check for profane words.

        Returns:
            A set of profane words found in the text.
        """
        import re

        # Ensure profanity filter is initialized
        if self._profanity is None:
            self._initialize()

        # After initialization, profanity filter should be available
        assert self._profanity is not None, "Profanity filter not initialized"

        # This is a more robust approach to find profane words
        profane_words: Set[str] = set()

        # First, get the censored text
        censored_text = self._profanity.censor(text)

        # If the text wasn't censored, there are no profane words
        if censored_text == text:
            return profane_words

        # Get the custom words list from the profanity filter
        # This includes both the default words and any custom words we added
        custom_words = set(self._custom_words)

        # Check for each custom word in the text (case-insensitive)
        for word in custom_words:
            # Use regex to find the word with word boundaries
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                profane_words.add(word)

        # If we didn't find any custom words but the text was censored,
        # we need to extract the censored words from the text
        if not profane_words and censored_text != text:
            # Find all sequences of censor characters in the censored text
            censor_pattern = r"\*+"
            censor_matches = re.finditer(censor_pattern, censored_text)

            # For each censored sequence, find the corresponding word in the original text
            for match in censor_matches:
                start, end = match.span()

                # Find the word boundaries in the original text
                # This is a simplified approach and might not work for all cases
                word_start = max(0, start - 10)  # Look back up to 10 characters
                word_end = min(len(text), end + 10)  # Look ahead up to 10 characters

                # Extract a chunk of the original text around the censored part
                chunk = text[word_start:word_end]

                # Split the chunk into words and check each one
                for word in chunk.split():
                    # Clean the word from punctuation
                    clean_word = re.sub(r"[^\w]", "", word).lower()

                    # Check if this word is profane
                    if self._profanity.contains_profanity(clean_word):
                        profane_words.add(clean_word)

        return profane_words

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text as profane or clean.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the profanity label and confidence score.
        """
        # Handle empty text
        if not text or not text.strip():
            logger.debug(f"{self.name}: Empty text provided, returning clean result")
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

            # After initialization, profanity filter should be available
            assert self._profanity is not None, "Failed to initialize profanity filter"

            # Censor the text first
            censored_text = self._profanity.censor(text)

            # Check if text was censored (contains profanity)
            contains_profanity = censored_text != text

            # Get profane words using our improved method
            profane_words = self._get_profane_words(text)
            profane_word_count = len(profane_words)

            # If we found profane words but the text wasn't censored, or vice versa,
            # we need to ensure consistency
            if profane_word_count > 0 and not contains_profanity:
                # We found profane words but the text wasn't censored
                # This shouldn't happen with our improved method, but just in case
                contains_profanity = True

                # Re-censor the text manually for each profane word
                import re

                censored_text = text
                for word in profane_words:
                    pattern = r"\b" + re.escape(word) + r"\b"
                    censored_text = re.sub(
                        pattern, "*" * len(word), censored_text, flags=re.IGNORECASE
                    )
            elif profane_word_count == 0 and contains_profanity:
                # The text was censored but we didn't find any profane words
                # This could happen if better_profanity censored something our method missed
                logger.debug(
                    f"{self.name}: Text was censored but no profane words were found, using better_profanity's detection"
                )

            # Calculate profanity score
            # Simple scoring: number of profane words / total words
            total_words = len(text.split())
            profanity_score = profane_word_count / max(1, total_words)

            # Determine label and confidence
            if contains_profanity:
                label = "profane"
                # Scale confidence based on profanity score
                confidence = 0.5 + (profanity_score * 0.5)
                logger.debug(
                    f"{self.name}: Text classified as profane with confidence {confidence:.2f}, "
                    f"found {profane_word_count} profane words"
                )
            else:
                label = "clean"
                confidence = 1.0 - profanity_score
                logger.debug(
                    f"{self.name}: Text classified as clean with confidence {confidence:.2f}"
                )

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
            logger.error(f"Error in {self.name}: {str(e)}")
            return ClassificationResult(
                label="clean",
                confidence=0.5,
                metadata={
                    "error": str(e),
                    "reason": "classification_error",
                    "input_length": len(text),
                },
            )
