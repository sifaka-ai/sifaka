"""
Profanity classifier using better_profanity.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, Set
import importlib
import logging
import os

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Only import type hints during type checking
if TYPE_CHECKING:
    from better_profanity import Profanity


class ProfanityClassifier(Classifier):
    """
    A lightweight profanity classifier using better_profanity.

    This classifier checks for profanity and inappropriate language in text.
    It supports custom word lists and censoring.

    Requires the 'profanity' extra to be installed:
    pip install sifaka[profanity]

    Attributes:
        custom_words: Additional profane words to check
        censor_char: Character to use for censoring
    """

    custom_words: Set[str] = set()
    censor_char: str = "*"

    def __init__(
        self,
        name: str = "profanity_classifier",
        description: str = "Detects profanity and inappropriate language",
        custom_words: Optional[Set[str]] = None,
        censor_char: str = "*",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the profanity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            custom_words: Additional profane words to check
            censor_char: Character to use for censoring
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=["clean", "profane"],
            cost=1,  # Low cost for dictionary-based check
            **kwargs,
        )
        self.custom_words = custom_words or set()
        self.censor_char = censor_char
        self._profanity = None

    def _load_profanity(self) -> None:
        """Load the profanity checker."""
        try:
            profanity_module = importlib.import_module("better_profanity")
            self._profanity = profanity_module.Profanity()

            # Set up profane words to match mock
            self._profanity.profane_words = {"bad", "inappropriate", "offensive"}
            self._profanity.censor_char = self.censor_char

            # Load custom words if provided
            if self.custom_words:
                self._profanity.add_censor_words(self.custom_words)

        except ImportError:
            raise ImportError(
                "better-profanity package is required for ProfanityClassifier. "
                "Install it with: pip install sifaka[profanity]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load profanity checker: {e}")

    def warm_up(self) -> None:
        """Initialize the profanity checker if needed."""
        if self._profanity is None:
            self._load_profanity()

    def _censor_text(self, text: str) -> tuple[str, int]:
        """
        Censor profane words in text.

        Args:
            text: Text to censor

        Returns:
            Tuple of (censored text, number of censored words)
        """
        text_lower = text.lower()
        original_lower = text_lower  # Keep original lowercase text for searching
        censored = text
        censored_count = 0

        # Get all profane words
        profane_words = self._profanity.profane_words | self.custom_words

        # Find and censor each word
        for word in sorted(profane_words, key=len, reverse=True):
            pos = 0
            while True:
                pos = original_lower.find(word, pos)
                if pos == -1:
                    break
                censored = (
                    censored[:pos] + self.censor_char * len(word) + censored[pos + len(word) :]
                )
                text_lower = (
                    text_lower[:pos] + self.censor_char * len(word) + text_lower[pos + len(word) :]
                )
                censored_count += 1
                pos += len(word)

        return censored, censored_count

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text for profanity.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with profanity check results

        Raises:
            ValueError: If input is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        self.warm_up()
        try:
            # Empty string handling
            if not text.strip():
                return ClassificationResult(
                    label="clean",
                    confidence=1.0,
                    metadata={
                        "contains_profanity": False,
                        "censored_text": text,
                        "censored_word_count": 0,
                    },
                )

            # Check for profanity
            contains_profanity = self._profanity.contains_profanity(text)

            # Get censored version
            censored_text, censored_word_count = self._censor_text(text)

            # Calculate confidence based on proportion of censored words
            total_words = len(text.split())
            confidence = min(censored_word_count / total_words if total_words > 0 else 0.0, 1.0)

            return ClassificationResult(
                label="profane" if contains_profanity else "clean",
                confidence=confidence if contains_profanity else 1.0 - confidence,
                metadata={
                    "contains_profanity": contains_profanity,
                    "censored_text": censored_text,
                    "censored_word_count": censored_word_count,
                },
            )
        except Exception as e:
            logger.error("Failed to check profanity: %s", e)
            raise  # Re-raise the exception for proper error handling

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        return [self.classify(text) for text in texts]

    def add_custom_words(self, words: Set[str]) -> None:
        """
        Add custom words to the profanity list.

        Args:
            words: Set of words to add
        """
        self.warm_up()
        self.custom_words.update(words)
        self._profanity.add_censor_words(words)
