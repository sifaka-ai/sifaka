"""
Profanity classifier using better_profanity.
"""

import importlib
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Final,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ProfanityChecker(Protocol):
    """Protocol for profanity checking engines."""

    @abstractmethod
    def contains_profanity(self, text: str) -> bool: ...
    @abstractmethod
    def censor(self, text: str) -> str: ...
    @property
    @abstractmethod
    def profane_words(self) -> Set[str]: ...
    @profane_words.setter
    @abstractmethod
    def profane_words(self, words: Set[str]) -> None: ...
    @property
    @abstractmethod
    def censor_char(self) -> str: ...
    @censor_char.setter
    @abstractmethod
    def censor_char(self, char: str) -> None: ...


@dataclass(frozen=True)
class ProfanityConfig:
    """
    Configuration for profanity detection.

    Note: This class is provided for backward compatibility.
    The preferred way to configure profanity detection is to use
    ClassifierConfig with params:

    ```python
    config = ClassifierConfig(
        labels=["clean", "profane", "unknown"],
        params={
            "custom_words": ["word1", "word2"],
            "censor_char": "*",
            "min_confidence": 0.7
        }
    )
    ```
    """

    custom_words: Set[str] = field(default_factory=frozenset)
    censor_char: str = "*"
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if len(self.censor_char) != 1:
            raise ValueError("censor_char must be a single character")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class CensorResult:
    """Result of text censoring operation."""

    original_text: str
    censored_text: str
    censored_word_count: int
    total_word_count: int

    @property
    def profanity_ratio(self) -> float:
        """Calculate ratio of profane words to total words."""
        return self.censored_word_count / max(self.total_word_count, 1)


class ProfanityClassifier(BaseClassifier):
    """
    A lightweight profanity classifier using better_profanity.

    This classifier checks for profanity and inappropriate language in text.
    It supports custom word lists and censoring.

    Requires the 'profanity' extra to be installed:
    pip install sifaka[profanity]
    """

    # Class-level constants
    DEFAULT_LABELS: Final[List[str]] = ["clean", "profane", "unknown"]
    DEFAULT_COST: Final[int] = 1  # Low cost for dictionary-based check

    def __init__(
        self,
        name: str = "profanity_classifier",
        description: str = "Detects profanity and inappropriate language",
        profanity_config: Optional[ProfanityConfig] = None,
        checker: Optional[ProfanityChecker] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the profanity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            profanity_config: Configuration for profanity detection
            checker: Custom profanity checker implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Store checker for later use
        self._checker = checker
        self._initialized = False

        # We'll use config.params for all configuration, so we don't need to store profanity_config separately

        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Add profanity config to params if provided
            if profanity_config is not None:
                # Store all profanity config values in params for consistency
                params.update(
                    {
                        "custom_words": list(profanity_config.custom_words),
                        "censor_char": profanity_config.censor_char,
                        "min_confidence": profanity_config.min_confidence,
                    }
                )

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, params=params, **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

    def _validate_checker(self, checker: Any) -> TypeGuard[ProfanityChecker]:
        """Validate that a checker implements the required protocol."""
        if not isinstance(checker, ProfanityChecker):
            raise ValueError(
                f"Checker must implement ProfanityChecker protocol, got {type(checker)}"
            )
        return True

    def _load_profanity(self) -> ProfanityChecker:
        """Load the profanity checker."""
        try:
            profanity_module = importlib.import_module("better_profanity")
            checker = profanity_module.Profanity()

            # Configure the checker
            checker.profane_words = {"bad", "inappropriate", "offensive"}

            # Get configuration from params for consistency
            censor_char = self.config.params.get("censor_char", "*")
            checker.censor_char = censor_char

            # Add custom words if provided
            custom_words = self.config.params.get("custom_words", [])
            if custom_words:
                # Convert to set if it's a list
                if isinstance(custom_words, list):
                    custom_words = set(custom_words)
                checker.profane_words.update(custom_words)

            self._validate_checker(checker)
            return checker

        except ImportError:
            raise ImportError(
                "better-profanity package is required for ProfanityClassifier. "
                "Install it with: pip install sifaka[profanity]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load profanity checker: {e}")

    @property
    def custom_words(self) -> Set[str]:
        """Get the custom profanity words."""
        # Always use config.params for consistency
        custom_words = self.config.params.get("custom_words", [])
        return set(custom_words) if isinstance(custom_words, list) else set()

    @property
    def censor_char(self) -> str:
        """Get the censoring character."""
        # Always use config.params for consistency
        return self.config.params.get("censor_char", "*")

    def add_custom_words(self, words: Set[str]) -> None:
        """Add custom words to the profanity list."""
        self.warm_up()
        if self._checker:
            self._checker.add_censor_words(words)

    def warm_up(self) -> None:
        """Initialize the profanity checker if needed."""
        if not self._initialized:
            self._checker = self._checker or self._load_profanity()
            self._initialized = True

    def _censor_text(self, text: str) -> CensorResult:
        """
        Censor profane words in text.

        Args:
            text: Text to censor

        Returns:
            CensorResult with censoring details
        """
        censored = self._checker.censor(text)
        total_words = len(text.split())
        censored_count = sum(
            1 for _, censored_char in zip(text, censored) if censored_char == self.censor_char
        ) // max(len(self.censor_char), 1)

        return CensorResult(
            original_text=text,
            censored_text=censored,
            censored_word_count=censored_count,
            total_word_count=total_words,
        )

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement profanity classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with profanity check results
        """
        self.warm_up()

        try:
            # Note: Empty text is handled by BaseClassifier.classify
            # so we don't need to handle it here

            # Check for profanity and censor text
            contains_profanity = self._checker.contains_profanity(text)
            censor_result = self._censor_text(text)

            # Calculate confidence based on proportion of censored words
            # Use min_confidence from config.params for consistency
            min_confidence = self.config.params.get("min_confidence", 0.5)
            confidence = max(
                censor_result.profanity_ratio,
                min_confidence if contains_profanity else 0.0,
            )

            return ClassificationResult(
                label="profane" if contains_profanity else "clean",
                confidence=confidence if contains_profanity else 1.0 - confidence,
                metadata={
                    "contains_profanity": contains_profanity,
                    "censored_text": censor_result.censored_text,
                    "censored_word_count": censor_result.censored_word_count,
                    "total_word_count": censor_result.total_word_count,
                    "profanity_ratio": censor_result.profanity_ratio,
                },
            )

        except Exception as e:
            logger.error("Failed to check profanity: %s", e)
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "reason": "profanity_check_error",
                },
            )

    @classmethod
    def create_with_custom_checker(
        cls,
        checker: ProfanityChecker,
        name: str = "custom_profanity_classifier",
        description: str = "Custom profanity checker",
        profanity_config: Optional[ProfanityConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "ProfanityClassifier":
        """
        Factory method to create a classifier with a custom checker.

        Args:
            checker: Custom profanity checker implementation
            name: Name of the classifier
            description: Description of the classifier
            profanity_config: Custom profanity configuration
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured ProfanityClassifier instance
        """
        # Validate checker first
        if not isinstance(checker, ProfanityChecker):
            raise ValueError(
                f"Checker must implement ProfanityChecker protocol, got {type(checker)}"
            )

        # Create instance with validated checker
        instance = cls(
            name=name,
            description=description,
            profanity_config=profanity_config,
            checker=checker,
            config=config,
            **kwargs,
        )

        return instance
