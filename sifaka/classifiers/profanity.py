"""
Profanity classifier using better_profanity.

This module provides a profanity classifier that uses the better_profanity package
to detect profane and inappropriate language in text. It categorizes text as either
'clean', 'profane', or 'unknown' based on the presence of profane words.

## Architecture

ProfanityClassifier follows the composition over inheritance pattern:
1. **Public API**: Classifier class with classify() and batch_classify() methods
2. **Core Logic**: ProfanityClassifierImplementation implements classification logic
3. **State Management**: Uses ClassifierState for internal state

## Lifecycle

1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract custom words and censor character from config.params
   - Set up default values

2. **Warm-up**: Load profanity checking resources
   - Load better_profanity package when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply profanity detection
   - Calculate confidence based on profanity ratio
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map profanity detection to labels
   - Include censored text in metadata
   - Include profanity statistics in metadata

## Usage Examples

```python
from sifaka.classifiers.profanity import create_profanity_classifier

# Create a profanity classifier with default settings
classifier = create_profanity_classifier()

# Classify text
result = classifier.classify("This is a clean message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

# Create a classifier with custom profanity words
custom_classifier = create_profanity_classifier(
    custom_words=["inappropriate", "offensive", "controversial"],
    censor_char="#",
    cache_size=100
)

# Classify text and view censored version
result = custom_classifier.classify("This is an inappropriate message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
print(f"Censored text: {result.metadata['censored_text']}")
```
"""

import importlib
from abc import abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import TypeGuard
from pydantic import ConfigDict, PrivateAttr

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
    Classifier,
    ClassifierImplementation,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState

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


class CensorResult:
    """Result of text censoring operation."""

    def __init__(
        self,
        original_text: str,
        censored_text: str,
        censored_word_count: int,
        total_word_count: int,
    ):
        self.original_text = original_text
        self.censored_text = censored_text
        self.censored_word_count = censored_word_count
        self.total_word_count = total_word_count

    @property
    def profanity_ratio(self) -> float:
        """Calculate ratio of profane words to total words."""
        return self.censored_word_count / max(self.total_word_count, 1)


class ProfanityClassifierImplementation:
    """
    Implementation of profanity classification logic using better_profanity.

    This implementation uses the better_profanity package to detect profane and
    inappropriate language in text. It provides a lightweight, dictionary-based
    approach to profanity detection with support for custom word lists and
    censoring options.

    ## Architecture

    ProfanityClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements profanity detection
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages better_profanity resources

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with config containing parameters
       - Extract custom words and censor character from config.params
       - Set up default values

    2. **Warm-up**: Load profanity checking resources
       - Load better_profanity package when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Validate input text
       - Apply profanity detection
       - Calculate confidence based on profanity ratio
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map profanity detection to labels
       - Include censored text in metadata
       - Include profanity statistics in metadata

    ## Examples

    ```python
    from sifaka.classifiers.profanity import create_profanity_classifier

    # Create a profanity classifier with default settings
    classifier = create_profanity_classifier()

    # Classify text
    result = classifier.classify("This is a clean message.")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")

    # Access censored text and statistics
    if result.label == "profane":
        print(f"Censored text: {result.metadata['censored_text']}")
        print(f"Profanity ratio: {result.metadata['profanity_ratio']:.2f}")
    ```
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["clean", "profane", "unknown"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for dictionary-based check

    def __init__(
        self,
        config: ClassifierConfig,
        checker: Optional[ProfanityChecker] = None,
    ) -> None:
        """
        Initialize the profanity classifier implementation.

        Args:
            config: Configuration for the classifier
            checker: Optional custom profanity checker implementation
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

        # Store checker in state if provided
        if checker is not None:
            if self._validate_checker(checker):
                self._state.cache["checker"] = checker

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
            # Check if checker is already in state
            if "checker" in self._state.cache:
                return self._state.cache["checker"]

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

            # Validate and store in state
            if self._validate_checker(checker):
                self._state.cache["checker"] = checker
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
        self.warm_up_impl()

        if "checker" in self._state.cache:
            checker = self._state.cache["checker"]
            checker.add_censor_words(words)

    def warm_up_impl(self) -> None:
        """Initialize the profanity checker if needed."""
        if not self._state.initialized:
            # Load profanity checker
            checker = self._load_profanity()
            self._state.cache["checker"] = checker
            self._state.initialized = True

    def _censor_text(self, text: str) -> CensorResult:
        """
        Censor profane words in text.

        Args:
            text: Text to censor

        Returns:
            CensorResult with censoring details
        """
        if not self._state.initialized or "checker" not in self._state.cache:
            raise RuntimeError("Profanity checker not initialized. Call warm_up_impl() first.")

        checker = self._state.cache["checker"]

        # Count total words
        total_words = len(text.split())

        # Censor the text
        censored_text = checker.censor(text)

        # Count censored words by comparing original and censored text
        censored_words = sum(
            1 for orig, censored in zip(text.split(), censored_text.split()) if orig != censored
        )

        return CensorResult(
            original_text=text,
            censored_text=censored_text,
            censored_word_count=censored_words,
            total_word_count=total_words,
        )

    def classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement classification logic for profanity detection.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence
        """
        if not self._state.initialized:
            self.warm_up_impl()

        try:
            # Handle empty text
            if not text:
                return ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={"reason": "empty_text"},
                )

            # Get checker from state
            checker = self._state.cache["checker"]

            # Check for profanity and censor text
            contains_profanity = checker.contains_profanity(text)
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


def create_profanity_classifier(
    name: str = "profanity_classifier",
    description: str = "Detects profanity and inappropriate language",
    custom_words: Optional[List[str]] = None,
    censor_char: str = "*",
    min_confidence: float = 0.5,
    cache_size: int = 0,
    cost: int = 1,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Factory function to create a profanity classifier.

    This function provides a simpler interface for creating a profanity classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        custom_words: Optional list of custom profanity words to check for
        censor_char: Character to use for censoring profane words
        min_confidence: Minimum confidence for profanity classification
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured Classifier instance with ProfanityClassifierImplementation

    Examples:
        ```python
        from sifaka.classifiers.profanity import create_profanity_classifier

        # Create a profanity classifier with default settings
        classifier = create_profanity_classifier()

        # Create a classifier with custom profanity words
        custom_classifier = create_profanity_classifier(
            custom_words=["inappropriate", "offensive", "controversial"],
            censor_char="#",
            cache_size=100
        )
        ```
    """
    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})
    params.update(
        {
            "custom_words": custom_words or [],
            "censor_char": censor_char,
            "min_confidence": min_confidence,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=ProfanityClassifierImplementation.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = ProfanityClassifierImplementation(config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_profanity_classifier_with_custom_checker(
    checker: ProfanityChecker,
    name: str = "custom_profanity_classifier",
    description: str = "Custom profanity checker",
    cache_size: int = 0,
    cost: int = 1,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Factory function to create a profanity classifier with a custom checker.

    This function provides a way to create a profanity classifier with a custom
    profanity checker implementation, which can be useful for testing or
    specialized profanity detection needs.

    Args:
        checker: Custom profanity checker implementation
        name: Name of the classifier
        description: Description of the classifier
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured Classifier instance with ProfanityClassifierImplementation

    Raises:
        ValueError: If the checker doesn't implement the ProfanityChecker protocol
    """
    # Validate checker first
    if not isinstance(checker, ProfanityChecker):
        raise ValueError(f"Checker must implement ProfanityChecker protocol, got {type(checker)}")

    # Prepare params
    params: Dict[str, Any] = kwargs.pop("params", {})

    # Create config
    config = ClassifierConfig(
        labels=ProfanityClassifierImplementation.DEFAULT_LABELS,
        cache_size=cache_size,
        cost=cost,
        params=params,
    )

    # Create implementation with custom checker
    implementation = ProfanityClassifierImplementation(config, checker=checker)

    # Initialize the implementation
    implementation.warm_up_impl()

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
