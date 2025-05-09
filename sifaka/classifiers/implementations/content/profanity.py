"""
Profanity classifier using better_profanity.

This module provides a profanity classifier that uses the better_profanity package
to detect profane and inappropriate language in text. It categorizes text as either
'clean', 'profane', or 'unknown' based on the presence of profane words.

## Architecture

ProfanityClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements profanity detection
4. **State Management**: Uses StateManager for internal state

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
from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

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

from sifaka.classifiers.base import BaseClassifier
from sifaka.classifiers.models import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)

# Type variable for the classifier
P = TypeVar("P", bound="ProfanityClassifier")


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


class ProfanityClassifier(BaseClassifier):
    """
    A lightweight profanity classifier using better_profanity.

    This classifier checks for profanity and inappropriate language in text.
    It supports custom word lists and censoring, and provides detailed metadata
    about detected profanity including censored text and profanity statistics.

    ## Architecture

    ProfanityClassifier follows the standard Sifaka classifier architecture:
    1. **Public API**: classify() and batch_classify() methods (inherited)
    2. **Caching Layer**: _classify_impl() handles caching (inherited)
    3. **Core Logic**: _classify_impl_uncached() implements profanity detection
    4. **State Management**: Uses StateManager for internal state

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

    ## Examples

    ```python
    from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

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

    Requires the 'profanity' extra to be installed:
    pip install sifaka[profanity]
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["clean", "profane", "unknown"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for dictionary-based check

    def __init__(
        self,
        name: str = "profanity_classifier",
        description: str = "Detects profanity and inappropriate language",
        checker: Optional[ProfanityChecker] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the profanity classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            checker: Custom profanity checker implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, params=params, **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize state
        # This is now handled by BaseClassifier in model_post_init

        # Store checker in state if provided
        if checker is not None:
            if self._validate_checker(checker):
                self._state.update("cache", {"checker": checker})

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
            if self._state.get("cache", {}).get("checker"):
                return self._state.get("cache")["checker"]

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
                cache = self._state.get("cache", {})
                cache["checker"] = checker
                self._state.update("cache", cache)
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

        checker = self._state.get("cache", {}).get("checker")
        if checker:
            checker.add_censor_words(words)

    def warm_up(self) -> None:
        """Initialize the profanity checker if needed."""
        if not self._state.get("initialized", False):
            # Load profanity checker
            checker = self._load_profanity()
            cache = self._state.get("cache", {})
            cache["checker"] = checker
            self._state.update("cache", cache)
            self._state.update("initialized", True)

    def _censor_text(self, text: str) -> CensorResult:
        """
        Censor profane words in text.

        Args:
            text: Text to censor

        Returns:
            CensorResult with censoring details
        """
        if not self._state.get("initialized", False):
            raise RuntimeError("Profanity checker not initialized. Call warm_up() first.")

        checker = self._state.get("cache", {}).get("checker")
        if not checker:
            raise RuntimeError("Profanity checker not found in state.")

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

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """
        Implement classification logic for profanity detection.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence
        """
        if not self._state.get("initialized", False):
            self.warm_up()

        try:
            # Handle empty text
            from sifaka.utils.text import handle_empty_text_for_classifier

            empty_result = handle_empty_text_for_classifier(text)
            if empty_result:
                return empty_result

            # Get checker from state
            checker = self._state.get("cache", {}).get("checker")

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

            result = ClassificationResult(
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

            # Track statistics
            stats = self._state.get("statistics", {})
            stats[result.label] = stats.get(result.label, 0) + 1
            self._state.update("statistics", stats)

            return result

        except Exception as e:
            logger.error("Failed to check profanity: %s", e)
            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__}
            errors = self._state.get("errors", [])
            errors.append(error_info)
            self._state.update("errors", errors)

            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "reason": "profanity_check_error",
                },
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, and cache information.

        Returns:
            Dictionary containing statistics
        """
        stats = {
            # Classification counts by label
            "classifications": self._state.get("statistics", {}),
            # Number of errors encountered
            "error_count": len(self._state.get("errors", [])),
            # Cache information
            "cache_enabled": self.config.cache_size > 0,
            "cache_size": self.config.cache_size,
            # State initialization status
            "initialized": self._state.get("initialized", False),
        }

        # Add cache hit ratio if caching is enabled
        if hasattr(self, "_result_cache"):
            stats["cache_entries"] = len(self._result_cache)

        # Add checker information if available
        if self._state.get("cache", {}).get("checker"):
            stats["has_checker"] = True

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and any cached data in the state.
        """
        # Clear classification result cache
        if hasattr(self, "_result_cache"):
            self._result_cache.clear()

        # Reset state cache but keep the checker
        checker = self._state.get("cache", {}).get("checker")
        self._state.update("cache", {"checker": checker} if checker else {})

        # Reset statistics
        self._state.update("statistics", {})

        # Reset errors list but keep initialized status
        self._state.update("errors", [])

    @classmethod
    def create(
        cls: Type[P],
        name: str = "profanity_classifier",
        description: str = "Detects profanity and inappropriate language",
        labels: Optional[List[str]] = None,
        custom_words: Optional[List[str]] = None,
        censor_char: str = "*",
        min_confidence: float = 0.5,
        cache_size: int = 0,
        cost: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> P:
        """
        Create a new instance with the given parameters.

        This factory method creates a new instance of the classifier with the
        specified parameters, handling the creation of the ClassifierConfig
        object and setting up the classifier with the appropriate parameters.

        Args:
            name: Name of the classifier
            description: Description of the classifier
            labels: List of valid labels
            custom_words: Optional list of custom profanity words to check for
            censor_char: Character to use for censoring profane words
            min_confidence: Minimum confidence for profanity classification
            cache_size: Size of the classification cache (0 to disable)
            cost: Computational cost of this classifier
            params: Additional configuration parameters
            **kwargs: Additional keyword arguments

        Returns:
            A new instance of the classifier
        """
        # Create params dictionary if not provided
        if params is None:
            params = {}

        # Add specific parameters
        params.update(
            {
                "custom_words": custom_words or [],
                "censor_char": censor_char,
                "min_confidence": min_confidence,
            }
        )

        # Add kwargs to params
        params.update(kwargs.pop("params", {}))

        # Create config
        config = ClassifierConfig(
            labels=labels or cls.DEFAULT_LABELS,
            cache_size=cache_size,
            cost=cost or cls.DEFAULT_COST,
            params=params,
        )

        # Create and return instance
        return cls(name=name, description=description, config=config, **kwargs)

    @classmethod
    def create_with_custom_checker(
        cls: Type[P],
        checker: ProfanityChecker,
        name: str = "custom_profanity_classifier",
        description: str = "Custom profanity checker",
        config: Optional[ClassifierConfig] = None,
        **kwargs: Any,
    ) -> P:
        """
        Factory method to create a classifier with a custom checker.

        This method creates a new instance of the classifier with a custom
        profanity checker implementation, which can be useful for testing
        or specialized profanity detection needs.

        Args:
            checker: Custom profanity checker implementation
            name: Name of the classifier
            description: Description of the classifier
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured ProfanityClassifier instance

        Raises:
            ValueError: If the checker doesn't implement the ProfanityChecker protocol
        """
        # Validate checker first
        try:
            # Test required methods to ensure it implements the protocol
            checker.contains_profanity("test")
            checker.censor("test")
            _ = checker.profane_words
            checker.profane_words = {"test"}
            _ = checker.censor_char
            checker.censor_char = "*"
        except (AttributeError, TypeError) as e:
            raise ValueError(
                f"Checker must implement ProfanityChecker protocol, got {type(checker)}: {e}"
            )

        # Create config if not provided
        if config is None:
            config = ClassifierConfig(
                labels=cls.DEFAULT_LABELS, cost=cls.DEFAULT_COST, params=kwargs.pop("params", {})
            )

        # Create instance with validated checker
        instance = cls(
            name=name,
            description=description,
            checker=checker,
            config=config,
            **kwargs,
        )

        # Initialize the instance
        instance.warm_up()

        return instance


def create_profanity_classifier(
    name: str = "profanity_classifier",
    description: str = "Detects profanity and inappropriate language",
    custom_words: Optional[List[str]] = None,
    censor_char: str = "*",
    min_confidence: float = 0.5,
    cache_size: int = 0,
    cost: int = 1,
    **kwargs: Any,
) -> ProfanityClassifier:
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
        Configured ProfanityClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.implementations.content.profanity import create_profanity_classifier

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
    # Create and return classifier using the class factory method
    return ProfanityClassifier.create(
        name=name,
        description=description,
        custom_words=custom_words,
        censor_char=censor_char,
        min_confidence=min_confidence,
        cache_size=cache_size,
        cost=cost,
        **kwargs,
    )
