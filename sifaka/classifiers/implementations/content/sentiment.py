"""
Sentiment classifier using VADER.

This module provides a sentiment classifier that uses the VADER (Valence Aware Dictionary
and sEntiment Reasoner) lexicon-based sentiment analysis tool. It categorizes text into
positive, neutral, negative, or unknown sentiment categories with confidence scores.

## Overview
The SentimentClassifier is a specialized classifier that leverages VADER, a lexicon and
rule-based sentiment analysis tool specifically attuned to sentiments expressed in social
media and conversational text. It provides fast, accurate sentiment analysis without
requiring training data or external API calls.

## Architecture
SentimentClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements sentiment analysis
4. **State Management**: Uses StateManager for internal state
5. **Thresholds**: Configurable thresholds for positive/negative sentiment
6. **Analyzer Loading**: On-demand loading of the VADER analyzer

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract thresholds from config.params
   - Set up default values

2. **Warm-up**: Load VADER resources
   - Load VADER analyzer when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply VADER sentiment analysis
   - Convert scores to standardized format
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map compound scores to sentiment labels
   - Convert scores to confidence values
   - Include detailed scores in metadata

## Usage Examples
```python
from sifaka.classifiers.implementations.content.sentiment import create_sentiment_classifier

# Create a sentiment classifier with default settings
classifier = create_sentiment_classifier()

# Classify text
result = classifier.classify("I love this product! It's amazing.")
print(f"Sentiment: {result.label}, Confidence: {result.confidence:.2f}")

# Create a classifier with custom thresholds
custom_classifier = create_sentiment_classifier(
    positive_threshold=0.1,  # More strict positive threshold
    negative_threshold=-0.1,  # More strict negative threshold
    cache_size=100           # Enable caching
)

# Batch classify multiple texts
texts = [
    "This is fantastic!",
    "I'm not sure about this.",
    "This is terrible."
]
results = custom_classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result.label}, Confidence: {result.confidence:.2f}")
    print(f"Compound score: {result.metadata['compound_score']:.2f}")

# Access all sentiment scores
result = classifier.classify("The product is good but the service was terrible.")
print(f"Compound score: {result.metadata['compound_score']:.2f}")
print(f"Positive score: {result.metadata['pos_score']:.2f}")
print(f"Negative score: {result.metadata['neg_score']:.2f}")
print(f"Neutral score: {result.metadata['neu_score']:.2f}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When VADER is not installed
- RuntimeError: When analyzer initialization fails
- Graceful handling of empty or invalid inputs
- Fallback to "unknown" with zero confidence for edge cases

## Configuration
Key configuration options include:
- positive_threshold: Threshold for positive sentiment detection (default: 0.05)
- negative_threshold: Threshold for negative sentiment detection (default: -0.05)
- cache_size: Size of the classification cache (0 to disable)
"""

import importlib
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    ClassVar,
    Type,
    TypeVar,
    Tuple,
    runtime_checkable,
)

from typing_extensions import TypeGuard
from pydantic import ConfigDict, PrivateAttr

from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config import ClassifierConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state
from sifaka.utils.config import extract_classifier_config_params

logger = get_logger(__name__)

# Type variable for the classifier
T = TypeVar("T", bound="SentimentClassifier")


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """
    Protocol for sentiment analysis engines.

    This protocol defines the interface that any sentiment analyzer must implement
    to be compatible with the SentimentClassifier. It requires a polarity_scores
    method that returns a dictionary of sentiment scores.

    ## Architecture
    The protocol follows a standard interface pattern:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Defines a single required method with clear input/output contract
    - Enables pluggable sentiment analysis implementations

    ## Implementation Requirements
    1. Implement polarity_scores() method that accepts a string and returns a dict
    2. The returned dict should contain at least a 'compound' score
    3. Scores should be in the range [-1, 1] where:
       - Positive values indicate positive sentiment
       - Negative values indicate negative sentiment
       - Values near zero indicate neutral sentiment
    4. Additional scores like 'pos', 'neg', and 'neu' are recommended

    ## Examples
    ```python
    from sifaka.classifiers.implementations.content.sentiment import SentimentAnalyzer

    class CustomAnalyzer:
        def polarity_scores(self, text: str) -> Dict[str, float]:
            # Simple implementation based on keywords
            positive_words = ["good", "great", "excellent"]
            negative_words = ["bad", "terrible", "awful"]

            text_lower = text.lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)

            # Calculate a compound score between -1 and 1
            if pos_count + neg_count == 0:
                compound = 0.0
            else:
                compound = (pos_count - neg_count) / (pos_count + neg_count)

            return {
                "compound": compound,
                "pos": max(0, compound),
                "neg": max(0, -compound),
                "neu": 1.0 - abs(compound)
            }

    # Verify protocol compliance
    analyzer = CustomAnalyzer()
    assert isinstance(analyzer, SentimentAnalyzer)

    # Use with SentimentClassifier
    classifier = SentimentClassifier(analyzer=analyzer)
    result = classifier.classify("This is great!")
    ```
    """

    @abstractmethod
    def polarity_scores(self, text: str) -> Dict[str, float]: ...


class SentimentClassifier(Classifier):
    """
    A lightweight sentiment classifier using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
    sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
    It analyzes text and returns sentiment scores along with a classification label.

    ## Architecture
    SentimentClassifier follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses VADER for sentiment analysis
    - Implements configurable thresholds for sentiment categories
    - Provides detailed sentiment scores in result metadata
    - Uses StateManager for efficient state tracking and caching
    - Supports both synchronous and batch classification

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract thresholds from config.params
       - Set up default values and constants

    2. **Warm-up**: Load VADER resources
       - Load VADER analyzer when needed (lazy initialization)
       - Initialize only once and cache for reuse
       - Handle initialization errors gracefully with clear messages

    3. **Classification**: Process input text
       - Validate input text and handle edge cases
       - Apply VADER sentiment analysis
       - Convert scores to standardized format
       - Apply thresholds to determine sentiment categories
       - Handle empty text with special case handling

    4. **Result Creation**: Return standardized results
       - Map compound scores to appropriate sentiment labels
       - Convert scores to confidence values
       - Include detailed scores in metadata for transparency
       - Track statistics for monitoring and debugging

    ## Examples
    ```python
    from sifaka.classifiers.implementations.content.sentiment import create_sentiment_classifier

    # Create a sentiment classifier with default settings
    classifier = create_sentiment_classifier()

    # Classify text
    result = classifier.classify("I love this product! It's amazing.")
    print(f"Sentiment: {result.label}, Confidence: {result.confidence:.2f}")

    # Access detailed scores
    print(f"Compound score: {result.metadata['compound_score']:.2f}")
    print(f"Positive score: {result.metadata['pos_score']:.2f}")
    print(f"Negative score: {result.metadata['neg_score']:.2f}")
    print(f"Neutral score: {result.metadata['neu_score']:.2f}")

    # Batch classify multiple texts
    texts = ["This is great!", "I hate this", "Maybe it's okay"]
    results = classifier.batch_classify(texts)
    for i, result in enumerate(results):
        print(f"Text {i+1}: {result.label} ({result.confidence:.2f})")
    ```

    ## Configuration Options
    - positive_threshold: Threshold for positive sentiment (default: 0.05)
    - negative_threshold: Threshold for negative sentiment (default: -0.05)
    - cache_size: Size of the classification cache (0 to disable)
    - analyzer: Custom sentiment analyzer implementation

    Requires the 'sentiment' extra to be installed:
    pip install sifaka[sentiment]
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["positive", "neutral", "negative", "unknown"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for lexicon-based analysis

    # Default thresholds
    DEFAULT_POSITIVE_THRESHOLD: ClassVar[float] = 0.05
    DEFAULT_NEGATIVE_THRESHOLD: ClassVar[float] = -0.05

    # State management already inherited from BaseClassifier as _state

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        analyzer: Optional[SentimentAnalyzer] = None,
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the sentiment classifier.

        This method sets up the classifier with the provided name, description,
        and configuration. If no configuration is provided, it creates a default
        configuration with sensible defaults for sentiment analysis.

        Args:
            name: The name of the classifier for identification and logging
            description: Human-readable description of the classifier's purpose
            analyzer: Custom sentiment analyzer implementation that follows the
                     SentimentAnalyzer protocol
            config: Optional classifier configuration with settings like thresholds,
                   cache size, and labels
            **kwargs: Additional configuration parameters that will be extracted
                     and added to the config.params dictionary
        """
        # Create config if not provided
        if config is None:
            # Extract thresholds from kwargs
            params = kwargs.pop("params", {})
            params.update(
                {
                    "positive_threshold": kwargs.pop(
                        "positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD
                    ),
                    "negative_threshold": kwargs.pop(
                        "negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD
                    ),
                }
            )

            # Create config with remaining kwargs
            config = ClassifierConfig[str](
                labels=self.DEFAULT_LABELS, cost=self.DEFAULT_COST, params=params, **kwargs
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize state - this is now handled by BaseClassifier in model_post_init
        # Store thresholds in state
        cache = self._state_manager.get("cache", {})
        cache["positive_threshold"] = self.config.params.get(
            "positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD
        )
        cache["negative_threshold"] = self.config.params.get(
            "negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD
        )
        self._state_manager.update("cache", cache)

        # Store analyzer in state if provided
        if analyzer is not None and self._validate_analyzer(analyzer):
            cache = self._state_manager.get("cache", {})
            cache["analyzer"] = analyzer
            self._state_manager.update("cache", cache)

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[SentimentAnalyzer]:
        """
        Validate that an analyzer implements the required protocol.

        This method checks if the provided analyzer implements the SentimentAnalyzer
        protocol, which requires a polarity_scores() method that returns sentiment scores.
        It uses the validate_component helper method from the base class.

        Args:
            analyzer: The analyzer object to validate, which should implement
                     the SentimentAnalyzer protocol

        Returns:
            True if the analyzer is valid and implements the required protocol

        Raises:
            ValueError: If the analyzer doesn't implement the SentimentAnalyzer protocol
                       or is missing required methods
        """
        return self.validate_component(analyzer, SentimentAnalyzer, "Analyzer")

    def _load_vader(self) -> SentimentAnalyzer:
        """
        Load the VADER sentiment analyzer.

        This method dynamically imports the VADER package and initializes
        the sentiment analyzer. It handles import errors gracefully with
        clear installation instructions and provides detailed error messages
        for troubleshooting.

        Returns:
            Initialized VADER analyzer that implements the SentimentAnalyzer protocol

        Raises:
            ImportError: If the VADER package is not installed, with instructions
                        on how to install it
            RuntimeError: If VADER initialization fails due to loading errors
                         or other runtime problems
        """
        try:
            # Check if analyzer is already in state
            if self._state_manager.get("cache", {}).get("analyzer"):
                return self._state_manager.get("cache")["analyzer"]

            vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
            analyzer = vader_module.SentimentIntensityAnalyzer()

            # Validate and store in state
            if self._validate_analyzer(analyzer):
                cache = self._state_manager.get("cache", {})
                cache["analyzer"] = analyzer
                self._state_manager.update("cache", cache)
                return analyzer

        except ImportError:
            raise ImportError(
                "VADER package is required for SentimentClassifier. "
                "Install it with: pip install sifaka[sentiment]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load VADER: {e}")

    def warm_up(self) -> None:
        """
        Initialize the sentiment analyzer if needed.

        This method loads the VADER analyzer if it hasn't been loaded yet.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources for faster first-time classification.

        The method ensures that initialization happens only once and handles
        errors gracefully with detailed error messages.

        Raises:
            RuntimeError: If analyzer initialization fails
        """
        if not self._state_manager.get("initialized", False):
            try:
                # Load analyzer
                analyzer = self._load_vader()

                # Store in state
                cache = self._state_manager.get("cache", {})
                cache["analyzer"] = analyzer
                self._state_manager.update("cache", cache)

                # Mark as initialized
                self._state_manager.update("initialized", True)
            except Exception as e:
                logger.error("Failed to initialize sentiment analyzer: %s", e)
                self._state_manager.update("error", f"Failed to initialize sentiment analyzer: {e}")
                raise RuntimeError(f"Failed to initialize sentiment analyzer: {e}") from e

    def _get_sentiment_label(self, compound_score: float) -> Tuple[str, float]:
        """
        Get sentiment label and confidence based on compound score.

        This method analyzes the compound sentiment score from VADER and determines
        the most appropriate sentiment label and confidence level based on configured
        thresholds. It maps the compound score to a standardized label and converts
        the score to a confidence value.

        Args:
            compound_score: The compound sentiment score from VADER (-1 to 1),
                           where positive values indicate positive sentiment,
                           negative values indicate negative sentiment, and
                           values near zero indicate neutral sentiment

        Returns:
            Tuple of (sentiment_label, confidence) where:
            - sentiment_label is one of "positive", "negative", "neutral", or "unknown"
            - confidence is a value between 0.0 and 1.0 representing the confidence
              in the sentiment classification
        """
        # Get thresholds from state
        cache = self._state_manager.get("cache", {})
        positive_threshold = cache.get("positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD)
        negative_threshold = cache.get("negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD)

        # Determine sentiment label
        if compound_score >= positive_threshold:
            label = "positive"
        elif compound_score <= negative_threshold:
            label = "negative"
        else:
            label = "neutral"

        # Convert compound score from [-1, 1] to confidence [0, 1]
        confidence = abs(compound_score)

        return label, confidence

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[Any, str]:
        """
        Implement sentiment classification logic without caching.

        This method contains the core sentiment analysis logic using VADER.
        It is called by the caching layer when a cache miss occurs. The method
        handles the entire classification process, from text validation to
        result creation, including error handling and statistics tracking.

        Args:
            text: The text to classify, which can be any string content
                 that the VADER analyzer can process

        Returns:
            ClassificationResult with sentiment scores, containing:
            - label: The sentiment label (positive, neutral, negative, or unknown)
            - confidence: A confidence score between 0.0 and 1.0
            - metadata: Detailed sentiment scores including compound_score,
                       pos_score, neg_score, and neu_score

        Raises:
            RuntimeError: If the sentiment analyzer is not initialized
        """
        # Ensure resources are initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Handle empty or whitespace-only text
        from sifaka.utils.text import handle_empty_text_for_classifier

        empty_result = handle_empty_text_for_classifier(
            text,
            metadata={
                "compound_score": 0.0,
                "pos_score": 0.0,
                "neg_score": 0.0,
                "neu_score": 1.0,
            },
        )
        if empty_result:
            return empty_result

        try:
            # Get analyzer from state
            analyzer = self._state_manager.get("cache", {}).get("analyzer")
            if not analyzer:
                raise RuntimeError("Sentiment analyzer not initialized")

            # Get sentiment scores from VADER
            scores = analyzer.polarity_scores(text)
            compound_score = scores["compound"]

            # Determine sentiment label and confidence
            label, confidence = self._get_sentiment_label(compound_score)

            # Special case for unknown
            if label == "unknown":
                confidence = 0.0

            # Return result with detailed metadata
            result = ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={
                    "compound_score": compound_score,
                    "pos_score": scores["pos"],
                    "neg_score": scores["neg"],
                    "neu_score": scores["neu"],
                },
            )

            # Track statistics
            stats = self._state_manager.get("statistics", {})
            stats[label] = stats.get(label, 0) + 1
            self._state_manager.update("statistics", stats)

            return result
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text sentiment: %s", e)

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

            return ClassificationResult[str](
                label="unknown",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "compound_score": 0.0,
                    "pos_score": 0.0,
                    "neg_score": 0.0,
                    "neu_score": 1.0,
                },
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, cache information, and thresholds.
        It aggregates data from the state manager to provide a comprehensive view of the
        classifier's performance and usage patterns.

        Returns:
            Dictionary containing statistics including:
            - classifications: Counts of classifications by sentiment label
            - error_count: Number of errors encountered
            - cache_enabled: Whether caching is enabled
            - cache_size: Maximum cache size
            - positive_threshold: Current positive sentiment threshold
            - negative_threshold: Current negative sentiment threshold
            - initialized: Whether the analyzer has been initialized
        """
        stats = {
            # Classification counts by label
            "classifications": self._state_manager.get("statistics", {}),
            # Number of errors encountered
            "error_count": len(self._state_manager.get("errors", [])),
            # Cache information
            "cache_enabled": self.config.cache_size > 0,
            "cache_size": self.config.cache_size,
            # State initialization status
            "initialized": self._state_manager.get("initialized", False),
            # Threshold information
            "positive_threshold": self._state_manager.get("cache", {}).get(
                "positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD
            ),
            "negative_threshold": self._state_manager.get("cache", {}).get(
                "negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD
            ),
        }

        # Add cache hit ratio if caching is enabled
        if hasattr(self, "_result_cache"):
            stats["cache_entries"] = len(self._result_cache)

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and resets statistics in the state
        but preserves the analyzer and initialization status.
        """
        # Clear classification result cache
        if hasattr(self, "_result_cache"):
            self._result_cache.clear()

        # Reset statistics
        self._state_manager.update("statistics", {})

        # Reset errors list but keep analyzer and initialization status
        self._state_manager.update("errors", [])

        # Keep the analyzer and thresholds in cache
        cache = self._state_manager.get("cache", {})
        preserved_cache = {
            k: v
            for k, v in cache.items()
            if k in ("analyzer", "positive_threshold", "negative_threshold")
        }
        self._state_manager.update("cache", preserved_cache)

    @classmethod
    def create(
        cls: Type[T],
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        labels: Optional[List[str]] = None,
        cache_size: int = 0,
        min_confidence: float = 0.0,
        cost: Optional[float] = None,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Factory method to create a sentiment classifier.

        Args:
            name: Name of the classifier
            description: Description of what this classifier does
            labels: Optional list of labels (defaults to predefined labels)
            cache_size: Size of the classification cache (0 for no caching)
            min_confidence: Minimum confidence threshold for results
            cost: Computational cost metric (defaults to class default)
            positive_threshold: Threshold for positive sentiment detection
            negative_threshold: Threshold for negative sentiment detection
            params: Optional dictionary of additional parameters
            **kwargs: Additional configuration parameters

        Returns:
            Configured SentimentClassifier instance
        """
        from sifaka.utils.config import extract_classifier_config_params

        # Set up default params with thresholds
        default_params = {
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
        }

        # Extract and merge configuration parameters
        config_dict = extract_classifier_config_params(
            labels=labels if labels else cls.DEFAULT_LABELS,
            cache_size=cache_size,
            min_confidence=min_confidence,
            cost=cost if cost is not None else cls.DEFAULT_COST,
            provided_params=params,
            default_params=default_params,
            **kwargs,
        )

        # Create config with merged parameters
        config = ClassifierConfig[str](**config_dict)

        # Create and return the classifier instance
        return cls(name=name, description=description, config=config)

    @classmethod
    def create_with_custom_analyzer(
        cls: Type[T],
        analyzer: SentimentAnalyzer,
        name: str = "custom_sentiment_classifier",
        description: str = "Custom sentiment analyzer",
        **kwargs: Any,
    ) -> T:
        """
        Factory method to create a classifier with a custom analyzer.

        This method creates a new instance of the classifier with a custom
        sentiment analyzer implementation, which can be useful for testing
        or specialized sentiment analysis needs.

        Args:
            analyzer: Custom sentiment analyzer implementation
            name: Name of the classifier
            description: Description of the classifier
            **kwargs: Additional configuration parameters

        Returns:
            Configured SentimentClassifier instance

        Raises:
            ValueError: If the analyzer doesn't implement the SentimentAnalyzer protocol
        """
        # Validate analyzer first
        if not isinstance(analyzer, SentimentAnalyzer):
            raise ValueError(
                f"Analyzer must implement SentimentAnalyzer protocol, got {type(analyzer)}"
            )

        # Create instance with validated analyzer and kwargs
        instance = cls(
            name=name,
            description=description,
            analyzer=analyzer,
            **kwargs,
        )

        # Set the analyzer and mark as initialized
        cache = instance._state_manager.get("cache", {})
        cache["analyzer"] = analyzer
        instance._state_manager.update("cache", cache)
        instance._state_manager.update("initialized", True)

        return instance


def create_sentiment_classifier(
    name: str = "sentiment_classifier",
    description: str = "Analyzes text sentiment using VADER",
    positive_threshold: float = 0.05,
    negative_threshold: float = -0.05,
    cache_size: int = 0,
    min_confidence: float = 0.0,
    cost: int = 1,
    **kwargs: Any,
) -> SentimentClassifier:
    """
    Factory function to create a sentiment classifier.

    This function provides a simpler interface for creating a sentiment classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        positive_threshold: Positive sentiment threshold (default: 0.05)
        negative_threshold: Negative sentiment threshold (default: -0.05)
        cache_size: Size of the classification cache (0 to disable)
        min_confidence: Minimum confidence for classification
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured SentimentClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.sentiment import create_sentiment_classifier

        # Create a sentiment classifier with default settings
        classifier = create_sentiment_classifier()

        # Create a classifier with custom thresholds
        custom_classifier = create_sentiment_classifier(
            positive_threshold=0.1,  # More strict positive threshold
            negative_threshold=-0.1,  # More strict negative threshold
            cache_size=100           # Enable caching
        )
        ```
    """
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "positive_threshold": positive_threshold,
            "negative_threshold": negative_threshold,
        }
    )

    # Create and return classifier using the class factory method
    return SentimentClassifier.create(
        name=name,
        description=description,
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
        **kwargs,
    )
