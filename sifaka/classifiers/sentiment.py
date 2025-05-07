"""
Sentiment classifier using VADER.

This module provides a sentiment classifier that uses the VADER (Valence Aware Dictionary
and sEntiment Reasoner) lexicon-based sentiment analysis tool. It categorizes text into
positive, neutral, negative, or unknown sentiment categories with confidence scores.

## Architecture

SentimentClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements sentiment analysis
4. **State Management**: Uses StateManager for internal state

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
from sifaka.classifiers.sentiment import create_sentiment_classifier

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
```
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

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, ClassifierState, create_classifier_state

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

    ## Implementation Requirements

    1. Implement polarity_scores() method that accepts a string and returns a dict
    2. The returned dict should contain at least a 'compound' score
    3. Scores should be in the range [-1, 1] where:
       - Positive values indicate positive sentiment
       - Negative values indicate negative sentiment
       - Values near zero indicate neutral sentiment

    ## Examples

    ```python
    from sifaka.classifiers.sentiment import SentimentAnalyzer

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
    ```
    """

    @abstractmethod
    def polarity_scores(self, text: str) -> Dict[str, float]: ...


class SentimentClassifier(BaseClassifier[str, str]):
    """
    A lightweight sentiment classifier using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based
    sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
    It analyzes text and returns sentiment scores along with a classification label.

    ## Architecture

    SentimentClassifier follows the standard Sifaka classifier architecture:
    1. **Public API**: classify() and batch_classify() methods (inherited)
    2. **Caching Layer**: _classify_impl() handles caching (inherited)
    3. **Core Logic**: _classify_impl_uncached() implements sentiment analysis
    4. **State Management**: Uses StateManager for internal state

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

    ## Examples

    ```python
    from sifaka.classifiers.sentiment import create_sentiment_classifier

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
    ```

    Requires the 'sentiment' extra to be installed:
    pip install sifaka[sentiment]
    """

    # Class-level constants
    DEFAULT_LABELS: ClassVar[List[str]] = ["positive", "neutral", "negative", "unknown"]
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for lexicon-based analysis

    # Default thresholds
    DEFAULT_POSITIVE_THRESHOLD: ClassVar[float] = 0.05
    DEFAULT_NEGATIVE_THRESHOLD: ClassVar[float] = -0.05

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

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

        Args:
            name: The name of the classifier
            description: Description of the classifier
            analyzer: Custom sentiment analyzer implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
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

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store thresholds in state
        state.cache["positive_threshold"] = self.config.params.get(
            "positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD
        )
        state.cache["negative_threshold"] = self.config.params.get(
            "negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD
        )

        # Store analyzer in state if provided
        if analyzer is not None and self._validate_analyzer(analyzer):
            state.cache["analyzer"] = analyzer

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[SentimentAnalyzer]:
        """
        Validate that an analyzer implements the required protocol.

        Args:
            analyzer: The analyzer to validate

        Returns:
            True if the analyzer is valid

        Raises:
            ValueError: If the analyzer doesn't implement the SentimentAnalyzer protocol
        """
        if not isinstance(analyzer, SentimentAnalyzer):
            raise ValueError(
                f"Analyzer must implement SentimentAnalyzer protocol, got {type(analyzer)}"
            )
        return True

    def _load_vader(self) -> SentimentAnalyzer:
        """
        Load the VADER sentiment analyzer.

        Returns:
            Initialized VADER analyzer

        Raises:
            ImportError: If the VADER package is not installed
            RuntimeError: If VADER initialization fails
        """
        try:
            # Get state
            state = self._state_manager.get_state()

            # Check if analyzer is already in state
            if "analyzer" in state.cache:
                return state.cache["analyzer"]

            vader_module = importlib.import_module("vaderSentiment.vaderSentiment")
            analyzer = vader_module.SentimentIntensityAnalyzer()

            # Validate and store in state
            if self._validate_analyzer(analyzer):
                state.cache["analyzer"] = analyzer
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
        explicitly to pre-initialize resources.
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            try:
                # Load analyzer
                analyzer = self._load_vader()

                # Store in state
                state.cache["analyzer"] = analyzer

                # Mark as initialized
                state.initialized = True
            except Exception as e:
                logger.error("Failed to initialize sentiment analyzer: %s", e)
                raise RuntimeError(f"Failed to initialize sentiment analyzer: {e}") from e

    def _get_sentiment_label(self, compound_score: float) -> Tuple[str, float]:
        """
        Get sentiment label and confidence based on compound score.

        Args:
            compound_score: The compound sentiment score from VADER (-1 to 1)

        Returns:
            Tuple of (sentiment_label, confidence)
        """
        # Get state
        state = self._state_manager.get_state()

        # Get thresholds from state
        positive_threshold = state.cache.get("positive_threshold", self.DEFAULT_POSITIVE_THRESHOLD)
        negative_threshold = state.cache.get("negative_threshold", self.DEFAULT_NEGATIVE_THRESHOLD)

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

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement sentiment classification logic without caching.

        This method contains the core sentiment analysis logic using VADER.
        It is called by the caching layer when a cache miss occurs.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment scores
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure resources are initialized
        if not state.initialized:
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
            analyzer = state.cache.get("analyzer")
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
            return ClassificationResult[str](
                label=label,
                confidence=confidence,
                metadata={
                    "compound_score": compound_score,
                    "pos_score": scores["pos"],
                    "neg_score": scores["neg"],
                    "neu_score": scores["neu"],
                },
            )
        except Exception as e:
            # Log the error and return a fallback result
            logger.error("Failed to classify text sentiment: %s", e)
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

    @classmethod
    def create(
        cls: Type[T],
        name: str = "sentiment_classifier",
        description: str = "Analyzes text sentiment using VADER",
        labels: Optional[List[str]] = None,
        cache_size: int = 0,
        min_confidence: float = 0.0,
        cost: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Create a new instance with the given parameters.

        This factory method creates a new instance of the classifier with the
        specified parameters, handling the creation of the ClassifierConfig
        object and setting up the classifier with the appropriate parameters.

        Args:
            name: Name of the classifier
            description: Description of the classifier
            labels: List of valid labels
            cache_size: Size of the classification cache (0 to disable)
            min_confidence: Minimum confidence for classification
            cost: Computational cost of this classifier
            params: Additional configuration parameters
            **kwargs: Additional keyword arguments

        Returns:
            A new instance of the classifier
        """
        # Create params dictionary if not provided
        if params is None:
            params = {}

        # Add kwargs to params
        params.update(kwargs)

        # Create config
        config = ClassifierConfig(
            labels=labels or cls.DEFAULT_LABELS,
            cache_size=cache_size,
            min_confidence=min_confidence,
            cost=cost or cls.DEFAULT_COST,
            params=params,
        )

        # Create and return instance
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

        # Initialize state
        state = instance._state.get_state()
        state.cache["analyzer"] = analyzer
        state.initialized = True

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
