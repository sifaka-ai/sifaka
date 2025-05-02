"""
Base classes for Sifaka classifiers.

This module provides the core interfaces and base implementations for text classification.

Examples:
    Creating a simple classifier:

    ```python
    from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig

    class SimpleClassifier(BaseClassifier[str, str]):
        def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
            # Simple implementation that checks text length
            if len(text) > 100:
                return ClassificationResult(
                    label="long",
                    confidence=0.9,
                    metadata={"length": len(text)}
                )
            return ClassificationResult(
                label="short",
                confidence=0.9,
                metadata={"length": len(text)}
            )

    # Create the classifier using the factory method
    classifier = SimpleClassifier.create(
        name="length_classifier",
        description="Classifies text as short or long",
        labels=["short", "long"],
        cache_size=100
    )

    # Use the classifier
    result = classifier.classify("Hello world")
    print(f"Label: {result.label}, Confidence: {result.confidence}")
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# No longer using lru_cache directly
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    Callable,
    runtime_checkable,
    cast,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field

# Input and output type vars
T = TypeVar('T')  # Input type (usually str)
R = TypeVar('R')  # Result type

@runtime_checkable
class TextProcessor(Protocol[T, R]):
    """
    Protocol for text processing components.

    This protocol defines the interface for components that process text inputs
    and return structured outputs.

    Examples:
        Implementing a simple text processor:

        ```python
        class SimpleProcessor:
            def process(self, text: str) -> Dict[str, str]:
                return {"length": str(len(text)), "first_char": text[0] if text else ""}

        # Check if it adheres to the protocol
        from typing import runtime_checkable
        assert isinstance(SimpleProcessor(), TextProcessor)
        ```
    """

    def process(self, text: T) -> Dict[str, R]: ...


@runtime_checkable
class ClassifierProtocol(Protocol[T, R]):
    """
    Protocol defining the interface for classifiers.

    This protocol defines the essential methods that all classifiers must implement.

    Examples:
        Checking if an object follows the classifier protocol:

        ```python
        from sifaka.classifiers.toxicity import ToxicityClassifier
        from sifaka.classifiers.base import ClassifierProtocol

        classifier = ToxicityClassifier()
        assert isinstance(classifier, ClassifierProtocol)
        ```
    """

    def classify(self, text: T) -> "ClassificationResult": ...
    def batch_classify(self, texts: List[T]) -> List["ClassificationResult"]: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def min_confidence(self) -> float: ...


@dataclass(frozen=True)
class ClassifierConfig(Generic[T]):
    """
    Immutable configuration for classifiers.

    This class follows the same pattern as RuleConfig.
    All classifier-specific configuration options should be placed in the params dictionary.

    Lifecycle:
        - Create once at classifier initialization
        - Immutable after creation
        - Use with_options() or with_params() to create modified versions

    Examples:
        Creating and using a classifier config:

        ```python
        from sifaka.classifiers.base import ClassifierConfig

        # Create a basic config
        config = ClassifierConfig(
            labels=["positive", "negative", "neutral"],
            cache_size=100,
            cost=5,
            params={
                "model_name": "sentiment-large",
                "threshold": 0.7
            }
        )

        # Create a modified version
        updated_config = config.with_params(threshold=0.8, use_gpu=True)

        # Access configuration values
        print(f"Labels: {config.labels}")
        print(f"Threshold: {updated_config.params['threshold']}")
        ```

    Args:
        labels: List of valid classification labels
        cache_size: Size of the classification result cache (0 to disable)
        cost: Computational cost of using this classifier
        min_confidence: Minimum confidence threshold for valid classifications
        params: Dictionary of classifier-specific parameters
    """

    labels: List[str]
    cache_size: int = 0
    cost: int = 1
    min_confidence: float = 0.5
    params: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, list) or not all(isinstance(l, str) for l in self.labels):
            raise ValueError("labels must be a list of strings")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    def with_options(self, **kwargs: Any) -> "ClassifierConfig[T]":
        """
        Create a new config with updated options.

        This method returns a new configuration object with the specified options
        updated, preserving immutability.

        Examples:
            ```python
            # Original config
            config = ClassifierConfig(labels=["yes", "no"], cache_size=10)

            # Create new config with increased cache size
            larger_cache = config.with_options(cache_size=100)

            assert larger_cache.cache_size == 100
            assert config.cache_size == 10  # Original unchanged
            ```

        Args:
            **kwargs: Key-value pairs of options to update

        Returns:
            A new ClassifierConfig with updated options
        """
        return ClassifierConfig(**{**self.__dict__, **kwargs})

    def with_params(self, **kwargs: Any) -> "ClassifierConfig[T]":
        """
        Create a new config with updated parameters.

        This method returns a new configuration with the params dictionary
        updated with the provided key-value pairs.

        Examples:
            ```python
            config = ClassifierConfig(
                labels=["yes", "no"],
                params={"threshold": 0.5}
            )

            # Update the threshold parameter
            updated = config.with_params(threshold=0.7, use_cache=True)

            assert updated.params["threshold"] == 0.7
            assert updated.params["use_cache"] == True
            assert config.params["threshold"] == 0.5  # Original unchanged
            ```

        Args:
            **kwargs: Key-value pairs to add or update in the params dictionary

        Returns:
            A new ClassifierConfig with updated params
        """
        new_params = {**self.params, **kwargs}
        return ClassifierConfig(
            labels=self.labels,
            cache_size=self.cache_size,
            cost=self.cost,
            min_confidence=self.min_confidence,
            params=new_params,
        )


class ClassificationResult(BaseModel, Generic[R]):
    """
    Result of a classification.

    This class provides an immutable representation of a classification result,
    including the predicted label, confidence score, and additional metadata.

    Examples:
        Creating and using a classification result:

        ```python
        from sifaka.classifiers.base import ClassificationResult

        # Create a result
        result = ClassificationResult(
            label="positive",
            confidence=0.85,
            metadata={
                "scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05},
                "text_length": 120
            }
        )

        # Access the result
        if result.confidence > 0.8:
            print(f"High confidence classification: {result.label}")

        # Add additional metadata
        updated = result.with_metadata(processed_at="2023-07-01T12:34:56")
        print(f"Processing timestamp: {updated.metadata['processed_at']}")
        ```

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)  # Immutable model

    label: str = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def with_metadata(self, **kwargs: Any) -> "ClassificationResult[R]":
        """
        Create a new result with additional metadata.

        This method returns a new result with the metadata dictionary
        updated with the provided key-value pairs.

        Examples:
            ```python
            result = ClassificationResult(label="toxic", confidence=0.92)

            # Add metadata about the classification
            with_details = result.with_metadata(
                category="severe_toxic",
                timestamp="2023-07-01T12:34:56"
            )

            print(f"Category: {with_details.metadata['category']}")
            ```

        Args:
            **kwargs: Key-value pairs to add to the metadata

        Returns:
            A new ClassificationResult with updated metadata
        """
        new_metadata = {**self.metadata, **kwargs}
        return ClassificationResult(
            label=self.label, confidence=self.confidence, metadata=new_metadata
        )


C = TypeVar("C", bound="BaseClassifier")


class BaseClassifier(ABC, BaseModel, Generic[T, R]):
    """
    Base class for all Sifaka classifiers.

    A classifier provides predictions that can be used by rules and other components
    in the Sifaka framework.

    Lifecycle:
        - Initialize with name, description, and config
        - Optionally call warm_up() to prepare resources
        - Use classify() or batch_classify() methods for classification
        - No explicit cleanup needed for most classifiers
        - Some implementations may manage external resources that need cleanup

    Examples:
        Creating a simple classifier implementation:

        ```python
        from sifaka.classifiers.base import (
            BaseClassifier,
            ClassificationResult,
            ClassifierConfig
        )

        class SentimentClassifier(BaseClassifier[str, str]):
            def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
                # Simple sentiment analysis (real impl would be more sophisticated)
                positive_words = ["good", "great", "excellent", "happy"]
                negative_words = ["bad", "terrible", "sad", "awful"]

                text_lower = text.lower()
                pos_count = sum(word in text_lower for word in positive_words)
                neg_count = sum(word in text_lower for word in negative_words)

                if pos_count > neg_count:
                    return ClassificationResult[str](
                        label="positive",
                        confidence=0.7 + (0.3 * (pos_count / (pos_count + neg_count + 1))),
                        metadata={"positive_words": pos_count, "negative_words": neg_count}
                    )
                elif neg_count > pos_count:
                    return ClassificationResult[str](
                        label="negative",
                        confidence=0.7 + (0.3 * (neg_count / (pos_count + neg_count + 1))),
                        metadata={"positive_words": pos_count, "negative_words": neg_count}
                    )
                else:
                    return ClassificationResult[str](
                        label="neutral",
                        confidence=0.6,
                        metadata={"positive_words": pos_count, "negative_words": neg_count}
                    )

            def warm_up(self) -> None:
                # For this simple classifier, no warm-up is needed
                # In real implementations, this might load models or resources
                pass

        # Create and use the classifier
        classifier = SentimentClassifier(
            name="sentiment",
            description="Simple sentiment classifier",
            config=ClassifierConfig[str](
                labels=["positive", "negative", "neutral"],
                cache_size=100
            )
        )

        result = classifier.classify("I had a great day today!")
        print(f"Sentiment: {result.label} (confidence: {result.confidence:.2f})")
        ```

    Type parameters:
        T: The input type (usually str)
        R: The result type
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the classifier", min_length=1)
    description: str = Field(description="Description of the classifier", min_length=1)
    config: ClassifierConfig[T]

    def model_post_init(self, _: Any) -> None:
        """
        Initialize after model creation.

        This method is called after the model is created and can be used for
        additional initialization steps.

        Lifecycle:
            - Called automatically after Pydantic model initialization
            - Used to set up internal state that depends on initial attributes
            - Does not directly acquire external resources (use warm_up for that)

        Args:
            _: Ignored parameter
        """
        # We don't need to do anything special for caching
        # The _classify_impl method will handle caching internally

    @property
    def min_confidence(self) -> float:
        """
        Get the minimum confidence threshold.

        Returns the minimum confidence level for a classification to be considered
        reliable, as defined in the classifier's configuration.

        Returns:
            Minimum confidence threshold (0-1)
        """
        return self.config.min_confidence

    def validate_input(self, text: Any) -> bool:
        """
        Validate input text.

        Ensures that the input is a valid string for classification.

        Error Handling:
            - Raises ValueError for invalid inputs
            - Return True only for valid inputs

        Examples:
            ```python
            classifier = ToxicityClassifier()

            try:
                classifier.validate_input("Valid text")  # Returns True
                classifier.validate_input(123)  # Raises ValueError
            except ValueError as e:
                print(f"Validation error: {e}")
            ```

        Args:
            text: The input to validate

        Returns:
            True if the input is valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return True

    def validate_batch_input(self, texts: Any) -> bool:
        """
        Validate batch input texts.

        Ensures that the batch input is a list of valid strings for classification.

        Error Handling:
            - Raises ValueError for invalid inputs
            - Returns True only for valid inputs

        Examples:
            ```python
            classifier = ToxicityClassifier()

            try:
                classifier.validate_batch_input(["Text1", "Text2"])  # Returns True
                classifier.validate_batch_input("Not a list")  # Raises ValueError
                classifier.validate_batch_input([1, 2, 3])  # Raises ValueError
            except ValueError as e:
                print(f"Validation error: {e}")
            ```

        Args:
            texts: The batch input to validate

        Returns:
            True if the input is valid

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings")
        return True

    @abstractmethod
    def _classify_impl_uncached(self, text: T) -> ClassificationResult[R]:
        """
        Implement classification logic.

        This abstract method must be implemented by subclasses to provide
        the core classification functionality.

        Error Handling:
            - Should handle classification errors internally
            - Should return a valid ClassificationResult even on errors
            - May set confidence=0 and include error details in metadata

        Lifecycle:
            - Called by _classify_impl when no cached result is available
            - Should implement the core classification logic
            - Should handle all errors internally and return a valid result

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _classify_impl_uncached")

    def _classify_impl(self, text: T) -> ClassificationResult[R]:
        """
        Wrapper around _classify_impl_uncached that handles caching.

        This method handles caching of classification results to improve
        performance for repeated classifications.

        Lifecycle:
            - Called by classify() after input validation
            - Checks cache for existing results if caching is enabled
            - Calls _classify_impl_uncached for cache misses
            - Stores results in cache if caching is enabled

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with label and confidence
        """
        # If caching is enabled, use a function-local cache
        if self.config.cache_size > 0:
            # Create a cache key from the text
            cache_key = str(text)

            # Check if we have a cached result
            if not hasattr(self, "_result_cache"):
                # Initialize the cache as a dict mapping strings to ClassificationResults
                self._result_cache: Dict[str, ClassificationResult[R]] = {}

            if cache_key in self._result_cache:
                return self._result_cache[cache_key]

            # Get the result
            result = self._classify_impl_uncached(text)

            # Cache the result
            if len(self._result_cache) >= self.config.cache_size:
                # Simple LRU: just clear the cache when it gets full
                # A more sophisticated implementation would use an OrderedDict
                self._result_cache.clear()
            self._result_cache[cache_key] = result

            return result
        else:
            # No caching, just call the implementation directly
            return self._classify_impl_uncached(text)

    def classify(self, text: T) -> ClassificationResult[R]:
        """
        Classify the input text.

        This is the main method for classifying single inputs.

        Error Handling:
            - Validates input with validate_input()
            - Handles empty text gracefully by returning unknown label
            - Delegates to _classify_impl for actual classification

        Examples:
            ```python
            from sifaka.classifiers.toxicity import create_toxicity_classifier

            # Create a toxicity classifier
            classifier = create_toxicity_classifier()

            # Classify a text
            result = classifier.classify("This product is awesome!")

            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.2f}")

            # Check for high confidence classifications
            if result.confidence > 0.8:
                print(f"High confidence classification: {result.label}")
            else:
                print("Low confidence classification")
            ```

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details

        Raises:
            ValueError: If input validation fails
        """
        self.validate_input(text)
        if isinstance(text, str) and not text.strip():
            return ClassificationResult[R](
                label="unknown", confidence=0.0, metadata={"reason": "empty_input"}
            )
        return self._classify_impl(text)

    def batch_classify(self, texts: List[T]) -> List[ClassificationResult[R]]:
        """
        Classify multiple texts in batch.

        This method allows efficient classification of multiple texts.
        Some implementations may optimize batch processing.

        Error Handling:
            - Validates input with validate_batch_input()
            - Processes each text individually with classify()
            - Returns results even if some classifications fail

        Examples:
            ```python
            from sifaka.classifiers.toxicity import create_toxicity_classifier

            classifier = create_toxicity_classifier()

            # Classify multiple texts
            texts = [
                "I love this product!",
                "This is terrible, I hate it.",
                "The weather is nice today."
            ]

            results = classifier.batch_classify(texts)

            # Process the results
            for text, result in zip(texts, results):
                print(f"Text: {text[:20]}...")
                print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
            ```

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults

        Raises:
            ValueError: If input validation fails
        """
        self.validate_batch_input(texts)
        return [self.classify(text) for text in texts]

    def warm_up(self) -> None:
        """
        Optional warm-up method for classifiers that need initialization.

        This method should be overridden by subclasses that need to load
        models or resources before classification.

        Lifecycle:
            - Not called automatically
            - Should be called before classification if expensive resources are needed
            - Implementations should be idempotent (safe to call multiple times)

        Examples:
            ```python
            from sifaka.classifiers.toxicity import create_toxicity_classifier

            # Create the classifier
            classifier = create_toxicity_classifier()

            # Warm up the classifier (loads the model)
            classifier.warm_up()

            # Now classify (model is already loaded)
            result = classifier.classify("Hello world")
            ```
        """
        pass

    @classmethod
    def create(
        cls: Type[C],
        name: str,
        description: str,
        labels: List[str],
        **config_kwargs: Any
    ) -> C:
        """
        Factory method to create a classifier instance.

        This method provides a consistent way to create classifier instances
        with proper configuration.

        Examples:
            ```python
            from sifaka.classifiers.base import BaseClassifier

            class MyClassifier(BaseClassifier[str, str]):
                # Implementation details...
                pass

            # Create an instance using the factory method
            classifier = MyClassifier.create(
                name="my_classifier",
                description="My custom classifier implementation",
                labels=["label1", "label2", "label3"],
                cache_size=100,
                min_confidence=0.6,
                params={"custom_param": "value"}
            )
            ```

        Args:
            name: Name of the classifier
            description: Description of what this classifier does
            labels: List of valid labels for classification
            **config_kwargs: Additional configuration parameters

        Returns:
            New classifier instance
        """
        # Extract params from config_kwargs if present
        params = config_kwargs.pop("params", {})

        # Create config with remaining kwargs
        config = ClassifierConfig[T](labels=labels, params=params, **config_kwargs)

        # Create instance
        return cls(name=name, description=description, config=config)


# Type alias for external usage
Classifier = BaseClassifier[str, Any]
TextClassifier = BaseClassifier[str, str]

# Export these types
__all__ = [
    "Classifier",
    "TextClassifier",
    "ClassificationResult",
    "ClassifierConfig",
    "ClassifierProtocol",
    "TextProcessor",
]
