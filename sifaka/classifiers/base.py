"""
Base classes for Sifaka classifiers.

This module provides the core interfaces and base implementations for text classification,
including protocols, configuration classes, result types, and the base classifier class.
Classifiers work alongside rules and critics to provide a complete validation and improvement system.

## Integration with Rules and Critics

Classifiers complement rules and critics in the following ways:
- Rules provide binary validation (pass/fail)
- Critics provide nuanced feedback and improvement suggestions
- Classifiers provide semantic understanding and categorization
- Rules can use classifier outputs for content validation
- Critics can use classifier outputs to guide improvements
- Classifiers can trigger rule violations or critic improvements

## Architecture Overview

The classifier system follows a layered architecture:

1. **BaseClassifier**: High-level interface for classification
2. **ClassifierConfig**: Configuration and settings management
3. **ClassificationResult**: Standardized result format
4. **Protocol Classes**: Interface definitions for classifiers and processors

## Component Lifecycle

### ClassifierConfig
1. **Creation**: Instantiate with labels and optional parameters
2. **Validation**: Values are validated in __post_init__
3. **Modification**: Create new instances with with_options() or with_params()
4. **Usage**: Pass to classifiers for configuration

### ClassificationResult
1. **Creation**: Instantiate with label, confidence, and optional metadata
2. **Access**: Read label, confidence, and metadata properties
3. **Enhancement**: Create new instances with additional metadata using with_metadata()

### BaseClassifier
1. **Initialization**: Set up with name, description, and config
2. **Configuration**: Define classification parameters
3. **Warm-up**: Optionally prepare resources with warm_up()
4. **Classification**: Process inputs with classify() or batch_classify()
5. **Caching**: Automatically cache results if configured

## Error Handling Patterns

The classifier system implements several error handling patterns:

1. **Input Validation**: Validates all inputs before processing
   - Checks input types with validate_input()
   - Handles empty text gracefully
   - Validates batch inputs with validate_batch_input()

2. **Classification Errors**: Handles errors during classification
   - Catches exceptions in _classify_impl
   - Returns valid results even when errors occur
   - Includes error details in metadata

3. **Confidence Thresholds**: Uses confidence scores for reliability
   - Sets minimum confidence thresholds in configuration
   - Allows filtering results based on confidence
   - Provides confidence scores for all classifications

## Usage Examples

```python
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig

# Creating a simple classifier
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

# Error handling example
try:
    result = classifier.classify(123)  # Wrong type
except ValueError as e:
    print(f"Validation error: {e}")

# Batch classification
texts = ["Short text", "This is a much longer text that will be classified differently"]
results = classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text[:20]}... - Label: {result.label}, Confidence: {result.confidence:.2f}")
```

## Instantiation Pattern

The recommended way to create classifiers is through the create() factory method:

```python
from sifaka.classifiers.toxicity import ToxicityClassifier

# Create a classifier using the factory method
classifier = ToxicityClassifier.create(
    name="toxicity_classifier",
    description="Classifies text as toxic or non-toxic",
    labels=["toxic", "non-toxic"],
    cache_size=100,
    min_confidence=0.7
)
```

Each classifier type may also provide specialized factory functions for easier instantiation.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

from sifaka.utils.logging import get_logger

from .config import ClassifierConfig
from .models import ClassificationResult

logger = get_logger(__name__)

T = TypeVar("T")  # Type of input text
R = TypeVar("R")  # Type of classification result
C = TypeVar("C", bound="BaseClassifier")  # Type for the classifier class


# TextProcessor is now imported from interfaces/classifier.py


# ClassifierProtocol is now imported from interfaces/classifier.py


# ClassifierConfig is now imported from config.py


# ClassificationResult is now imported from models.py


C = TypeVar("C", bound="BaseClassifier")


class BaseClassifier(ABC, BaseModel, Generic[T, R]):
    """
    Base class for all Sifaka classifiers.

    A classifier provides predictions that can be used by rules and other components
    in the Sifaka framework. This abstract base class defines the core interface
    and functionality for all classifiers, implementing common patterns for
    input validation, caching, and result handling.

    ## Architecture

    BaseClassifier follows a layered architecture:
    1. **Public API**: classify() and batch_classify() methods
    2. **Caching Layer**: _classify_impl() handles caching
    3. **Core Logic**: _classify_impl_uncached() implements classification logic
    4. **Validation**: validate_input() and validate_batch_input() ensure valid inputs

    ## Lifecycle

    1. **Initialization**: Set up with name, description, and config
       - Create with required parameters
       - Initialize internal state
       - Set up caching if enabled

    2. **Warm-up**: Optionally prepare resources
       - Call warm_up() to load models or resources
       - Prepare any expensive resources before classification
       - This step is optional but recommended for performance

    3. **Classification**: Process inputs
       - Call classify() for single inputs
       - Call batch_classify() for multiple inputs
       - Results are cached if caching is enabled

    4. **Result Handling**: Process classification results
       - Check confidence against min_confidence threshold
       - Extract label and metadata
       - Make decisions based on classification

    5. **Cleanup**: No explicit cleanup needed for most classifiers
       - Some implementations may manage external resources
       - Those implementations should provide cleanup methods

    ## Error Handling

    The class implements these error handling patterns:
    - Input validation with validate_input() and validate_batch_input()
    - Empty text handling in classify()
    - Type checking for inputs
    - Exception handling in classification methods
    - Confidence thresholds for reliability

    ## Examples

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

    Using the factory method for creation:

    ```python
    from sifaka.classifiers.base import BaseClassifier, ClassificationResult

    class KeywordClassifier(BaseClassifier[str, str]):
        def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
            # Implementation details...
            pass

    # Create using the factory method
    classifier = KeywordClassifier.create(
        name="keyword_classifier",
        description="Classifies text based on keywords",
        labels=["tech", "sports", "politics", "entertainment"],
        cache_size=200,
        min_confidence=0.6
    )
    ```

    Handling empty inputs and errors:

    ```python
    from sifaka.classifiers.base import BaseClassifier, ClassificationResult
    import logging

    logger = logging.getLogger(__name__)

    class RobustClassifier(BaseClassifier[str, str]):
        def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
            try:
                # Core classification logic
                # ...

                # Return result
                return ClassificationResult(
                    label="some_label",
                    confidence=0.8,
                    metadata={"processed_successfully": True}
                )
            except Exception as e:
                # Log the error
                logger.error(f"Classification error: {e}")

                # Return a fallback result
                return ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
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
        the core classification functionality. It contains the actual
        classification algorithm and is called by _classify_impl when
        no cached result is available.

        ## Lifecycle

        1. **Invocation**: Called by _classify_impl
           - Receives validated input text
           - Called only when no cached result is available
           - Input has already been validated

        2. **Processing**: Apply classification algorithm
           - Implement the core classification logic
           - Process the text according to the classifier's purpose
           - Calculate confidence scores

        3. **Result Creation**: Return standardized result
           - Create a ClassificationResult with label and confidence
           - Include relevant metadata
           - Handle all errors internally

        ## Error Handling

        Implementations should follow these error handling patterns:
        - Catch and handle all exceptions internally
        - Return a valid ClassificationResult even on errors
        - Set confidence=0 for failed classifications
        - Include error details in metadata
        - Log errors for debugging

        ## Implementation Guidelines

        1. **Robust Implementation**:
           ```python
           def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
               try:
                   # Core classification logic
                   # ...
                   return ClassificationResult(
                       label="some_label",
                       confidence=0.8,
                       metadata={"processed_successfully": True}
                   )
               except Exception as e:
                   logger.error(f"Classification error: {e}")
                   return ClassificationResult(
                       label="unknown",
                       confidence=0.0,
                       metadata={"error": str(e)}
                   )
           ```

        2. **Performance Considerations**:
           - Consider implementing timeouts for expensive operations
           - Use efficient algorithms for text processing
           - Consider batching operations when possible

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

        This is the main method for classifying single inputs. It handles input
        validation, empty text checking, and delegates to the internal implementation
        methods for the actual classification work.

        ## Lifecycle

        1. **Input Validation**: Check input validity
           - Validate input type with validate_input()
           - Check for empty text
           - Return early with "unknown" label for empty text

        2. **Classification**: Process the input
           - Delegate to _classify_impl for actual classification
           - _classify_impl handles caching and delegates to _classify_impl_uncached
           - The implementation method performs the core classification logic

        3. **Result Return**: Return the classification result
           - Return ClassificationResult with label, confidence, and metadata
           - Result can be used for decision making in the application

        ## Error Handling

        This method implements these error handling patterns:
        - Input validation with validate_input()
        - Special handling for empty text
        - Propagation of implementation errors
        - Structured error information in results

        ## Examples

        Basic usage:

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

        Handling empty text:

        ```python
        # Empty text handling
        result = classifier.classify("")
        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert result.metadata.get("reason") == "empty_input"
        ```

        Error handling:

        ```python
        try:
            # This will raise ValueError for non-string input
            result = classifier.classify(123)
        except ValueError as e:
            print(f"Invalid input: {e}")
            # Handle the error appropriately
        ```

        Confidence thresholds:

        ```python
        # Using min_confidence from the classifier
        result = classifier.classify("Some text to classify")

        if result.confidence >= classifier.min_confidence:
            print(f"Reliable classification: {result.label}")
            # Take action based on the classification
        else:
            print(f"Low confidence classification: {result.label} ({result.confidence:.2f})")
            # Handle low confidence case (e.g., manual review)
        ```

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details

        Raises:
            ValueError: If input validation fails
        """
        self.validate_input(text)
        if isinstance(text, str):
            from sifaka.utils.text import handle_empty_text_for_classifier

            empty_result = handle_empty_text_for_classifier(text)
            if empty_result:
                return empty_result
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
        cls: Type[C], name: str, description: str, labels: List[str], **config_kwargs: Any
    ) -> C:
        """
        Factory method to create a classifier instance.

        This method provides a consistent way to create classifier instances
        with proper configuration. It simplifies the instantiation process
        by handling the creation of the ClassifierConfig object and setting
        up the classifier with the appropriate parameters.

        ## Lifecycle

        1. **Parameter Processing**: Process input parameters
           - Extract required parameters (name, description, labels)
           - Extract optional configuration parameters
           - Separate params dictionary from other config options

        2. **Configuration Creation**: Create configuration object
           - Create ClassifierConfig with provided parameters
           - Set up labels list
           - Configure cache_size, min_confidence, etc.
           - Include classifier-specific params

        3. **Instance Creation**: Create classifier instance
           - Instantiate the classifier class
           - Pass name, description, and config
           - Return the configured instance

        ## Error Handling

        This method handles these error cases:
        - Parameter validation (delegated to ClassifierConfig)
        - Type checking for critical parameters
        - Proper extraction of params dictionary

        ## Examples

        Basic usage:

        ```python
        from sifaka.classifiers.base import BaseClassifier

        class MyClassifier(BaseClassifier[str, str]):
            def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
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

        Creating with specialized parameters:

        ```python
        # Create a classifier with specific configuration
        sentiment_classifier = SentimentClassifier.create(
            name="sentiment",
            description="Analyzes text sentiment",
            labels=["positive", "negative", "neutral"],
            cache_size=200,
            min_confidence=0.7,
            params={
                "model_name": "sentiment-large",
                "use_gpu": True,
                "batch_size": 16
            }
        )
        ```

        Creating multiple classifiers with different configurations:

        ```python
        # Create classifiers with different confidence thresholds
        high_precision = ToxicityClassifier.create(
            name="toxicity_precise",
            description="High-precision toxicity detection",
            labels=["toxic", "non-toxic"],
            min_confidence=0.9  # High threshold for precision
        )

        high_recall = ToxicityClassifier.create(
            name="toxicity_sensitive",
            description="High-recall toxicity detection",
            labels=["toxic", "non-toxic"],
            min_confidence=0.3  # Low threshold for recall
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


# Create type aliases for common classifier types
TextClassifier = BaseClassifier[str, str]

# Create a more intuitive alias to avoid confusion
Classifier = BaseClassifier

# Export these types
__all__ = [
    "BaseClassifier",
    "Classifier",  # Alias for BaseClassifier
    "TextClassifier",
]
