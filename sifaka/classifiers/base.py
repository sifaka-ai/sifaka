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

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
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
from dataclasses import dataclass
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
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, create_classifier_state, ClassifierState
from sifaka.utils.errors import (
    ValidationError,
    ConfigurationError,
    ClassifierError,
    format_error_metadata,
    handle_errors,
    with_error_handling,
)

logger = get_logger(__name__)

T = TypeVar("T")  # Type of input text
R = TypeVar("R")  # Type of classification result


@runtime_checkable
class TextProcessor(Protocol[T, R]):

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
    """
    Protocol for text processing components.

    This protocol defines the interface for components that process text inputs
    and return structured outputs. It's used as a common interface for various
    text processing operations in the Sifaka framework.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the process() method
       - Define a method that takes text input and returns a dictionary
       - Ensure the method signature matches the protocol

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use the processor for text processing
       - Pass text to the process() method
       - Receive structured output as a dictionary
       - Use the output for further processing or analysis

    ## Error Handling

    Implementations should handle these error cases:
    - Empty or invalid text inputs
    - Processing failures
    - Resource availability issues

    ## Examples

    Implementing a simple text processor:

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    ```python
    from sifaka.classifiers.base import TextProcessor
    from typing import Dict, runtime_checkable

    class SimpleProcessor:
        def process(self, text: str) -> Dict[str, str]:
            # Handle empty text
            if not text:
                return {"error": "Empty text", "length": "0"}

            # Process the text
            return {
                "length": str(len(text)),
                "first_char": text[0] if text else "",
                "has_digits": str(any(c.isdigit() for c in text))
            }

    # Check if it adheres to the protocol
    processor = SimpleProcessor()
    assert isinstance(processor, TextProcessor)

    # Use the processor
    result = processor.process("Hello 123")
    print(f"Text length: {result['length']}")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
    print(f"First character: {result['first_char']}")
    print(f"Contains digits: {result['has_digits']}")
    ```

    Using with error handling:

    ```python
    from sifaka.classifiers.base import TextProcessor
    from typing import Dict, Optional

    class RobustProcessor:
        def __init__(self):
            self.error_count = 0

        def process(self, text: str) -> Dict[str, str]:
            try:
                if not text:
                    return {"status": "error", "reason": "empty_input"}

                # Process the text
                return {
                    "status": "success",
                    "length": str(len(text)),
                    "words": str(len(text.split()))

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
                }
            except Exception as e:
                self.error_count += 1
                return {
                    "status": "error",
                    "reason": str(e),
                    "error_count": str(self.error_count)
                }
    ```
    """

    def process(self, text: T) -> Dict[str, R]: ...


@runtime_checkable
class ClassifierImplementation(Protocol[T, R]):
    """
    Protocol for classifier implementations.

    This protocol defines the core classification logic that can be composed with
    the Classifier class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the required methods
       - Implement classify_impl() for the core classification logic
       - Implement warm_up_impl() for resource initialization

    2. **Composition**: Use with the Classifier class
       - Classifier delegates to the implementation
       - Implementation focuses on core logic

    3. **Usage**: Implementation is used internally by Classifier

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
       - Not typically used directly by client code
       - Allows for separation of concerns

    ## Error Handling

    Implementations should handle these error cases:
    - Classification failures
    - Resource initialization errors
    - Invalid inputs (after basic validation)

    ## Examples

    Creating a simple implementation:

    ```python
    from sifaka.classifiers.base import ClassifierImplementation, ClassificationResult

    class SimpleImplementation(ClassifierImplementation[str, str]):
        def __init__(self, config):
            self.config = config
            # State is managed by StateManager, no need to initialize here
            # Initialization is handled by StateManager

        def classify_impl(self, text: str) -> ClassificationResult[str]:
            # Simple implementation based on text length
            if len(text) > 100:

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
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

        def warm_up_impl(self) -> None:
            # No special initialization needed
            state.initialized = True
    ```
    """

    def classify_impl(self, text: T) -> "ClassificationResult[R]": ...
    def warm_up_impl(self) -> None: ...


@runtime_checkable
class ClassifierProtocol(Protocol[T, R]):
    """
    Protocol defining the interface for classifiers.

    This protocol defines the essential methods that all classifiers must implement.
    It provides a common interface for all classification components in the Sifaka
    framework, allowing for consistent usage patterns and interchangeability.

    ## Lifecycle

    1. **Implementation**: Create a class that implements all required methods
       - Implement classify() for single text classification
       - Implement batch_classify() for processing multiple texts
       - Provide name, description, and min_confidence properties

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use the classifier for text classification
       - Call classify() with text input
       - Receive ClassificationResult with label and confidence
       - Use batch_classify() for efficient processing of multiple texts

    ## Error Handling

    Implementations should handle these error cases:

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
    - Invalid input types (non-string inputs)
    - Empty text inputs
    - Classification failures
    - Resource availability issues

    ## Examples

    Checking if an object follows the classifier protocol:

    ```python
    from sifaka.classifiers.toxicity import ToxicityClassifier
    from sifaka.classifiers.base import ClassifierProtocol

    classifier = ToxicityClassifier()
    assert isinstance(classifier, ClassifierProtocol)
    ```

    Creating a minimal classifier that implements the protocol:

    ```python
    from sifaka.classifiers.base import ClassifierProtocol, ClassificationResult
    from typing import List

    class MinimalClassifier:
        @property
        def name(self) -> str:
            return "minimal_classifier"

        @property
        def description(self) -> str:
            return "A minimal classifier implementation"

        @property
        def min_confidence(self) -> float:
            return 0.5

        def classify(self, text: str) -> ClassificationResult:
            # Simple implementation based on text length
            if not text:
                return ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    metadata={"reason": "empty_input"}
                )

            if len(text) > 50:
                return ClassificationResult(
                    label="long",
                    confidence=0.9,
                    metadata={"length": len(text)}
                )
            else:
                return ClassificationResult(
                    label="short",
                    confidence=0.8,
                    metadata={"length": len(text)}

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
                )

        def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
            return [self.classify(text) for text in texts]

    # Verify protocol compliance
    classifier = MinimalClassifier()
    assert isinstance(classifier, ClassifierProtocol)

    # Use the classifier
    result = classifier.classify("This is a test")
    print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
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

    This class provides a standardized way to configure classifiers with
    immutable properties. It follows the same pattern as RuleConfig, where
    all classifier-specific configuration options are placed in the params dictionary.

    The immutable design ensures configuration consistency during classifier
    operation and prevents accidental modification of settings.

    ## Lifecycle

    1. **Creation**: Instantiate with required and optional parameters
       - Provide labels list (required)
       - Set cache_size, cost, and min_confidence as needed
       - Add classifier-specific options in params dictionary

    2. **Validation**: Values are validated in __post_init__

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
       - Labels must be a list of strings
       - Cache size must be non-negative
       - Cost must be non-negative
       - Min confidence must be between 0 and 1

    3. **Usage**: Access configuration properties during classification
       - Read labels list for valid classification outputs
       - Use cache_size for result caching
       - Access min_confidence for threshold checks
       - Read classifier-specific params as needed

    4. **Modification**: Create new instances with updated values
       - Use with_options() to update top-level properties
       - Use with_params() to update classifier-specific parameters
       - Original configuration remains unchanged (immutable)

    ## Error Handling

    The class implements these error handling patterns:
    - Validation of all parameters in __post_init__
    - Immutability to prevent runtime configuration errors
    - Type checking for critical parameters
    - Range validation for numeric parameters

    ## Examples

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

    Error handling with validation:

    ```python
    from sifaka.classifiers.base import ClassifierConfig

    try:
        # This will raise an error due to invalid min_confidence
        config = ClassifierConfig(
            labels=["valid", "invalid"],
            min_confidence=1.5  # Must be between 0 and 1
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        # Use default values instead
        config = ClassifierConfig(
            labels=["valid", "invalid"],
            min_confidence=0.7  # Valid value
        )
    ```

    Creating specialized configurations:

    ```python
    from sifaka.classifiers.base import ClassifierConfig

    # Create a config for a high-precision classifier
    high_precision = ClassifierConfig(
        labels=["spam", "ham"],
        min_confidence=0.9,  # Require high confidence
        params={"precision_focused": True}
    )

    # Create a config for a high-recall classifier
    high_recall = ClassifierConfig(
        labels=["spam", "ham"],
        min_confidence=0.3,  # Accept lower confidence
        params={"recall_focused": True}
    )
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
            raise ConfigurationError("labels must be a list of strings")
        if self.cache_size < 0:
            raise ConfigurationError("cache_size must be non-negative")
        if self.cost < 0:
            raise ConfigurationError("cost must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError("min_confidence must be between 0 and 1")

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

        # State management using StateManager
        _state_manager = PrivateAttr(default_factory=create_classifier_state)
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
    The immutable design ensures result consistency and prevents accidental
    modification after classification.

    ## Lifecycle

    1. **Creation**: Instantiate with classification results

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
       - Provide label and confidence (required)
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get classification details
       - Access label for the classification result
       - Check confidence for result reliability
       - Examine metadata for additional information

    3. **Enhancement**: Create new instances with additional metadata
       - Use with_metadata() to add or update metadata
       - Original result remains unchanged (immutable)
       - Chain multiple with_metadata() calls as needed

    4. **Usage**: Use in application logic
       - Check confidence against thresholds
       - Make decisions based on label
       - Extract detailed information from metadata
       - Pass to other components for further processing

    ## Error Handling

    The class implements these error handling patterns:
    - Validation of confidence range (0-1)
    - Immutability to prevent result tampering
    - Type checking for critical parameters
    - Structured metadata for error details

    ## Examples

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

    Error handling with confidence thresholds:

    ```python
    from sifaka.classifiers.base import ClassificationResult

    # Create results with different confidence levels
    high_confidence = ClassificationResult(label="spam", confidence=0.95)
    medium_confidence = ClassificationResult(label="spam", confidence=0.7)
    low_confidence = ClassificationResult(label="spam", confidence=0.3)

    # Apply different handling based on confidence
    def process_result(result):
        if result.confidence > 0.9:
            print(f"Automatic action: {result.label}")
            return "automatic"
        elif result.confidence > 0.6:
            print(f"Suggested action: {result.label}")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
            return "suggested"
        else:
            print(f"Manual review needed: {result.label} ({result.confidence:.2f})")
            return "review"

    # Process the results
    process_result(high_confidence)   # "automatic"
    process_result(medium_confidence) # "suggested"
    process_result(low_confidence)    # "review"
    ```

    Working with metadata:

    ```python
    from sifaka.classifiers.base import ClassificationResult

    # Create a result with detailed metadata
    result = ClassificationResult(
        label="toxic",
        confidence=0.92,
        metadata={
            "category": "hate_speech",
            "severity": "high",
            "keywords": ["offensive_term1", "offensive_term2"]
        }
    )

    # Extract and use metadata
    if result.metadata.get("severity") == "high":
        print(f"High severity {result.label} content detected")
        print(f"Flagged keywords: {', '.join(result.metadata.get('keywords', []))}")

    # Chain metadata additions
    enhanced = result.with_metadata(
        timestamp="2023-07-01T12:34:56"
    ).with_metadata(
        action_taken="content_removed"
    )

    print(f"Action taken: {enhanced.metadata['action_taken']}")
    print(f"Timestamp: {enhanced.metadata['timestamp']}")
    ```

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
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


C = TypeVar("C", bound="Classifier")


class Classifier(BaseModel, Generic[T, R]):

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
    """
    Classifier that uses composition over inheritance.

    This class delegates classification to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.

    ## Architecture

    Classifier follows a compositional architecture:

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)
    1. **Public API**: classify() and batch_classify() methods
    2. **Delegation**: Delegates to implementation for core logic
    3. **Validation**: validate_input() and validate_batch_input() ensure valid inputs
    4. **Configuration**: Manages configuration through ClassifierConfig

    ## Lifecycle

    1. **Initialization**: Set up with name, description, implementation, and config
       - Create with required parameters
       - Store implementation object
       - Set up configuration

    2. **Warm-up**: Prepare resources
       - Call warm_up() to initialize resources
       - Delegates to implementation.warm_up_impl()
       - This step is optional but recommended for performance

    3. **Classification**: Process inputs
       - Call classify() for single inputs
       - Call batch_classify() for multiple inputs
       - Delegates core logic to implementation

    4. **Result Handling**: Process classification results
       - Check confidence against min_confidence threshold
       - Extract label and metadata
       - Make decisions based on classification

    ## Error Handling

    The class implements these error handling patterns:
    - Input validation with validate_input() and validate_batch_input()
    - Empty text handling in classify()
    - Type checking for inputs
    - Exception handling in classification methods
    - Confidence thresholds for reliability

    ## Examples

    Creating a classifier with an implementation:

    ```python
    from sifaka.classifiers.base import (
        Classifier,
        ClassifierImplementation,
        ClassificationResult,
        ClassifierConfig
    )

    # Create an implementation
    class SentimentImplementation(ClassifierImplementation[str, str]):
        def __init__(self, config):
            self.config = config
            # State is managed by StateManager, no need to initialize here
            # Initialization is handled by StateManager

        def classify_impl(self, text: str) -> ClassificationResult[str]:
            # Simple sentiment analysis
            positive_words = ["good", "great", "excellent", "happy"]
            negative_words = ["bad", "terrible", "sad", "awful"]

            text_lower = text.lower()
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)

            if pos_count > neg_count:
                return ClassificationResult(
                    label="positive",
                    confidence=0.8,
                    metadata={"positive_words": pos_count, "negative_words": neg_count}
                )
            elif neg_count > pos_count:
                return ClassificationResult(
                    label="negative",
                    confidence=0.8,
                    metadata={"positive_words": pos_count, "negative_words": neg_count}
                )
            else:
                return ClassificationResult(
                    label="neutral",
                    confidence=0.6,
                    metadata={"positive_words": pos_count, "negative_words": neg_count}
                )

        def warm_up_impl(self) -> None:
            # No special initialization needed
            state.initialized = True

    # Create config
    config = ClassifierConfig(
        labels=["positive", "negative", "neutral"],
        cache_size=100
    )

    # Create implementation
    implementation = SentimentImplementation(config)

    # Create classifier with implementation
    classifier = Classifier(
        name="sentiment",
        description="Simple sentiment classifier",
        config=config,
        implementation=implementation
    )

    # Use the classifier
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
    _implementation: ClassifierImplementation[T, R] = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        config: ClassifierConfig[T],
        implementation: ClassifierImplementation[T, R],
        **kwargs: Any,
    ):
        """
        Initialize the classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            config: Configuration for the classifier
            implementation: Implementation of the classification logic
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

        # Initialize cache if needed
        if self.config.cache_size > 0:
            self._result_cache: Dict[str, ClassificationResult[R]] = {}

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
            - Raises ValidationError for invalid inputs
            - Return True only for valid inputs

        Args:
            text: The input to validate

        Returns:
            True if the input is valid

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        return True

    def validate_batch_input(self, texts: Any) -> bool:
        """
        Validate batch input texts.

        Ensures that the batch input is a list of valid strings for classification.

        Error Handling:
            - Raises ValidationError for invalid inputs
            - Returns True only for valid inputs

        Args:
            texts: The batch input to validate

        Returns:
            True if the input is valid

        Raises:
            ValidationError: If input is invalid
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValidationError("Input must be a list of strings")
        return True

    @handle_errors(reraise=True, log_errors=True)
    def classify(self, text: T) -> ClassificationResult[R]:
        """
        Classify the input text.

        This is the main method for classifying single inputs. It handles input
        validation, empty text checking, and delegates to the implementation
        for the actual classification work.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with prediction details

        Raises:
            ValidationError: If input validation fails
            ClassifierError: If classification fails
        """
        self.validate_input(text)
        if isinstance(text, str) and not text.strip():
            return ClassificationResult[R](
                label="unknown", confidence=0.0, metadata={"reason": "empty_input"}
            )

        try:
            # If caching is enabled, check cache first
            if self.config.cache_size > 0:
                cache_key = str(text)
                if cache_key in self._result_cache:
                    return self._result_cache[cache_key]

                # Get result from implementation
                result = self._implementation.classify_impl(text)

                # Cache the result
                if len(self._result_cache) >= self.config.cache_size:
                    # Simple LRU: just clear the cache when it gets full
                    self._result_cache.clear()
                self._result_cache[cache_key] = result

                return result
            else:
                # No caching, delegate directly to implementation
                return self._implementation.classify_impl(text)
        except Exception as e:
            # Log the error
            logger.error(f"Classification error in {self.name}: {e}")

            # Return a fallback result with standardized error metadata
            return ClassificationResult[R](
                label="unknown", confidence=0.0, metadata=format_error_metadata(e)
            )

    @handle_errors(reraise=True, log_errors=True)
    def batch_classify(self, texts: List[T]) -> List[ClassificationResult[R]]:
        """
        Classify multiple texts in batch.

        This method allows efficient classification of multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults

        Raises:
            ValidationError: If input validation fails
            ClassifierError: If classification fails
        """
        self.validate_batch_input(texts)

        try:
            results = []
            for text in texts:
                try:
                    results.append(self.classify(text))
                except Exception as e:
                    # Log the error but continue processing other texts
                    logger.error(f"Error classifying text in batch: {e}")
                    results.append(
                        ClassificationResult[R](
                            label="unknown", confidence=0.0, metadata=format_error_metadata(e)
                        )
                    )
            return results
        except Exception as e:
            # If the entire batch fails, log and return fallback results for all texts
            logger.error(f"Batch classification error in {self.name}: {e}")
            return [
                ClassificationResult[R](
                    label="unknown", confidence=0.0, metadata=format_error_metadata(e)
                )
                for _ in texts
            ]

    @handle_errors(reraise=True, log_errors=True)
    def warm_up(self) -> None:
        """
        Prepare resources for classification.

        This method delegates to the implementation's warm_up_impl method
        to initialize any resources needed for classification.

        Raises:
            ClassifierError: If initialization fails
        """
        with with_error_handling(f"Initializing classifier {self.name}", logger=logger):
            self._implementation.warm_up_impl()

    @classmethod
    def create(
        cls: Type[C],
        name: str,
        description: str,
        labels: List[str],
        implementation: ClassifierImplementation[T, R],
        **config_kwargs: Any,
    ) -> C:
        """
        Factory method to create a classifier instance.

        This method provides a consistent way to create classifier instances
        with proper configuration and implementation.

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
           - Pass name, description, config, and implementation
           - Return the configured instance

        ## Error Handling

        This method handles these error cases:
        - Parameter validation (delegated to ClassifierConfig)
        - Type checking for critical parameters
        - Proper extraction of params dictionary

        ## Examples

        Basic usage:

        ```python
        from sifaka.classifiers.base import (
            Classifier,
            ClassifierImplementation,
            ClassificationResult
        )

        # Create an implementation
        class MyImplementation(ClassifierImplementation[str, str]):
            def __init__(self, config):
                self.config = config

            def classify_impl(self, text: str) -> ClassificationResult[str]:
                # Implementation details...
                return ClassificationResult(
                    label="example",
                    confidence=0.9,
                    metadata={"length": len(text)}
                )

            def warm_up_impl(self) -> None:
                # No special initialization needed
                pass

        # Create an instance using the factory method
        classifier = Classifier.create(
            name="my_classifier",
            description="My custom classifier implementation",
            labels=["label1", "label2", "label3"],
            implementation=MyImplementation(config),
            cache_size=100,
            min_confidence=0.6,
            params={"custom_param": "value"}
        )
        ```

        Args:
            name: Name of the classifier
            description: Description of what this classifier does
            labels: List of valid labels for classification
            implementation: Implementation of the classification logic
            **config_kwargs: Additional configuration parameters

        Returns:
            New classifier instance
        """
        # Extract params from config_kwargs if present
        params = config_kwargs.pop("params", {})

        # Create config with remaining kwargs
        config = ClassifierConfig[T](labels=labels, params=params, **config_kwargs)

        # Create instance
        return cls(name=name, description=description, config=config, implementation=implementation)


# Create type aliases for common classifier types
TextClassifier = Classifier[str, str]

# Export these types
__all__ = [
    "Classifier",
    "ClassifierConfig",
    "ClassificationResult",
    "ClassifierProtocol",
    "ClassifierImplementation",
    "TextProcessor",
    "TextClassifier",
]
