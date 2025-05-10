"""
Classifier protocol interfaces.

This module defines the protocol interfaces for classifiers in the Sifaka framework.
These protocols establish the contract that all classifier implementations must follow.
"""

from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable

from ..models import ClassificationResult

# Type variables for generic protocols
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result label type


@runtime_checkable
class TextProcessor(Protocol[T, R]):
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

    3. **Usage**: Use as a type hint or for runtime checks
       - Use as a type hint for function parameters
       - Check compliance with isinstance() at runtime
       - Pass to functions expecting a TextProcessor

    ## Error Handling

    Implementations should handle these error cases:
    - Invalid input types
    - Empty text inputs
    - Processing failures
    - Resource availability issues

    ## Examples

    Checking if an object follows the text processor protocol:

    ```python
    from sifaka.classifiers.interfaces import TextProcessor

    def process_text(processor: TextProcessor, text: str) -> Dict[str, Any]:
        return processor.process(text)

    # Check if an object implements the protocol
    if isinstance(my_object, TextProcessor):
        result = process_text(my_object, "Hello world")
    ```

    Creating a minimal text processor that implements the protocol:

    ```python
    from sifaka.classifiers.interfaces import TextProcessor
    from typing import Dict, Any

    class SimpleProcessor:
        def process(self, text: str) -> Dict[str, Any]:
            return {
                "length": len(text),
                "words": len(text.split()),
                "uppercase": text.upper()
            }

    # Verify protocol compliance
    processor = SimpleProcessor()
    assert isinstance(processor, TextProcessor)

    # Use the processor
    result = processor.process("This is a test")
    print(f"Text length: {result['length']}")
    ```
    """

    def process(self, text: T) -> Dict[str, Any]: ...


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
       - Implement state management methods
       - Implement initialization and warm-up methods
       - Implement error handling and execution tracking

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use as a type hint or for runtime checks
       - Use as a type hint for function parameters
       - Check compliance with isinstance() at runtime
       - Pass to functions expecting a ClassifierProtocol

    ## State Management

    Implementations should manage state using these patterns:
    - Use _state_manager for all mutable state
    - Initialize state during construction
    - Provide methods to access and modify state
    - Track execution statistics in state

    ## Error Handling

    Implementations should handle these error cases:
    - Invalid input types (non-string inputs)
    - Empty text inputs
    - Classification failures
    - Resource availability issues
    - Initialization failures
    - State management errors

    ## Execution Tracking

    Implementations should track execution using these patterns:
    - Track execution count
    - Track execution time
    - Track success/failure counts
    - Track cache hits/misses
    - Provide statistics through get_statistics()

    ## Examples

    Checking if an object follows the classifier protocol:

    ```python
    from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier
    from sifaka.classifiers.interfaces import ClassifierProtocol

    classifier = ToxicityClassifier()
    assert isinstance(classifier, ClassifierProtocol)
    ```

    Creating a minimal classifier that implements the protocol:

    ```python
    from sifaka.classifiers.interfaces import ClassifierProtocol
    from sifaka.classifiers.models import ClassificationResult
    from typing import List, Dict, Any

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

        @property
        def _state_manager(self) -> Any:
            # In a real implementation, this would return a state manager
            return self._state_manager_impl

        def initialize(self) -> None:
            # Initialize resources
            pass

        def warm_up(self) -> None:
            # Warm up resources
            self.initialize()

        def get_statistics(self) -> Dict[str, Any]:
            # Return execution statistics
            return {
                "execution_count": 0,
                "success_count": 0,
                "error_count": 0
            }

        def clear_cache(self) -> None:
            # Clear cached data
            pass

        def classify(self, text: str) -> ClassificationResult:
            # Simple implementation
            if "good" in text.lower():
                return ClassificationResult(
                    label="positive",
                    confidence=0.8,
                    metadata={"contains_good": True}
                )
            return ClassificationResult(
                label="neutral",
                confidence=0.6,
                metadata={}
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

    def classify(self, text: T) -> ClassificationResult[R]: ...
    def batch_classify(self, texts: List[T]) -> List[ClassificationResult[R]]: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def min_confidence(self) -> float: ...
    @property
    def _state_manager(self) -> Any: ...
    def initialize(self) -> None: ...
    def warm_up(self) -> None: ...
    def get_statistics(self) -> Dict[str, Any]: ...
    def clear_cache(self) -> None: ...
