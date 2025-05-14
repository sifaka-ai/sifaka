"""
Classifier interfaces for Sifaka.

This module defines the interfaces for classifiers in the Sifaka framework.
These interfaces establish a common contract for classifier behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ClassifierProtocol**: Base interface for all classifiers
   - **TextProcessor**: Interface for text processing components

## Usage Examples

```python
from sifaka.interfaces.classifier import ClassifierProtocol, TextProcessor

# Create a classifier implementation
class MyClassifier(ClassifierProtocol[str, str]):
    @property
    def name(self) -> str:
        return "my_classifier"

    @property
    def description(self) -> str:
        return "A simple classifier implementation"

    @property
    def min_confidence(self) -> float:
        return 0.5

    @property
    def _state_manager(self) -> Any:
        return self._state_manager_impl

    def initialize(self) -> None:
        pass

    def warm_up(self) -> None:
        pass

    def get_statistics(self) -> Dict[str, Any]:
        return {"execution_count": 0}

    def clear_cache(self) -> None:
        pass

    def classify(self, text: str) -> ClassificationResult:
        return ClassificationResult(label="example", confidence=0.8)

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        return [self.classify(text) if self else "" for text in texts]
```

## Error Handling

- ValueError: Raised for invalid inputs
- RuntimeError: Raised for execution failures
- TypeError: Raised for type mismatches
"""

from typing import Any, Dict, List, Protocol, TypeVar, TYPE_CHECKING

# Type variables for generic protocols
T = TypeVar("T", contravariant=True)  # Input type
R = TypeVar("R", covariant=True)  # Result label type
InputType = TypeVar("InputType")  # Input type for ClassifierProtocol
OutputType = TypeVar("OutputType")  # Output type for ClassifierProtocol

# Forward reference for ClassificationResult to avoid circular imports
if TYPE_CHECKING:
    from sifaka.core.results import ClassificationResult


class TextProcessor(Protocol):
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
    """

    def process(self, text: Any) -> Dict[str, Any]:
        """Process the input text and return a dictionary of results."""
        ...


class ClassifierProtocol(Protocol):
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
    """

    def classify(self, text: Any) -> "ClassificationResult":
        """Classify a single text input."""
        ...

    def batch_classify(self, texts: List[Any]) -> List["ClassificationResult"]:
        """Classify multiple text inputs."""
        ...

    @property
    def name(self) -> str:
        """Get the classifier name."""
        ...

    @property
    def description(self) -> str:
        """Get the classifier description."""
        ...

    @property
    def min_confidence(self) -> float:
        """Get the minimum confidence threshold."""
        ...

    @property
    def _state_manager(self) -> Any:
        """Get the state manager."""
        ...

    def initialize(self) -> None:
        """Initialize the classifier."""
        ...

    def warm_up(self) -> None:
        """Warm up the classifier."""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        ...

    def clear_cache(self) -> None:
        """Clear the classifier cache."""
        ...
