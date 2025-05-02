"""
Base classes and protocols for adapter-based rules.

This module provides the foundation for adapting various components to function as validation rules,
such as classifiers, models, or external services.

## Adapter Pattern

The adapter pattern allows existing components to be used with Sifaka's rule system:

1. **Adaptable Protocol**: Defines the minimum interface needed for adaptation
2. **BaseAdapter**: Abstract base class for adapting components to validators
3. **Concrete Adapters**: Specific implementations for different component types

## Component Lifecycle

### Adapters
1. **Initialization**: Create with component to adapt
2. **Validation**: Process input text through adapted component
3. **Result Conversion**: Convert component-specific outputs to RuleResults

## Implementation Pattern

To create a new adapter:

```python
from sifaka.adapters.rules.base import BaseAdapter, Adaptable
from sifaka.rules.base import RuleResult

class MyComponent:
    """A component that can classify text."""

    @property
    def name(self) -> str:
        return "my_component"

    @property
    def description(self) -> str:
        return "A component that can classify text"

    def classify(self, text: str) -> bool:
        # Custom implementation
        return True

class MyAdapter(BaseAdapter[str]):
    """Adapter for MyComponent."""

    def validate(self, input_text: str, **kwargs) -> RuleResult:
        # Handle empty text first
        empty_result = self.handle_empty_text(input_text)
        if empty_result:
            return empty_result

        # Use the adaptee to classify the text
        result = self.adaptee.classify(input_text)

        # Convert the result to a RuleResult
        return RuleResult(
            passed=result,
            message="Text validation successful" if result else "Text validation failed"
        )
```
"""

from typing import Any, Dict, Generic, Protocol, Type, TypeVar, cast, runtime_checkable

from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult, ValidationError


T = TypeVar("T")  # Input type
A = TypeVar("A", bound="Adaptable")  # Adaptee type


@runtime_checkable
class Adaptable(Protocol):
    """
    Protocol for components that can be adapted to rules.

    Any component that can be adapted to a Sifaka rule must implement
    this protocol, which requires a name and description.

    Lifecycle:
    1. Implementation: Component implements the required properties
    2. Adaptation: Component is adapted using a compatible adapter
    3. Usage: Adapted component is used as a Sifaka rule validator

    Examples:
        ```python
        class MyClassifier:
            @property
            def name(self) -> str:
                return "sentiment_classifier"

            @property
            def description(self) -> str:
                return "Classifies text sentiment"
        ```
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            A string name for the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            A string description of the component's purpose
        """
        ...


class BaseAdapter(BaseValidator[T], Generic[T, A]):
    """
    Base class for adapters that convert components to validators.

    This abstract class provides the foundation for adapting external
    components to work with Sifaka's rule system. It handles common
    tasks like validating the adaptee and providing an interface for
    validation.

    Type Parameters:
        T: The input type to validate
        A: The adaptee type, must implement Adaptable

    Lifecycle:
    1. Initialization: Receive component to adapt
    2. Validation: Process input through adaptee
    3. Result Conversion: Translate adaptee outputs to RuleResults

    Examples:
        ```python
        class SentimentAdapter(BaseAdapter[str, SentimentClassifier]):
            def validate(self, text: str, **kwargs) -> RuleResult:
                # Handle empty text
                empty_result = self.handle_empty_text(text)
                if empty_result:
                    return empty_result

                # Use the adaptee to classify the text
                classification = self.adaptee.classify(text)

                # Convert the classification to a RuleResult
                return RuleResult(
                    passed=classification.label in ["positive", "neutral"],
                    message=f"Sentiment: {classification.label}",
                    metadata={"confidence": classification.confidence}
                )
        ```
    """

    @property
    def validation_type(self) -> type:
        """
        Get the type of input this validator can validate.

        Returns:
            The type this validator can validate (default: str)
        """
        return str

    def __init__(self, adaptee: A) -> None:
        """
        Initialize with adaptee.

        Args:
            adaptee: The component being adapted

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        self._validate_adaptee(adaptee)
        self._adaptee = adaptee

    def _validate_adaptee(self, adaptee: Any) -> None:
        """
        Validate that adaptee implements the required protocol.

        This internal method ensures the adaptee meets the requirements
        for adaptation.

        Args:
            adaptee: The component to validate

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        if not isinstance(adaptee, Adaptable):
            raise ConfigurationError(
                f"Adaptee must implement Adaptable protocol, got {type(adaptee)}"
            )

    @property
    def adaptee(self) -> A:
        """
        Get the adaptee.

        Returns:
            The component being adapted
        """
        return self._adaptee

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate using the adaptee.

        This method should be implemented by subclasses to perform
        the actual validation using the adaptee.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValidationError: If validation fails
            NotImplementedError: If not implemented by subclass
        """
        try:
            # This is an abstract method that should be implemented by subclasses
            raise NotImplementedError("Subclasses must implement the validate method")
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e


def create_adapter(
    adapter_type: Type[BaseAdapter[T, A]],
    adaptee: A,
    **kwargs: Any
) -> BaseAdapter[T, A]:
    """
    Factory function to create an adapter with standardized configuration.

    This function simplifies the creation of adapters by providing a
    consistent interface.

    Args:
        adapter_type: The class of the adapter to create
        adaptee: The component to adapt
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        A configured adapter instance

    Examples:
        ```python
        # Create a classifier adapter
        from sifaka.adapters.rules import ClassifierAdapter

        classifier = SentimentClassifier()
        adapter = create_adapter(
            adapter_type=ClassifierAdapter,
            adaptee=classifier,
            threshold=0.8
        )
        ```
    """
    return adapter_type(adaptee=adaptee, **kwargs)
