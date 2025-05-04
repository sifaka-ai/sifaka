# State Management in Sifaka

This document outlines the standardized approach to state management in the Sifaka codebase.

## Principles

1. **Consistency**: Use the same state management pattern across all components
2. **Encapsulation**: Keep state private and provide controlled access
3. **Immutability**: Prefer immutable state where possible
4. **Clarity**: Make it clear what is state vs. configuration

## Standardized Approach

### 1. Class Constants

Use `ClassVar` for true constants that don't change per instance:

```python
from typing import ClassVar, List

class MyComponent:
    DEFAULT_LABELS: ClassVar[List[str]] = ["label1", "label2"]
    DEFAULT_COST: ClassVar[float] = 1.0
```

### 2. Instance State

Use `StateManager` for all mutable instance state:

```python
from pydantic import BaseModel, PrivateAttr
from sifaka.utils.state import StateManager, create_classifier_state

class MyComponent(BaseModel):
    # Configuration (immutable)
    name: str
    description: str

    # State (mutable)
    _state_manager: StateManager = PrivateAttr(default_factory=create_classifier_state)
```

### 3. Initialization Pattern

Use a consistent initialization pattern with StateManager:

```python
def __init__(self, name: str, description: str, config: Optional[Config] = None):
    # Initialize configuration
    super().__init__(name=name, description=description, config=config or Config())

    # State is managed by StateManager, no need to initialize here

def warm_up(self) -> None:
    """Initialize resources if not already initialized."""
    state = self._state_manager.get_state()
    if not state.initialized:
        # Initialize resources
        state.model = self._load_model()
        state.initialized = True
```

### 4. State Access

Provide controlled access to state through StateManager:

```python
@property
def is_initialized(self) -> bool:
    """Check if the component is initialized."""
    return self._state_manager.get_state().initialized

def get_state_summary(self) -> Dict[str, Any]:
    """Get a summary of the component's state."""
    state = self._state_manager.get_state()
    return {
        "initialized": state.initialized,
        "model_loaded": state.model is not None,
        "cache_size": len(state.cache),
    }
```

## Examples

### Classifier Example

```python
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.utils.state import StateManager, create_classifier_state

class ExampleClassifier(BaseModel):
    """Example classifier implementation."""

    # Class-level constants (use ClassVar for true constants)
    DEFAULT_LABELS: ClassVar[List[str]] = ["label1", "label2", "unknown"]
    DEFAULT_COST: ClassVar[float] = 1.0

    # Configuration (immutable)
    name: str
    description: str
    config: ClassifierConfig

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_classifier_state)

    def warm_up(self) -> None:
        """Initialize the classifier if needed."""
        state = self._state_manager.get_state()
        if not state.initialized:
            state.model = self._load_model()
            state.initialized = True

    def _load_model(self) -> Any:
        """Load the model."""
        # Implementation details
        return {}
```

### Rule Example

```python
from typing import Any, ClassVar, Dict, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.utils.state import StateManager, create_rule_state

class ExampleRule(BaseModel):
    """Example rule implementation."""

    # Class-level constants (use ClassVar for true constants)
    DEFAULT_PRIORITY: ClassVar[str] = "MEDIUM"
    DEFAULT_COST: ClassVar[float] = 1.0

    # Configuration (immutable)
    name: str
    description: str
    config: RuleConfig

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        state = self._state_manager.get_state()
        if not state.initialized:
            state.validator = self._create_validator()
            state.initialized = True

    def _create_validator(self) -> Any:
        """Create the validator."""
        # Implementation details
        return {}
```

## Migration Guide

When migrating existing components to the standardized state management approach:

1. Identify all state variables in the component
2. Convert class variables to `ClassVar` if they are true constants
3. Replace direct state variables with `_state_manager` using the appropriate factory function
4. Keep configuration as regular Pydantic fields
5. Update the `warm_up()` method to use `_state_manager.get_state()`
6. Update all references to state variables to use `_state_manager.get_state()`
7. Use the appropriate state class (ClassifierState, RuleState, CriticState, etc.)

## State Manager Factory Functions

Sifaka provides factory functions for creating state managers for different component types:

```python
from sifaka.utils.state import (
    create_classifier_state,
    create_rule_state,
    create_critic_state,
    create_model_state,
    create_chain_state,
    create_adapter_state,
)

# Create a state manager for a classifier
classifier_state_manager = create_classifier_state()

# Create a state manager for a rule
rule_state_manager = create_rule_state()

# Create a state manager for a critic
critic_state_manager = create_critic_state()

# Create a state manager for a model provider
model_state_manager = create_model_state()

# Create a state manager for a chain
chain_state_manager = create_chain_state()

# Create a state manager for an adapter
adapter_state_manager = create_adapter_state()
```

## Additional Component Examples

### Model Provider Example

```python
from typing import ClassVar, Optional, Any
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.utils.state import StateManager, create_model_state
from sifaka.models.base import APIClient, TokenCounter

class ExampleModelProvider(BaseModel):
    """Example model provider implementation."""

    # Class-level constants
    DEFAULT_MODEL: ClassVar[str] = "example-model"

    # Configuration (immutable)
    model_name: str
    api_key: Optional[str] = None

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_model_state)

    def warm_up(self) -> None:
        """Initialize the model provider if needed."""
        state = self._state_manager.get_state()
        if not state.initialized:
            # Initialize API client
            state.client = self._create_client()

            # Initialize token counter
            state.token_counter = self._create_token_counter()

            # Mark as initialized
            state.initialized = True

    def _create_client(self) -> APIClient:
        """Create the API client."""
        # Implementation details
        return APIClient(api_key=self.api_key)

    def _create_token_counter(self) -> TokenCounter:
        """Create the token counter."""
        # Implementation details
        return TokenCounter(model=self.model_name)

    def generate(self, prompt: str) -> str:
        """Generate text using the model."""
        # Ensure initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Use client to generate text
        response = state.client.generate(prompt)

        # Return the generated text
        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens in the text."""
        # Ensure initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Use token counter to count tokens
        return state.token_counter.count_tokens(text)
```

### Adapter Example

```python
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.utils.state import StateManager, create_adapter_state

# Type variables
T = TypeVar("T")  # Input type
A = TypeVar("A")  # Adaptee type

class ExampleAdapter(BaseModel, Generic[T, A]):
    """Example adapter implementation."""

    # Configuration (immutable)
    name: str
    description: str

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_adapter_state)

    def __init__(self, name: str, description: str, adaptee: A):
        """Initialize the adapter with a component to adapt."""
        # Initialize configuration
        super().__init__(name=name, description=description)

        # Initialize state
        state = self._state_manager.get_state()
        state.adaptee = adaptee
        state.initialized = True

    @property
    def adaptee(self) -> A:
        """Get the component being adapted."""
        return self._state_manager.get_state().adaptee

    def validate(self, input_value: T) -> bool:
        """Validate input using the adaptee."""
        # Get state
        state = self._state_manager.get_state()

        # Check cache if available
        if input_value in state.cache:
            return state.cache[input_value]

        # Use adaptee to validate
        result = self._validate_with_adaptee(input_value)

        # Cache result
        state.cache[input_value] = result

        return result

    def _validate_with_adaptee(self, input_value: T) -> bool:
        """Validate input using the adaptee."""
        # Implementation details
        adaptee = self._state_manager.get_state().adaptee
        return adaptee.is_valid(input_value)
```

### Guardrails Adapter Example

```python
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.utils.state import StateManager, create_adapter_state
from sifaka.rules.base import RuleResult

class GuardrailsAdapter(BaseModel):
    """Example Guardrails adapter implementation."""

    # Configuration (immutable)
    name: str = "guardrails_adapter"
    description: str = "Adapter for Guardrails validators"

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_adapter_state)

    def __init__(self, rail_spec: str):
        """Initialize with a Guardrails rail specification."""
        # Initialize configuration
        super().__init__()

        # Initialize state
        state = self._state_manager.get_state()

        # Create Guardrails validator
        state.adaptee = self._create_guardrails_validator(rail_spec)

        # Initialize cache
        state.cache = {}

        # Mark as initialized
        state.initialized = True

    def _create_guardrails_validator(self, rail_spec: str) -> Any:
        """Create a Guardrails validator from the rail specification."""
        # Implementation details
        return {"rail_spec": rail_spec}

    def validate(self, text: str) -> RuleResult:
        """Validate text using the Guardrails validator."""
        # Get state
        state = self._state_manager.get_state()

        # Check cache
        if text in state.cache:
            return state.cache[text]

        # Use Guardrails validator to validate
        validator = state.adaptee

        # Simulate validation
        passed = len(text) > 10  # Simplified example

        # Create result
        result = RuleResult(
            passed=passed,
            message="Text validation passed" if passed else "Text too short",
            metadata={"validator": "guardrails"}
        )

        # Cache result
        state.cache[text] = result

        return result
```

## Practical Example: Building a Custom Classifier

Here's a complete example of building a custom classifier with standardized state management:

```python
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr
from sifaka.classifiers.base import BaseClassifier, ClassificationResult, ClassifierConfig
from sifaka.utils.state import StateManager, create_classifier_state

class CustomSentimentClassifier(BaseClassifier[str, str]):
    """
    Custom sentiment classifier implementation with standardized state management.

    This classifier uses a simple keyword-based approach to classify text sentiment.
    It demonstrates the standardized state management pattern in Sifaka.
    """

    # Class-level constants
    POSITIVE_WORDS: ClassVar[List[str]] = ["good", "great", "excellent", "happy", "positive"]
    NEGATIVE_WORDS: ClassVar[List[str]] = ["bad", "terrible", "awful", "sad", "negative"]

    # Configuration (immutable)
    name: str
    description: str
    config: ClassifierConfig[str]

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager: StateManager = PrivateAttr(default_factory=create_classifier_state)

    def warm_up(self) -> None:
        """Initialize the classifier if needed."""
        state = self._state_manager.get_state()
        if not state.initialized:
            # Prepare word lists (could be loading from a file or model in a real implementation)
            state.feature_names = {
                "positive_words": self.POSITIVE_WORDS,
                "negative_words": self.NEGATIVE_WORDS
            }

            # Initialize cache
            state.cache = {}

            # Mark as initialized
            state.initialized = True
            print(f"Initialized classifier: {self.name}")

    def _classify_impl_uncached(self, text: str) -> ClassificationResult[str]:
        """
        Implement the core classification logic.

        This method is called by the BaseClassifier when the result is not in the cache.
        It implements the actual classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with the classification label and confidence
        """
        # Ensure initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Handle empty text
        if not text:
            return ClassificationResult(
                label="unknown",
                confidence=1.0,
                metadata={"reason": "empty_text"}
            )

        # Count positive and negative words
        positive_count = sum(1 for word in state.feature_names["positive_words"] if word.lower() in text.lower())
        negative_count = sum(1 for word in state.feature_names["negative_words"] if word.lower() in text.lower())
        total_count = positive_count + negative_count

        # Determine sentiment
        if total_count == 0:
            return ClassificationResult(
                label="neutral",
                confidence=0.7,
                metadata={"positive_count": 0, "negative_count": 0}
            )

        # Calculate confidence based on word counts
        if positive_count > negative_count:
            confidence = positive_count / (positive_count + negative_count)
            return ClassificationResult(
                label="positive",
                confidence=confidence,
                metadata={"positive_count": positive_count, "negative_count": negative_count}
            )
        else:
            confidence = negative_count / (positive_count + negative_count)
            return ClassificationResult(
                label="negative",
                confidence=confidence,
                metadata={"positive_count": positive_count, "negative_count": negative_count}
            )

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        cache_size: int = 100,
        min_confidence: float = 0.7,
        **kwargs: Any
    ) -> "CustomSentimentClassifier":
        """
        Create a custom sentiment classifier.

        This factory method creates a new instance with the specified configuration.

        Args:
            name: Name of the classifier
            description: Description of the classifier
            labels: Optional list of labels (defaults to ["positive", "negative", "neutral", "unknown"])
            cache_size: Size of the classification cache
            min_confidence: Minimum confidence threshold
            **kwargs: Additional configuration parameters

        Returns:
            A new CustomSentimentClassifier instance
        """
        # Set default labels if not provided
        if labels is None:
            labels = ["positive", "negative", "neutral", "unknown"]

        # Create configuration
        config = ClassifierConfig(
            labels=labels,
            cache_size=cache_size,
            min_confidence=min_confidence,
            params=kwargs
        )

        # Create and return classifier
        return cls(name=name, description=description, config=config)


# Usage example
def main():
    # Create the classifier
    classifier = CustomSentimentClassifier.create(
        name="simple_sentiment",
        description="A simple sentiment classifier",
        cache_size=100,
        min_confidence=0.6
    )

    # Classify some text
    texts = [
        "I had a great day today!",
        "This is terrible, I'm very upset.",
        "Just a normal day, nothing special.",
        ""  # Empty text
    ]

    for text in texts:
        result = classifier.classify(text)
        print(f"Text: '{text}'")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Metadata: {result.metadata}")
        print()


if __name__ == "__main__":
    main()
```

## Best Practices

1. **Initialization**: Use lazy initialization with `warm_up()` for expensive resources
2. **State Access**: Access state through `_state_manager.get_state()`
3. **State Modification**: Modify state through `_state_manager.get_state()`
4. **Error Handling**: Handle initialization errors gracefully
5. **Documentation**: Document the state management approach in docstrings
6. **Testing**: Test state initialization, access, and modification
7. **Consistency**: Use the same state management pattern across all components
8. **Caching**: Use the state's cache dictionary for caching results
9. **Immutability**: Keep configuration immutable and state mutable
10. **Factory Methods**: Use factory methods for creating components with proper state initialization
