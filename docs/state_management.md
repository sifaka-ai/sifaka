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

Use `PrivateAttr` for all mutable instance state:

```python
from pydantic import BaseModel, PrivateAttr

class MyComponent(BaseModel):
    # Configuration (immutable)
    name: str
    description: str
    
    # State (mutable)
    _initialized: bool = PrivateAttr(default=False)
    _model: Optional[Any] = PrivateAttr(default=None)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
```

### 3. Initialization Pattern

Use a consistent initialization pattern:

```python
def __init__(self, name: str, description: str, config: Optional[Config] = None):
    # Initialize configuration
    super().__init__(name=name, description=description, config=config or Config())
    
    # Initialize state
    self._initialized = False
    self._model = None
    
def warm_up(self) -> None:
    """Initialize resources if not already initialized."""
    if not self._initialized:
        # Initialize resources
        self._model = self._load_model()
        self._initialized = True
```

### 4. State Access

Provide controlled access to state:

```python
@property
def is_initialized(self) -> bool:
    """Check if the component is initialized."""
    return self._initialized

def get_state_summary(self) -> Dict[str, Any]:
    """Get a summary of the component's state."""
    return {
        "initialized": self._initialized,
        "model_loaded": self._model is not None,
        "cache_size": len(self._cache),
    }
```

## Examples

### Classifier Example

```python
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr

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

    # Private attributes for state management (use PrivateAttr)
    _initialized: bool = PrivateAttr(default=False)
    _model: Optional[Any] = PrivateAttr(default=None)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def warm_up(self) -> None:
        """Initialize the classifier if needed."""
        if not self._initialized:
            self._model = self._load_model()
            self._initialized = True

    def _load_model(self) -> Any:
        """Load the model."""
        # Implementation details
        return {}
```

### Rule Example

```python
from typing import Any, ClassVar, Dict, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr

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

    # Private attributes for state management (use PrivateAttr)
    _initialized: bool = PrivateAttr(default=False)
    _validator: Optional[Any] = PrivateAttr(default=None)
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._initialized:
            self._validator = self._create_validator()
            self._initialized = True

    def _create_validator(self) -> Any:
        """Create the validator."""
        # Implementation details
        return {}
```

## Migration Guide

When migrating existing components to the standardized state management approach:

1. Identify all state variables in the component
2. Convert class variables to `ClassVar` if they are true constants
3. Convert instance variables to `PrivateAttr` if they represent mutable state
4. Keep configuration as regular Pydantic fields
5. Add an `_initialized` flag if the component needs lazy initialization
6. Implement a `warm_up()` method for lazy initialization
7. Update all references to state variables

## Best Practices

1. **Initialization**: Use lazy initialization with `warm_up()` for expensive resources
2. **State Access**: Provide properties or methods for controlled access to state
3. **State Modification**: Provide methods for controlled modification of state
4. **Error Handling**: Handle initialization errors gracefully
5. **Documentation**: Document the state management approach in docstrings
6. **Testing**: Test state initialization, access, and modification
