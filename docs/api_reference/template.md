# [Component Type] API Reference

This document provides detailed API reference for all [Component Type] in Sifaka.

## Overview

[Brief description of the component type and its role in the Sifaka ecosystem]

[Component Type] implement these main interfaces:
- **Interface1**: [Brief description]
- **Interface2**: [Brief description]
- **Interface3**: [Brief description]

## Core Interfaces

### [Interface1]

`[Interface1]` is the abstract base class for all [component type] in Sifaka.

```python
from sifaka.[component_type].base import [Interface1], [Config], [Result]
from typing import List, Dict, Any

class My[ComponentType]([Interface1][InputType, OutputType]):
    """Custom [component type] implementation."""
    
    def method1(self, input: InputType) -> bool:
        """[Brief description]."""
        return True
    
    def method2(self, input: InputType, params: List[Dict[str, Any]]) -> OutputType:
        """[Brief description]."""
        return input
    
    def method3(self, input: InputType) -> [Result][OutputType]:
        """[Brief description]."""
        return [Result](
            score=0.9,
            metadata={}
        )
```

### [Interface2]

`[Interface2]` is a protocol for [specific functionality] components.

```python
from sifaka.[component_type].base import [Interface2]

class My[ComponentType]([Interface2][InputType]):
    """Custom [component type] implementation."""
    
    def method1(self, input: InputType) -> bool:
        """[Brief description]."""
        return True
```

### [Interface3]

`[Interface3]` is a protocol for [specific functionality] components.

```python
from sifaka.[component_type].base import [Interface3], [Result]

class My[ComponentType]([Interface3][InputType, OutputType]):
    """Custom [component type] implementation."""
    
    def method3(self, input: InputType) -> [Result][OutputType]:
        """[Brief description]."""
        return [Result](
            score=0.9,
            metadata={}
        )
```

## Configuration

### [ComponentType]Config

`[ComponentType]Config` is the configuration class for [component type].

```python
from sifaka.[component_type].base import [ComponentType]Config

# Create a [component type] configuration
config = [ComponentType]Config(
    name="my_[component_type]",
    description="A custom [component type]",
    param1=0.7,
    param2=3,
    params={
        "custom_param": "value",
    }
)

# Access configuration values
print(f"Name: {config.name}")
print(f"Param1: {config.param1}")
print(f"Custom param: {config.params['custom_param']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    param1=0.8,
    params={"custom_param": "new_value"}
)
```

## Results

### [ComponentType]Result

`[ComponentType]Result` represents the result of a [component type] operation.

```python
from sifaka.[component_type].base import [ComponentType]Result

# Create [component type] result
result = [ComponentType]Result(
    value="result_value",
    score=0.7,
    metadata={
        "key1": "value1",
        "key2": "value2"
    }
)

# Access result values
print(f"Value: {result.value}")
print(f"Score: {result.score}")
print(f"Metadata: {result.metadata}")
```

## Factory Functions

Sifaka provides factory functions for creating [component type]. Always use these factory functions instead of instantiating [component type] classes directly.

### create_[specific_component_type]

```python
def create_[specific_component_type](
    param1: Type,
    param2: Type = default_value,
    name: str = "default_name",
    description: str = "Default description",
    **kwargs
) -> [Interface1][InputType, OutputType]:
    """
    Create a [specific component type].
    
    Args:
        param1: Description of param1
        param2: Description of param2
        name: Name of the [component type]
        description: Description of the [component type]
        **kwargs: Additional keyword arguments
        
    Returns:
        A [specific component type] instance
        
    Raises:
        ValueError: If parameters are invalid
    """
```

## [Component Type] Types

Sifaka provides several types of [component type]:

### [SpecificType1]

`[SpecificType1]` [brief description of functionality].

```python
from sifaka.[component_type].[specific_type1] import create_[specific_type1]

# Create a [specific type1]
component = create_[specific_type1](
    param1="value1",
    param2="value2",
    name="[specific_type1]",
    description="[Brief description]"
)
```

### [SpecificType2]

`[SpecificType2]` [brief description of functionality].

```python
from sifaka.[component_type].[specific_type2] import create_[specific_type2]
from sifaka.models.openai import create_openai_chat_provider

# Create dependencies
model = create_openai_chat_provider(model_name="gpt-4")

# Create a [specific type2]
component = create_[specific_type2](
    dependency=model,
    param1="value1",
    name="[specific_type2]",
    description="[Brief description]"
)
```

## Usage Examples

### Basic [Component Type] Usage

```python
from sifaka.[component_type].[specific_type1] import create_[specific_type1]

# Create a [component type]
component = create_[specific_type1](
    param1="value1",
    param2="value2"
)

# Use the [component type]
result = component.method1("input")
print(f"Result: {result}")

# Get detailed results
detailed_result = component.method3("input")
print(f"Score: {detailed_result.score}")
print(f"Metadata: {detailed_result.metadata}")
```

### Custom [Component Type] Implementation

```python
from sifaka.[component_type].base import [Interface1], [ComponentType]Config, [Result]
from typing import List, Dict, Any

class My[ComponentType]([Interface1][InputType, OutputType]):
    """Custom [component type] implementation."""
    
    def __init__(self, config: [ComponentType]Config):
        super().__init__(config)
        self.param1 = config.params.get("param1", "default")
        self.param2 = config.params.get("param2", 10)
    
    def method1(self, input: InputType) -> bool:
        """[Brief description]."""
        # Implementation
        return True
    
    def method2(self, input: InputType, params: List[Dict[str, Any]]) -> OutputType:
        """[Brief description]."""
        # Implementation
        return input
    
    def method3(self, input: InputType) -> [Result][OutputType]:
        """[Brief description]."""
        # Implementation
        return [Result](
            score=0.9,
            metadata={}
        )

# Create the [component type]
component = My[ComponentType](
    [ComponentType]Config(
        name="my_[component_type]",
        description="Custom [component type]",
        params={"param1": "custom", "param2": 20}
    )
)

# Use the [component type]
result = component.method1("input")
```

### Using [Component Type] with [Related Component]

[Component Type] are typically used with [Related Component] to [brief description of interaction]:

```python
from sifaka.[component_type].[specific_type1] import create_[specific_type1]
from sifaka.[related_component].[specific_related] import create_[specific_related]

# Create components
component1 = create_[specific_type1](param1="value1")
component2 = create_[specific_related](param1="value1")

# Use components together
result = component2.use_with(component1, "input")
print(f"Result: {result}")
```

## Implementation Details

[Component Type] in Sifaka follow a standardized implementation pattern:

1. **State Management**: [Component Type] use the `_state_manager` pattern for managing state
2. **Configuration**: [Component Type] use `[ComponentType]Config` for configuration
3. **Factory Functions**: [Component Type] provide factory functions for easy instantiation
4. **Interfaces**: [Component Type] implement the required interfaces

### State Management

[Component Type] use the `_state_manager` pattern for managing state:

```python
from pydantic import PrivateAttr
from sifaka.[component_type].base import [Interface1], [ComponentType]Config, create_[component_type]_state

class My[ComponentType]([Interface1][InputType, OutputType]):
    """Custom [component type] implementation."""
    
    _state_manager = PrivateAttr(default_factory=create_[component_type]_state)
    
    def __init__(self, config: [ComponentType]Config):
        super().__init__(config)
        # Initialize any [component type]-specific attributes
    
    def warm_up(self):
        """Initialize expensive resources."""
        state = self._state_manager.get_state()
        if not state.initialized:
            # Initialize state
            state.initialized = True
    
    def method1(self, input: InputType) -> bool:
        """[Brief description]."""
        state = self._state_manager.get_state()
        # Use state for implementation
        return True
```

## Best Practices

1. **Use factory functions** for creating [component type]
2. **Use standardized state management** with `_state_manager`
3. **Implement all required interfaces**
4. **Handle empty input gracefully** in all methods
5. **Include detailed metadata** in results
6. [Component-specific best practice]
7. [Component-specific best practice]
8. [Component-specific best practice]
9. [Component-specific best practice]
10. **Implement warm_up()** for lazy initialization of expensive resources

## Error Handling

[Component Type] implement several error handling patterns:

### Handling Empty Input

```python
def method1(self, input: InputType) -> bool:
    """[Brief description]."""
    if not input:
        return True  # Empty input is valid by default
    
    # Normal implementation
    return len(input) > 10
```

### Handling [Specific Error Type]

```python
def method2(self, input: InputType, params: List[Dict[str, Any]]) -> OutputType:
    """[Brief description]."""
    try:
        # Try implementation
        return self._implementation(input, params)
    except [SpecificError]:
        # Fall back
        return self._fallback(input, params)
```

## See Also

- [[Component Type] Component Documentation](../../components/[component_type].md)
- [Implementation Notes for [Component Type]](../../implementation_notes/[specific_component_type].md)
- [[Related Component] API Reference](../[related_component]/README.md)
