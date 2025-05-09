# Pydantic v2 Usage Guide for Sifaka

This document outlines the standard patterns for using Pydantic v2 in the Sifaka codebase. Following these patterns ensures consistency across the codebase and takes advantage of the performance improvements and new features in Pydantic v2.

## Model Definition

### Use `ConfigDict` for model configuration

```python
from pydantic import BaseModel, ConfigDict, Field

class MyModel(BaseModel):
    # Use ConfigDict instead of class Config
    model_config = ConfigDict(
        frozen=True,  # Makes the model immutable
        extra="forbid",  # Prevents extra fields
        arbitrary_types_allowed=False,  # Default, but can be set to True if needed
    )
    
    name: str = Field(description="The name of the model")
    value: int = Field(ge=0, description="A non-negative value")
```

### Field Validation

Use `Field` for field validation and metadata:

```python
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    # Field with validation
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    
    # Required field with description
    name: str = Field(..., description="The name of the model")
    
    # Field with default factory
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
```

## Serialization and Deserialization

### Use `model_dump()` instead of `dict()`

```python
# Convert model to dict
model_dict = my_model.model_dump()

# Convert model to dict, excluding some fields
model_dict = my_model.model_dump(exclude={"metadata"})

# Convert model to dict, including only some fields
model_dict = my_model.model_dump(include={"name", "value"})
```

### Use `model_validate()` instead of `parse_obj()`

```python
# Create model from dict
data = {"name": "example", "value": 42}
model = MyModel.model_validate(data)

# Create model from another model
other_model = OtherModel(name="example", value=42)
model = MyModel.model_validate(other_model.model_dump())
```

### Use `model_validate_json()` instead of `parse_raw()`

```python
# Create model from JSON string
json_str = '{"name": "example", "value": 42}'
model = MyModel.model_validate_json(json_str)
```

## Immutable Models

For configuration objects and other models that should be immutable, use `frozen=True` in the `ConfigDict`:

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    
    # Method to create a new instance with updated values
    def with_options(self, **kwargs: Any) -> "ModelConfig":
        """Create a new config with updated options."""
        return ModelConfig(**{**self.model_dump(), **kwargs})
```

## Generic Models

For generic models, use the `Generic` base class:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Result(BaseModel, Generic[T]):
    value: T
    success: bool = True
```

## Error Handling

When working with Pydantic models, handle validation errors appropriately:

```python
from pydantic import ValidationError

try:
    model = MyModel.model_validate(data)
except ValidationError as e:
    # Handle validation error
    print(f"Validation error: {e}")
```

## Best Practices

1. **Use descriptive field names**: Choose clear, descriptive field names that indicate the purpose of the field.

2. **Add field descriptions**: Use the `description` parameter in `Field` to document the purpose and constraints of each field.

3. **Validate field values**: Use validation parameters like `ge`, `le`, `min_length`, `max_length`, etc., to enforce constraints on field values.

4. **Use immutable models for configuration**: Make configuration models immutable with `frozen=True` to prevent accidental modification.

5. **Provide factory methods**: For immutable models, provide factory methods like `with_options()` to create new instances with updated values.

6. **Use type annotations**: Always use type annotations for fields to enable static type checking and better IDE support.

7. **Handle validation errors**: Always handle validation errors appropriately to provide helpful error messages to users.

8. **Use model_dump() and model_validate()**: Always use the new Pydantic v2 methods for serialization and deserialization.

## Migration from Pydantic v1

If you encounter code that uses Pydantic v1 patterns, here's how to migrate it to v2:

| Pydantic v1 | Pydantic v2 |
|-------------|-------------|
| `class Config:` | `model_config = ConfigDict(...)` |
| `dict()` | `model_dump()` |
| `parse_obj()` | `model_validate()` |
| `parse_raw()` | `model_validate_json()` |
| `schema()` | `model_json_schema()` |
| `validate()` | `model_validate()` |
| `construct()` | `model_construct()` |

## Examples from Sifaka

### Model Configuration

```python
class ModelConfig(BaseModel):
    """Immutable configuration for model providers."""
    
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific configuration parameters",
    )
    
    def with_options(self, **kwargs: Any) -> "ModelConfig":
        """Create a new config with updated options."""
        return ModelConfig(**{**self.model_dump(), **kwargs})
```

### Generic Result Model

```python
class ModelResult(BaseModel, Generic[T]):
    """Result of a model operation."""
    
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    output: T = Field(description="The generated output")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the operation",
    )
```

By following these patterns consistently across the Sifaka codebase, we ensure that all components use Pydantic v2 features effectively and consistently.
