# Sifaka Docstring Style Guide

This document defines the standard docstring format for the Sifaka codebase. Following these guidelines ensures consistency across the project and enables automatic documentation generation.

## General Principles

1. **All public APIs must have docstrings**
2. **Use Google-style docstrings**
3. **Include examples in all component docstrings**
4. **Be concise but complete**
5. **Use type annotations consistently**

## Docstring Format

Sifaka uses Google-style docstrings with the following sections:

```python
"""
Short summary of the component's purpose.

Detailed description that explains what this component does,
its key features, and any important concepts. This can be
multiple paragraphs if needed.

Lifecycle:
    1. Initialization: How the component is initialized
    2. Usage: How the component is typically used
    3. Cleanup: Any cleanup steps if applicable

Examples:
    ```python
    # Simple example
    from sifaka.module import Component
    
    component = Component(param1=value1)
    result = component.method("input")
    print(result)
    
    # More complex example
    component = Component(
        param1=value1,
        param2=value2,
        config={"option": "value"}
    )
    ```

Args:
    param1: Description of first parameter
    param2: Description of second parameter
    **kwargs: Description of keyword arguments

Returns:
    Description of return value

Raises:
    ExceptionType: When and why this exception is raised
    
Notes:
    Any additional information that doesn't fit elsewhere.
    
See Also:
    Related components or functions
"""
```

## Component-Specific Templates

### Module Docstrings

```python
"""
Module description.

This module provides components for [purpose], including:
- Component1: Brief description
- Component2: Brief description

Usage Example:
    ```python
    from sifaka.module import Component
    
    component = Component()
    result = component.method()
    ```
"""
```

### Class Docstrings

```python
"""
Class description.

Detailed description of what this class does and how it works.

Lifecycle:
    1. Initialization: How the class is initialized
    2. Usage: How the class is typically used
    3. Cleanup: Any cleanup steps if applicable

Examples:
    ```python
    from sifaka.module import Class
    
    instance = Class(param1=value1)
    result = instance.method("input")
    ```

Attributes:
    attr1: Description of first attribute
    attr2: Description of second attribute
"""
```

### Method Docstrings

```python
"""
Method description.

Detailed description of what this method does.

Args:
    param1: Description of first parameter
    param2: Description of second parameter

Returns:
    Description of return value

Raises:
    ExceptionType: When and why this exception is raised

Examples:
    ```python
    result = instance.method(param1=value1)
    ```
"""
```

### Factory Function Docstrings

```python
"""
Create a [component] with the specified configuration.

Detailed description of what this factory function does.

Args:
    param1: Description of first parameter
    param2: Description of second parameter
    **kwargs: Additional configuration parameters

Returns:
    Configured component instance

Examples:
    ```python
    component = create_component(param1=value1, param2=value2)
    result = component.method()
    ```
"""
```

## Component-Specific Guidelines

### Rules

Rule docstrings should include:
- What the rule validates
- Configuration options
- Validation behavior for edge cases (empty text, etc.)
- Example usage with factory function

### Validators

Validator docstrings should include:
- Validation logic details
- How to configure the validator
- Error handling behavior
- Example usage with factory function

### Classifiers

Classifier docstrings should include:
- Classification categories/labels
- Confidence score interpretation
- Resource requirements (if applicable)
- Example usage with factory function

### Critics

Critic docstrings should include:
- Critique methodology
- Improvement strategies
- Memory management (if applicable)
- Example usage with factory function

### Chains

Chain docstrings should include:
- Component interaction flow
- Configuration options
- Error handling behavior
- Example usage with factory function

## Type Annotations

Use type annotations consistently:
- Use `Optional[Type]` for optional parameters
- Use `Union[Type1, Type2]` for parameters that can be multiple types
- Use `List[Type]`, `Dict[KeyType, ValueType]`, etc. for container types
- Use `Any` only when absolutely necessary
- Use type variables and generics for flexible typing

## Examples

Examples should:
- Be runnable (if copied and pasted)
- Show typical usage patterns
- Include imports
- Show both simple and complex cases where appropriate
- Demonstrate error handling when relevant

## Best Practices

1. **Keep docstrings up to date** when changing code
2. **Document edge cases and limitations**
3. **Use consistent terminology** throughout the codebase
4. **Explain the "why"** not just the "what"
5. **Include cross-references** to related components
