# Contributing to Sifaka

## Documentation Standards

### 1. Module Documentation
Every module should have a comprehensive docstring at the top of the file following this structure:

```python
"""
Module Name

A brief description of the module's purpose and functionality.

## Overview
Detailed explanation of the module's role in the system.

## Components
List and description of main components in the module.

## Usage Examples
Code examples showing common usage patterns.

## Error Handling
Description of error handling strategies and common exceptions.

## Configuration
Documentation of configuration options and parameters.
"""
```

### 2. Class Documentation
Every class should have a detailed docstring:

```python
class ExampleClass:
    """
    Brief description of the class.

    Detailed description of the class's purpose, behavior, and role in the system.

    Attributes:
        attr1 (type): Description of the first attribute
        attr2 (type): Description of the second attribute

    Methods:
        method1(): Description of the first method
        method2(): Description of the second method

    Example:
        ```python
        # Example usage
        instance = ExampleClass()
        result = instance.method1()
        ```
    """
```

### 3. Method Documentation
Every public method should have a docstring:

```python
def example_method(param1: str, param2: int) -> bool:
    """
    Brief description of the method.

    Detailed description of what the method does, including:
    - Purpose and functionality
    - Parameter descriptions
    - Return value description
    - Side effects
    - Exceptions raised

    Args:
        param1 (str): Description of the first parameter
        param2 (int): Description of the second parameter

    Returns:
        bool: Description of the return value

    Raises:
        ValueError: Description of when this exception is raised
        TypeError: Description of when this exception is raised

    Example:
        ```python
        # Example usage
        result = example_method("test", 42)
        ```
    """
```

### 4. Type Hints
- Use Python type hints for all function parameters and return values
- Use generics (TypeVar, Generic) for flexible types
- Document complex type relationships

### 5. Examples
- Include practical examples in docstrings
- Show common usage patterns
- Demonstrate error handling
- Include configuration examples

### 6. Error Handling
- Document all possible exceptions
- Explain error recovery strategies
- Provide error message examples
- Document error codes if applicable

### 7. Configuration
- Document all configuration options
- Provide default values
- Explain parameter effects
- Include validation rules

## Documentation Checklist

### For New Features
- [ ] Module-level documentation
- [ ] Class documentation
- [ ] Method documentation
- [ ] Type hints
- [ ] Usage examples
- [ ] Error handling
- [ ] Configuration options
- [ ] Update architecture docs if needed

### For Bug Fixes
- [ ] Document the bug
- [ ] Explain the fix
- [ ] Update affected documentation
- [ ] Add regression tests

### For API Changes
- [ ] Update interface documentation
- [ ] Document breaking changes
- [ ] Update examples
- [ ] Update type hints

## Best Practices

### 1. Writing Style
- Use clear, concise language
- Be consistent in terminology
- Use active voice
- Avoid jargon without explanation

### 2. Code Examples
- Keep examples simple and focused
- Show common use cases
- Include error handling
- Test all examples

### 3. Organization
- Group related information
- Use consistent formatting
- Include cross-references
- Maintain a logical flow

### 4. Maintenance
- Keep documentation up to date
- Review documentation with code changes
- Update examples as needed
- Remove outdated information

## Tools and Resources

### 1. Documentation Tools
- Sphinx for API documentation
- MkDocs for user guides
- doctest for testing examples
- mypy for type checking

### 2. Style Guides
- Google Python Style Guide
- NumPy Documentation Style Guide
- PEP 257 - Docstring Conventions
- PEP 484 - Type Hints

### 3. Templates
- Module template
- Class template
- Method template
- Configuration template

## Review Process

### 1. Documentation Review
- Check completeness
- Verify accuracy
- Test examples
- Review formatting

### 2. Code Review
- Check type hints
- Verify docstrings
- Test documentation
- Review examples

### 3. Final Steps
- Update version numbers
- Update changelog
- Update index
- Build documentation