## Docstring Template Examples

### Module Docstring Template

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

### Class Docstring Template

```python
class MyClass:
    """
    Brief description of the class.

    Detailed description of the class's purpose, functionality, and usage.

    ## Architecture
    Description of the class's architecture and design patterns.

    ## Lifecycle
    Description of the class's lifecycle (initialization, operation, cleanup).

    ## Error Handling
    Description of how the class handles errors and exceptions.

    ## Examples
    Code examples showing common usage patterns.

    Attributes:
        attr1 (type): Description of attribute 1
        attr2 (type): Description of attribute 2
    """
```

### Method Docstring Template

```python
def my_method(param1: str, param2: int) -> bool:
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
        result = my_method("test", 42)
        ```
    """
```

## Next Steps

1. **Create Docstring Templates**: Create templates for each component type
2. **Update High Priority Components**: Focus on standardizing high-priority components
3. **Add Docstring Tests**: Add tests to verify docstring examples work correctly
4. **Update Documentation**: Update API reference documentation with standardized docstrings
