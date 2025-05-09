"""
Formatter Interface Module

Protocol interfaces for Sifaka's formatter system.

## Overview
This module defines the core protocol interfaces for result formatters in the
Sifaka framework. These interfaces establish a common contract for formatter
behavior, enabling better modularity, extensibility, and interoperability
between different formatter implementations.

## Components
1. **ResultFormatterProtocol**: Base result formatter interface
   - Result formatting
   - Type conversion
   - Error handling
   - Resource management

## Usage Examples
```python
from typing import Any, Dict
from sifaka.chain.interfaces.formatter import ResultFormatterProtocol

class DictFormatter(ResultFormatterProtocol[Any, Dict[str, Any]]):
    def format(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return {"text": result}
        elif isinstance(result, (int, float)):
            return {"value": result}
        else:
            raise ValueError(f"Cannot format result of type {type(result)}")

# Use the formatter
formatter = DictFormatter()
try:
    # Format string result
    str_result = formatter.format("Hello, world!")
    print("String result:", str_result)  # {"text": "Hello, world!"}

    # Format numeric result
    num_result = formatter.format(42)
    print("Numeric result:", num_result)  # {"value": 42}

    # Format dict result
    dict_result = formatter.format({"key": "value"})
    print("Dict result:", dict_result)  # {"key": "value"}
except ValueError as e:
    print(f"Formatting error: {e}")
```

## Error Handling
- ValueError: Raised for invalid result types
- TypeError: Raised for type conversion failures
- RuntimeError: Raised for formatting failures

## Configuration
- input_type: Type of input to format
- result_type: Type of formatted output
- type_conversions: Mapping of type conversions
"""

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
ResultType = TypeVar("ResultType", covariant=True)


@runtime_checkable
class ResultFormatterProtocol(Protocol[InputType, ResultType]):
    """
    Interface for result formatters.

    Detailed description of what the class does, including:
    - Defines the contract for components that format results
    - Ensures consistent formatting behavior across implementations
    - Handles type conversions and formatting rules
    - Manages formatting errors and resources

    Type parameters:
        InputType: The type of input to format
        ResultType: The type of formatted output

    Example:
        ```python
        class SimpleFormatter(ResultFormatterProtocol[Any, Dict[str, Any]]):
            def format(self, result: Any) -> Dict[str, Any]:
                if isinstance(result, dict):
                    return result
                elif isinstance(result, str):
                    return {"text": result}
                elif isinstance(result, (int, float)):
                    return {"value": result}
                else:
                    raise ValueError(f"Cannot format {type(result)}")

        # Use the formatter
        formatter = SimpleFormatter()
        result = formatter.format("Hello")  # {"text": "Hello"}
        ```
    """

    @abstractmethod
    def format(self, result: InputType) -> ResultType:
        """
        Format a result.

        Detailed description of what the method does, including:
        - Formats an input result into the desired output format
        - Handles type conversions and formatting rules
        - Ensures consistent output structure
        - Validates input and output types

        Args:
            result: The result to format

        Returns:
            The formatted result

        Raises:
            ValueError: If the result is invalid
            TypeError: If type conversion fails
            RuntimeError: If formatting fails

        Example:
            ```python
            # Format a result
            try:
                formatted = formatter.format(
                    result="Hello, world!"
                )
                print(f"Formatted result: {formatted}")
            except ValueError as e:
                print(f"Formatting error: {e}")
            ```
        """
        pass
