"""
Chain Formatters Package

Formatting components for Sifaka's chain system.

## Overview
This package provides specialized formatters for different aspects of chain
execution results. These formatters ensure consistent output structure and
formatting across the chain system, making results easier to process and
understand.

## Components
1. **ResultFormatter**: Main result formatting class
   - Chain result formatting
   - Validation feedback formatting
   - Critique feedback formatting
   - Error message generation

## Usage Examples
```python
from sifaka.chain.formatters import ResultFormatter
from sifaka.validation import ValidationResult

# Create formatter
formatter = ResultFormatter[str]()

# Format validation result
validation_result = ValidationResult(
    output="Generated text",
    rule_results=[
        {"rule": "length", "passed": False, "message": "Text too short"}
    ]
)
result = formatter.format_result(
    output="Generated text",
    validation_result=validation_result,
    critique_details={"feedback": "Good but needs improvement"}
)

# Get validation feedback
feedback = formatter.format_feedback_from_validation(validation_result)
print("Validation feedback:", feedback)
```

## Error Handling
- ValueError: Raised for invalid result types
- TypeError: Raised for type validation failures
- KeyError: Raised for missing required fields

## Configuration
- validation_result: ValidationResult object with output and rule results
- critique_details: Optional dictionary with critique information
- output: The generated output to include in results
"""

from .result import ResultFormatter

__all__ = [
    "ResultFormatter",
]
