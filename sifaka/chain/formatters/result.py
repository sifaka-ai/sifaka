"""
Result Formatter Module

Formats and processes results from Sifaka's chain system.

## Overview
This module provides the ResultFormatter class which handles formatting and
processing of chain execution results. It supports formatting validation
results, critique details, and feedback messages into standardized formats
for consistent handling across the chain system.

## Components
1. **ResultFormatter**: Main result formatting class
   - Result formatting
   - Validation feedback
   - Critique feedback
   - Error message generation

2. **Feedback Generation**: Specialized formatters
   - Validation feedback formatting
   - Critique feedback formatting
   - Error message formatting

## Usage Examples
```python
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.validation import ValidationResult
from sifaka.chain.result import ChainResult

# Create formatter
formatter = ResultFormatter[str]()

# Format validation result
validation_result = ValidationResult(
    output="Generated text",
    rule_results=[
        {"rule": "length", "passed": False, "message": "Text too short"}
    ]
)
feedback = formatter.format_feedback_from_validation(validation_result)
print("Validation feedback:", feedback)

# Format critique details
critique_details = {
    "issues": ["Text lacks clarity", "Poor structure"],
    "suggestions": ["Add more details", "Improve organization"]
}
feedback = formatter.format_feedback_from_critique(critique_details)
print("Critique feedback:", feedback)

# Format complete result
result = formatter.format_result(
    output="Generated text",
    validation_result=validation_result,
    critique_details=critique_details
)
print("Output:", result.output)
print("All rules passed:", all(r.passed for r in result.rule_results))
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

from typing import Any, Dict, Generic, Optional, TypeVar

from ..interfaces.formatter import ResultFormatterProtocol
from ...validation import ValidationResult
from ..result import ChainResult
from ...utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ResultFormatter(
    ResultFormatterProtocol[ValidationResult[OutputType], ChainResult[OutputType]],
    Generic[OutputType],
):
    """
    Formats and processes chain execution results.

    ## Overview
    This class provides centralized formatting and processing of chain execution
    results, including validation results, critique details, and feedback messages.
    It implements the ResultFormatterProtocol interface.

    ## Architecture
    ResultFormatter follows a formatter pattern:
    1. **Result Processing**: Processes raw results into standard format
    2. **Feedback Generation**: Creates formatted feedback messages
    3. **Output Formatting**: Produces consistent output structure

    ## Lifecycle
    1. **Result Handling**: Process execution results
       - Format validation results
       - Format critique details
       - Combine into chain results

    2. **Feedback Generation**: Create feedback messages
       - Process validation feedback
       - Process critique feedback
       - Format error messages

    ## Error Handling
    - ValueError: Raised for invalid result types
    - TypeError: Raised for type validation failures
    - KeyError: Raised for missing required fields

    ## Examples
    ```python
    from sifaka.chain.formatters.result import ResultFormatter

    # Create formatter
    formatter = ResultFormatter[str]()

    # Format validation result
    validation_result = ValidationResult(...)
    feedback = formatter.format_feedback_from_validation(validation_result)

    # Format complete result
    result = formatter.format_result(
        output="Generated text",
        validation_result=validation_result,
        critique_details={"feedback": "Good but needs improvement"}
    )
    ```

    Type parameters:
        OutputType: The type of output being formatted
    """

    def format_result(
        self,
        output: OutputType,
        validation_result: ValidationResult[OutputType],
        critique_details: Optional[Dict[str, Any]] = None,
    ) -> ChainResult[OutputType]:
        """
        Format a chain execution result.

        ## Overview
        This method combines the output, validation result, and optional critique
        details into a standardized ChainResult object. It ensures consistent
        result structure across the chain system.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate output
           - Process validation result
           - Handle critique details

        2. **Result Creation**: Create chain result
           - Combine components
           - Apply standard format
           - Include all details

        Args:
            output: The generated output
            validation_result: The validation result
            critique_details: Optional critique details

        Returns:
            The formatted chain result
        """
        return ChainResult(
            output=output,
            rule_results=validation_result.rule_results,
            critique_details=critique_details,
        )

    def format_feedback_from_validation(
        self, validation_result: ValidationResult[OutputType]
    ) -> str:
        """
        Format feedback from a validation result.

        Args:
            validation_result: The validation result

        Returns:
            The formatted feedback
        """
        feedback = "The following issues were found:\n"
        for result in validation_result.rule_results:
            if not result.passed:
                feedback += f"- {result.message}\n"
        return feedback

    def format_feedback_from_critique(self, critique_details: Dict[str, Any]) -> str:
        """
        Format feedback from critique details.

        Args:
            critique_details: The critique details

        Returns:
            The formatted feedback
        """
        if "feedback" in critique_details:
            return critique_details["feedback"]

        feedback = "The following issues were found:\n"

        if "issues" in critique_details and critique_details["issues"]:
            for issue in critique_details["issues"]:
                feedback += f"- {issue}\n"

        if "suggestions" in critique_details and critique_details["suggestions"]:
            feedback += "\nSuggestions for improvement:\n"
            for suggestion in critique_details["suggestions"]:
                feedback += f"- {suggestion}\n"

        return feedback

    def format(self, result: ValidationResult[OutputType]) -> ChainResult[OutputType]:
        """
        Format a result.

        This method implements the ResultFormatterProtocol.format method.

        Args:
            result: The result to format

        Returns:
            A formatted result

        Raises:
            ValueError: If the result is invalid
        """
        if not isinstance(result, ValidationResult):
            raise ValueError(f"Expected ValidationResult, got {type(result)}")

        return ChainResult(
            output=result.output,
            rule_results=result.rule_results,
            critique_details=None,
        )
