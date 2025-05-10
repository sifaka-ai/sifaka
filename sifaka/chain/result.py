"""
Chain Result Module

This module provides the ChainResult class which represents the result of running a chain,
including the generated output, validation results, and optional critique details.

## Overview
The ChainResult class is a standardized representation of chain execution results,
with support for validation results, critique details, and execution statistics.
It extends the BaseResult class from the core module to provide a consistent
result structure across the Sifaka framework.

## Components
1. **ChainResult**: Main result class that encapsulates chain execution results
2. **RuleResult**: Individual rule validation results from the rules module
3. **CritiqueDetails**: Optional details from critic refinement

## Usage Examples
```python
from sifaka.chain import ChainResult
from sifaka.rules import RuleResult

# Create a chain result
result = ChainResult(
    output="Generated text output",
    rule_results=[
        RuleResult(
            rule_name="length_rule",
            passed=True,
            details={"length": 100}
        )
    ],
    critique_details={
        "feedback": "Good text, but could be more concise",
        "suggestions": ["Remove redundant phrases", "Use active voice"]
    }
)

# Access result properties
print(f"Output: {result.output}")
print(f"All rules passed: {result.all_passed}")
if result.critique_details:
    print(f"Critique feedback: {result.critique_details.get('feedback', '')}")
```

## Error Handling
- ValueError: Raised when validation fails
- TypeError: Raised when type validation fails

## Configuration
- output: The generated output from the chain
- rule_results: List of validation results from rules
- critique_details: Optional dictionary of critique details
- attempts: Number of generation attempts made
- metadata: Additional metadata about the result
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import Field, ConfigDict

from sifaka.core.base import BaseResult, ComponentResultEnum
from ..rules import RuleResult

OutputType = TypeVar("OutputType")


class ChainResult(BaseResult, Generic[OutputType]):
    """
    Result from running a chain, including the output and validation details.

    This class provides a standardized representation of chain execution results,
    with support for validation results, critique details, and execution statistics.
    It extends the BaseResult class to provide a consistent result structure across
    the Sifaka framework.

    ## Architecture
    ChainResult follows a structured data model:
    1. **Core Data**: Essential data
       - Output: Generated output
       - Rule Results: Validation results
       - Critique Details: Optional feedback
    2. **Metadata**: Additional information
       - Attempts: Number of generation attempts
       - Execution Time: Processing time statistics
       - Status: Success or failure status

    ## Lifecycle
    1. **Creation**: Initialize with data
       - Set output and validation results
       - Set critique details if available
       - Record execution statistics
    2. **Usage**: Access data and status
       - Read output and validation results
       - Check validation status
       - Get feedback and suggestions
    3. **Analysis**: Examine execution details
       - Check number of attempts
       - Review execution time
       - Analyze validation failures

    ## Error Handling
    - ValueError: Raised when validation fails
    - TypeError: Raised when type validation fails

    ## Examples
    ```python
    from sifaka.chain import ChainResult
    from sifaka.rules import RuleResult

    # Create a chain result
    result = ChainResult(
        output="Generated text output",
        rule_results=[
            RuleResult(
                rule_name="length_rule",
                passed=True,
                details={"length": 100}
            )
        ],
        critique_details={
            "feedback": "Good text, but could be more concise",
            "suggestions": ["Remove redundant phrases", "Use active voice"]
        },
        attempts=2,
        metadata={"model": "gpt-3.5-turbo", "temperature": 0.7}
    )

    # Access result properties
    print(f"Output: {result.output}")
    print(f"All rules passed: {result.all_passed}")
    if result.critique_details:
        print(f"Critique feedback: {result.critique_details.get('feedback', '')}")
    ```

    Attributes:
        output (OutputType): The generated output from the chain
        rule_results (List[RuleResult]): List of validation results from rules
        critique_details (Optional[Dict[str, Any]]): Optional dictionary of critique details
        attempts (int): Number of generation attempts made
        status (ComponentResultEnum): Status of the result (SUCCESS, FAILURE, etc.)
        metadata (Dict[str, Any]): Additional metadata about the result
        processing_time_ms (float): Processing time in milliseconds
        timestamp (datetime): When the result was created
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    output: OutputType = Field(description="The generated output from the chain")
    rule_results: List[RuleResult] = Field(
        default_factory=list, description="List of validation results from rules"
    )
    critique_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional dictionary of critique details"
    )
    attempts: int = Field(default=1, description="Number of generation attempts made", ge=1)
    status: ComponentResultEnum = Field(
        default=ComponentResultEnum.SUCCESS, description="Status of the result"
    )

    @property
    def all_passed(self) -> bool:
        """Check if all rules passed validation."""
        return all(r.passed for r in self.rule_results)

    @property
    def failed_rules(self) -> List[RuleResult]:
        """Get list of failed rule results."""
        return [r for r in self.rule_results if not r.passed]

    def with_critique(self, critique_details: Dict[str, Any]) -> "ChainResult[OutputType]":
        """Create a new result with updated critique details."""
        return self.model_copy(update={"critique_details": critique_details})

    def with_attempts(self, attempts: int) -> "ChainResult[OutputType]":
        """Create a new result with updated attempts count."""
        return self.model_copy(update={"attempts": attempts})

    def get_formatted_feedback(self) -> str:
        """Get formatted feedback from rule results and critique details."""
        feedback = []

        # Add rule feedback
        if self.rule_results:
            feedback.append("Rule validation results:")
            for i, rule in enumerate(self.rule_results, 1):
                status = "✓" if rule.passed else "✗"
                feedback.append(f"  {i}. {status} {rule.rule_name}: {rule.message}")

        # Add critique feedback
        if self.critique_details:
            feedback.append("\nCritique feedback:")
            if "feedback" in self.critique_details:
                feedback.append(f"  {self.critique_details['feedback']}")
            if "suggestions" in self.critique_details:
                feedback.append("\nSuggestions:")
                for i, suggestion in enumerate(self.critique_details["suggestions"], 1):
                    feedback.append(f"  {i}. {suggestion}")

        return "\n".join(feedback)
