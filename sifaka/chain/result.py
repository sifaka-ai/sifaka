"""
Chain Result Module

This module provides the ChainResult class for representing the results of chain execution.
It standardizes the format of chain results, including output, validation results, and metadata.

## Overview
The ChainResult class encapsulates all information related to a chain execution, including
the generated output, validation results, execution metrics, and additional metadata.
It provides computed properties for easy access to validation status, scores, issues, and suggestions.

## Components
1. **ChainResult**: Represents the result of running a chain, with methods for accessing
   validation information and converting to different formats.
2. **ValidationResult**: Referenced from interfaces module, represents individual validation results.

## Usage Examples
```python
from sifaka.chain.interfaces import ValidationResult
from sifaka.chain.result import ChainResult

# Create validation results
validation_results = [
    ValidationResult(
        passed=True,
        message="Length validation passed",
        score=1.0
    ),
    ValidationResult(
        passed=False,
        message="Toxicity validation failed",
        score=0.3,
        issues=["Contains toxic content"],
        suggestions=["Remove toxic content"]
    )
]

# Create chain result
result = ChainResult(
    output="Generated text",
    validation_results=validation_results,
    prompt="Write a story",
    execution_time=0.5,
    attempt_count=2
)

# Access result properties
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
print(f"Validation score: {result.validation_score}")
print(f"Execution time: {result.execution_time} seconds")
print(f"Attempt count: {result.attempt_count}")

# Get issues and suggestions
for issue in result.issues:
    print(f"Issue: {issue}")
for suggestion in result.suggestions:
    print(f"Suggestion: {suggestion}")

# Add additional metadata
enhanced_result = result.with_metadata(model_name="gpt-4", temperature=0.7)
```

## Error Handling
The ChainResult class is designed to be robust and handle edge cases gracefully:
- Empty validation results are handled appropriately in computed properties
- The class provides immutable operations through methods like `with_metadata()`

## Configuration
The ChainResult class does not require specific configuration but works with the
following components:
- ValidationResult objects from the interfaces module
- Pydantic's BaseModel for data validation and serialization
"""

from typing import Any, Dict, List
import time
from pydantic import BaseModel, Field, computed_field

from .interfaces import ValidationResult


class ChainResult(BaseModel):
    """
    Result of running a chain.

    This class encapsulates all information related to a chain execution, including
    the generated output, validation results, execution metrics, and additional metadata.
    It provides computed properties for easy access to validation status, scores, issues,
    and suggestions.

    ## Architecture
    The ChainResult class is designed as an immutable Pydantic model with fields for
    storing chain execution results and computed properties for derived information.
    It follows a value object pattern, where new instances are created rather than
    modifying existing ones.

    ## Lifecycle
    1. Creation: Instantiated with output and optional validation results
    2. Usage: Accessed to retrieve output, validation status, and metadata
    3. Extension: Enhanced with additional metadata using `with_metadata()`

    ## Error Handling
    The ChainResult class handles edge cases gracefully:
    - Empty validation results are handled appropriately in computed properties
    - Invalid inputs are validated through Pydantic's validation system
    - Immutable operations prevent accidental state modification

    ## Examples
    ```python
    # Create a basic result
    result = ChainResult(
        output="Generated text",
        prompt="Write a story",
        validation_results=[validation_result1, validation_result2]
    )

    # Check validation status
    if result.all_passed:
        print("All validations passed!")
    else:
        print(f"Validation score: {result.validation_score}")
        print(f"Issues: {result.issues}")
    ```

    Attributes:
        output (str): The generated output text
        validation_results (List[ValidationResult]): Results of validation
        prompt (str): The original prompt
        execution_time (float): Execution time in seconds
        attempt_count (int): Number of generation attempts
        metadata (Dict[str, Any]): Additional metadata
        timestamp (float): Result creation timestamp
    """

    output: str = Field(description="The generated output")
    validation_results: List[ValidationResult] = Field(
        default_factory=list, description="Results of validation"
    )
    prompt: str = Field(description="The original prompt")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    attempt_count: int = Field(default=1, ge=1, description="Number of generation attempts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: float = Field(default_factory=time.time, description="Result creation timestamp")

    @computed_field
    def all_passed(self) -> bool:
        """
        Check if all validations passed.

        Returns True if all validations passed or if there are no validation results.
        When no validation results are present, this is considered a pass by default.

        Returns:
            bool: True if all validations passed or no validations exist, False otherwise

        Example:
            ```python
            if result.all_passed:
                print("All validations passed!")
            ```
        """
        if not self.validation_results:
            return True
        return all(r.passed for r in self.validation_results)

    @computed_field
    def validation_score(self) -> float:
        """
        Get the average validation score across all validation results.

        Calculates the arithmetic mean of all validation scores. If no validation
        results are present, returns 1.0 (perfect score) by default.

        Returns:
            float: The average validation score (between 0.0 and 1.0), or 1.0 if no validations

        Example:
            ```python
            score = result.validation_score
            print(f"Overall validation score: {score:.2f}")
            ```
        """
        if not self.validation_results:
            return 1.0
        return sum(r.score for r in self.validation_results) / len(self.validation_results)

    @computed_field
    def issues(self) -> List[str]:
        """
        Get all issues from validation results.

        Collects and flattens all issues from all validation results into a single list.
        This provides a convenient way to access all identified issues across multiple
        validation steps.

        Returns:
            List[str]: Consolidated list of all issues from all validation results

        Example:
            ```python
            for issue in result.issues:
                print(f"Issue found: {issue}")
            ```
        """
        issues = []
        for result in self.validation_results:
            issues.extend(result.issues)
        return issues

    @computed_field
    def suggestions(self) -> List[str]:
        """
        Get all suggestions from validation results.

        Collects and flattens all suggestions from all validation results into a single list.
        This provides a convenient way to access all improvement suggestions across multiple
        validation steps.

        Returns:
            List[str]: Consolidated list of all suggestions from all validation results

        Example:
            ```python
            for suggestion in result.suggestions:
                print(f"Suggested improvement: {suggestion}")
            ```
        """
        suggestions = []
        for result in self.validation_results:
            suggestions.extend(result.suggestions)
        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Creates a complete dictionary representation of the ChainResult, including
        all fields and computed properties. This is useful for serialization or
        when working with frameworks that expect dictionary inputs.

        Returns:
            Dict[str, Any]: Dictionary representation of the result with all properties

        Example:
            ```python
            result_dict = result.to_dict()
            json_result = json.dumps(result_dict)
            ```
        """
        return {
            "output": self.output,
            "validation_results": [r.model_dump() for r in self.validation_results],
            "prompt": self.prompt,
            "execution_time": self.execution_time,
            "attempt_count": self.attempt_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "validation_score": self.validation_score,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }

    def with_metadata(self, **kwargs: Any) -> "ChainResult":
        """
        Create a new result with additional metadata.

        Creates an immutable copy of the current result with additional metadata merged
        with the existing metadata. This is useful for adding context or additional
        information without modifying the original result.

        Args:
            **kwargs: Metadata key-value pairs to add to the result

        Returns:
            ChainResult: New result instance with updated metadata

        Example:
            ```python
            # Add model information to the result
            enhanced_result = result.with_metadata(
                model_name="gpt-4",
                temperature=0.7,
                top_p=0.95
            )
            ```
        """
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})
