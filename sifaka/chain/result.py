"""
Chain Result Module

This module provides the ChainResult class for representing the results of chain execution.
It standardizes the format of chain results, including output, validation results, and metadata.

## Components
1. **ChainResult**: Represents the result of running a chain

## Usage Examples
```python
from sifaka.chain.v2.result import ChainResult
from sifaka.chain.v2.interfaces import ValidationResult

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
```
"""

from typing import Any, Dict, List, Optional
import time
from pydantic import BaseModel, Field, computed_field

from .interfaces import ValidationResult


class ChainResult(BaseModel):
    """Result of running a chain."""
    
    output: str = Field(description="The generated output")
    validation_results: List[ValidationResult] = Field(
        default_factory=list, description="Results of validation"
    )
    prompt: str = Field(description="The original prompt")
    execution_time: float = Field(
        default=0.0, ge=0.0, description="Execution time in seconds"
    )
    attempt_count: int = Field(
        default=1, ge=1, description="Number of generation attempts"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Result creation timestamp"
    )
    
    @computed_field
    def all_passed(self) -> bool:
        """
        Check if all validations passed.
        
        Returns:
            True if all validations passed, False otherwise
        """
        if not self.validation_results:
            return True
        return all(r.passed for r in self.validation_results)
    
    @computed_field
    def validation_score(self) -> float:
        """
        Get the average validation score.
        
        Returns:
            The average validation score, or 1.0 if no validations
        """
        if not self.validation_results:
            return 1.0
        return sum(r.score for r in self.validation_results) / len(self.validation_results)
    
    @computed_field
    def issues(self) -> List[str]:
        """
        Get all issues from validation results.
        
        Returns:
            List of all issues
        """
        issues = []
        for result in self.validation_results:
            issues.extend(result.issues)
        return issues
    
    @computed_field
    def suggestions(self) -> List[str]:
        """
        Get all suggestions from validation results.
        
        Returns:
            List of all suggestions
        """
        suggestions = []
        for result in self.validation_results:
            suggestions.extend(result.suggestions)
        return suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
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
        
        Args:
            **kwargs: Metadata key-value pairs
            
        Returns:
            New result with updated metadata
        """
        return self.model_copy(
            update={"metadata": {**self.metadata, **kwargs}}
        )
