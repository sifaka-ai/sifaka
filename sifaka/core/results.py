"""
Consolidated Result Classes for Sifaka

This module provides a standardized set of result classes for all Sifaka components.
It consolidates previously scattered result implementations into a single, consistent module.

## Overview
This module defines the base result classes that all component-specific results inherit from,
as well as specialized result classes for different component types. It provides a unified
interface for result handling across the Sifaka framework.

## Components
1. **BaseResult**: Base class for all results
2. **RuleResult**: Results from rules and validators
3. **ClassificationResult**: Results from classifiers
4. **ChainResult**: Results from chains
5. **CriticResult**: Results from critics
6. **RetrievalResult**: Results from retrieval operations
7. **ModelResult**: Results from model operations
8. **Factory functions**: For creating standardized results

## Usage Examples
```python
from sifaka.core.results import (
    BaseResult, RuleResult, ClassificationResult, ChainResult,
    create_rule_result, create_classification_result, create_chain_result
)

# Create a rule result
rule_result = create_rule_result(
    passed=True,
    message="Validation passed",
    rule_name="LengthRule",
    severity="info"
)

# Create a classification result
classification_result = create_classification_result(
    label="positive",
    confidence=0.85,
    metadata={"model": "sentiment-classifier"}
)

# Create a chain result
chain_result = create_chain_result(
    output="Generated text",
    prompt="Write a story",
    validation_results=[rule_result],
    execution_time=0.5
)
```

## Error Handling
The module provides standardized error result creation with:
- Consistent error metadata structure
- Component identification
- Error type classification
- Severity levels

## Configuration
Result creation can be configured with:
- Custom metadata
- Component identification
- Error type classification
- Severity levels
- Processing time tracking
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Type, cast

from pydantic import BaseModel, Field, ConfigDict, computed_field

# Type variables
T = TypeVar("T")  # Generic type for results
L = TypeVar("L")  # Label type for classification results
OutputType = TypeVar("OutputType")  # Output type for model results


class BaseResult(BaseModel, Generic[T]):
    """
    Base result for all components.

    This class provides a standardized result model for all Sifaka components,
    defining common result fields that are shared across different component types.

    Attributes:
        passed: Whether the operation passed
        message: Result message
        metadata: Dictionary of additional metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds
        timestamp: Result timestamp
    """

    passed: bool
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    def with_metadata(self, **kwargs: Any) -> "BaseResult[T]":
        """Create a new result with additional metadata."""
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})

    def with_issues(self, issues: List[str]) -> "BaseResult[T]":
        """Create a new result with updated issues."""
        return self.model_copy(update={"issues": issues})

    def with_suggestions(self, suggestions: List[str]) -> "BaseResult[T]":
        """Create a new result with updated suggestions."""
        return self.model_copy(update={"suggestions": suggestions})

    def with_score(self, score: float) -> "BaseResult[T]":
        """Create a new result with updated score."""
        return self.model_copy(update={"score": score})


class RuleResult(BaseResult):
    """
    Result for rule validation.

    This class extends BaseResult to add rule-specific result information.

    Attributes:
        rule_name: Name of the rule that produced this result
        severity: Severity level of the result (error, warning, info)
        category: Category of the rule that produced this result
        tags: List of tags associated with the rule
        rule_id: Identifier of the rule that produced this result
    """

    rule_name: str = Field(
        default="unnamed_rule",
        description="Name of the rule that produced this result",
    )
    severity: str = Field(
        default="error",
        description="Severity level of the result",
    )
    category: str = Field(
        default="general",
        description="Category of the rule that produced this result",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the rule",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Identifier of the rule that produced this result",
    )

    def with_rule_id(self, rule_id: str) -> "RuleResult":
        """Create a new result with the rule ID set."""
        return self.model_copy(update={"rule_id": rule_id})

    def with_severity(self, severity: str) -> "RuleResult":
        """Create a new result with updated severity."""
        return self.model_copy(update={"severity": severity})

    def with_category(self, category: str) -> "RuleResult":
        """Create a new result with updated category."""
        return self.model_copy(update={"category": category})

    def with_tags(self, tags: List[str]) -> "RuleResult":
        """Create a new result with updated tags."""
        return self.model_copy(update={"tags": tags})


class ClassificationResult(BaseResult, Generic[T, L]):
    """
    Result for classification operations.

    This class extends BaseResult to add classification-specific result information.

    Attributes:
        label: Classification label
        confidence: Confidence score for the classification
    """

    label: L
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CriticResult(BaseResult):
    """
    Result for critic operations.

    This class extends BaseResult to add critic-specific result information.

    Attributes:
        feedback: Human-readable feedback
    """

    feedback: str = Field(
        description="Human-readable feedback",
    )


class ValidationResult(BaseResult[Any]):
    """
    Result for validation operations.

    This class provides a standardized result for validation operations.
    It is used by ChainResult to track validation results.

    This class is compatible with sifaka.chain.interfaces.ValidationResult
    to ensure type compatibility across the codebase.

    Attributes:
        passed: Whether the validation passed
        message: Validation message
        score: Validation score (0.0 to 1.0)
        issues: List of issues found
        suggestions: List of improvement suggestions
        metadata: Additional metadata
    """

    # All required fields are already defined in BaseResult

    @classmethod
    def from_interface_validation_result(cls, result: Any) -> "ValidationResult":
        """
        Create a ValidationResult from an interface ValidationResult.

        Args:
            result: The interface ValidationResult to convert

        Returns:
            A new ValidationResult instance
        """
        if isinstance(result, ValidationResult):
            return result

        # Check if it has the required attributes
        if hasattr(result, "passed") and hasattr(result, "message"):
            return cls(
                passed=result.passed,
                message=result.message,
                score=getattr(result, "score", 0.0),
                issues=getattr(result, "issues", []),
                suggestions=getattr(result, "suggestions", []),
                metadata=getattr(result, "metadata", {}),
            )

        # If it doesn't have the required attributes, create a default result
        return cls(
            passed=False,
            message="Invalid validation result",
            score=0.0,
            issues=["The provided validation result is not compatible"],
            suggestions=["Use a compatible ValidationResult class"],
        )


class ModelResult(BaseResult, Generic[OutputType]):
    """
    Result of a model operation.

    This class provides an immutable representation of a model operation result,
    including the generated output and additional metadata.

    ## Lifecycle

    1. **Creation**: Instantiate with operation results
       - Provide output (required)
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get operation details
       - Access output for the operation result
       - Examine metadata for additional information

    Attributes:
        output: The generated output
        metadata: Additional metadata about the operation
    """

    output: OutputType = Field(description="The generated output")


class GenerationResult(ModelResult[str]):
    """
    Result of a text generation operation.

    This class extends ModelResult with text generation specific fields,
    providing a standardized format for text generation results.

    ## Lifecycle

    1. **Creation**: Instantiate with generation results
       - Provide output (required)
       - Add model_name, prompt_tokens, and completion_tokens
       - Add optional metadata dictionary
       - Values are validated during creation

    2. **Access**: Read properties to get generation details
       - Access output for the generated text
       - Check prompt_tokens and completion_tokens for token usage
       - Examine metadata for additional information

    Attributes:
        output: The generated text
        model_name: Name of the model used for generation
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens (prompt + completion)
        metadata: Additional metadata about the operation
    """

    model_name: str = Field(description="Name of the model used for generation")
    prompt_tokens: int = Field(default=0, ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        default=0, ge=0, description="Number of tokens in the completion"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total number of tokens")


class ErrorResult(BaseResult):
    """
    Result for error conditions.

    This class provides a standardized result for error conditions,
    including error type, message, and additional metadata.

    Attributes:
        error_type: Type of error
        error_message: Detailed error message
        metadata: Additional error metadata
    """

    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Detailed error message")


class ChainResult(BaseResult):
    """
    Result for chain operations.

    This class extends BaseResult to add chain-specific result information.

    Attributes:
        output: The generated output
        validation_results: Results of validation
        prompt: The original prompt
        execution_time: Execution time in seconds
        attempt_count: Number of generation attempts
    """

    output: str = Field(description="The generated output")
    validation_results: List[ValidationResult] = Field(
        default_factory=list, description="Results of validation"
    )
    prompt: str = Field(description="The original prompt")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    attempt_count: int = Field(default=1, ge=1, description="Number of generation attempts")

    @classmethod
    def from_interface_validation_results(
        cls,
        output: str,
        validation_results: List[Any],  # Accept any ValidationResult-like objects
        prompt: str,
        passed: bool = True,
        message: str = "Chain execution completed",
        execution_time: float = 0.0,
        attempt_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        score: float = 1.0,
        issues: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        processing_time_ms: Optional[float] = None,
    ) -> "ChainResult":
        """
        Create a ChainResult from interface ValidationResult objects.

        This method allows creating a ChainResult from ValidationResult objects
        defined in sifaka.interfaces.chain.models, ensuring type compatibility.

        Args:
            output: The generated output
            validation_results: Results of validation (from interfaces.chain.models.ValidationResult)
            prompt: The original prompt
            passed: Whether the chain execution passed
            message: Human-readable result message
            execution_time: Execution time in seconds
            attempt_count: Number of generation attempts
            metadata: Additional result metadata
            score: Confidence score (0.0 to 1.0)
            issues: List of identified issues
            suggestions: List of improvement suggestions
            processing_time_ms: Processing time in milliseconds

        Returns:
            A new ChainResult instance
        """
        # Convert interface ValidationResults to core ValidationResults if needed
        core_validation_results = []

        # Convert each validation result to a core ValidationResult
        for result in validation_results:
            # Use the helper method to convert the result
            core_validation_results.append(
                ValidationResult.from_interface_validation_result(result)
            )

        return cls(
            output=output,
            validation_results=core_validation_results,
            prompt=prompt,
            execution_time=execution_time,
            attempt_count=attempt_count,
            passed=passed,
            message=message,
            metadata=metadata or {},
            score=score,
            issues=issues or [],
            suggestions=suggestions or [],
            processing_time_ms=processing_time_ms or 0.0,
        )

    @computed_field
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        if not self.validation_results:
            return True
        return all(r.passed for r in self.validation_results)

    @computed_field
    def validation_score(self) -> float:
        """Get the average validation score across all validation results."""
        if not self.validation_results:
            return 1.0
        return sum(r.score for r in self.validation_results) / len(self.validation_results)

    @computed_field
    def validation_issues(self) -> List[str]:
        """Get all issues from validation results."""
        issues = []
        for result in self.validation_results:
            issues.extend(result.issues)
        return issues

    @computed_field
    def validation_suggestions(self) -> List[str]:
        """Get all suggestions from validation results."""
        suggestions = []
        for result in self.validation_results:
            suggestions.extend(result.suggestions)
        return suggestions


class RetrievalResult(BaseResult, Generic[T]):
    """
    Result of a retrieval operation.

    This class defines the result of a retrieval operation,
    including the retrieved documents and query information.
    It extends BaseResult to provide a consistent result structure
    across the Sifaka framework.

    Attributes:
        documents: The retrieved documents
        query: The query used for retrieval
        processed_query: The processed query (after preprocessing)
        total_results: The total number of results found
    """

    documents: List[Any] = Field(
        default_factory=list,
        description="The retrieved documents",
    )
    query: str = Field(
        description="The query used for retrieval",
    )
    processed_query: Optional[str] = Field(
        default=None,
        description="The processed query (after preprocessing)",
    )
    total_results: int = Field(
        default=0,
        description="The total number of results found",
        ge=0,
    )

    @property
    def top_document(self) -> Optional[Any]:
        """
        Get the top document from the results.

        Returns:
            The top document, or None if no documents were retrieved
        """
        if not self.documents:
            return None
        return self.documents[0]

    @property
    def top_content(self) -> Optional[T]:
        """
        Get the content of the top document.

        Returns:
            The content of the top document, or None if no documents were retrieved
        """
        top_doc = self.top_document
        if top_doc is None:
            return None
        # Cast the content to the correct type to satisfy mypy
        content: T = top_doc.content  # type: ignore
        return content

    @property
    def has_results(self) -> bool:
        """
        Check if the result has any documents.

        Returns:
            True if there are documents, False otherwise
        """
        return len(self.documents) > 0

    def get_contents(self) -> List[T]:
        """
        Get the contents of all retrieved documents.

        Returns:
            A list of document contents
        """
        return [doc.content for doc in self.documents]


# Factory functions for creating standardized results


def create_base_result(
    passed: bool,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 0.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> BaseResult:
    """
    Create a standardized base result.

    Args:
        passed: Whether the validation passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized BaseResult
    """
    return BaseResult(
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_rule_result(
    passed: bool,
    message: str,
    rule_name: str = "unnamed_rule",
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "error",
    category: str = "general",
    tags: Optional[List[str]] = None,
    rule_id: Optional[str] = None,
    score: float = 0.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> RuleResult:
    """
    Create a standardized rule result.

    Args:
        passed: Whether the validation passed
        message: Human-readable result message
        rule_name: Name of the rule that produced this result
        metadata: Additional result metadata
        severity: Severity level of the result
        category: Category of the rule that produced this result
        tags: List of tags associated with the rule
        rule_id: Identifier of the rule that produced this result
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized RuleResult
    """
    return RuleResult(
        passed=passed,
        message=message,
        rule_name=rule_name,
        metadata=metadata or {},
        severity=severity,
        category=category,
        tags=tags or [],
        rule_id=rule_id,
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_classification_result(
    label: L,
    confidence: float,
    passed: bool = True,
    message: str = "Classification completed",
    metadata: Optional[Dict[str, Any]] = None,
    score: Optional[float] = None,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> ClassificationResult[Any, L]:
    """
    Create a standardized classification result.

    Args:
        label: Classification label
        confidence: Confidence score (0.0 to 1.0)
        passed: Whether the classification passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Optional score (defaults to confidence if not provided)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized ClassificationResult
    """
    if score is None:
        score = confidence

    return ClassificationResult(
        label=label,
        confidence=confidence,
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_critic_result(
    score: float,
    feedback: str,
    passed: Optional[bool] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> CriticResult:
    """
    Create a standardized critic result.

    Args:
        score: Quality score (0.0 to 1.0)
        feedback: Human-readable feedback
        passed: Whether the critique passed (defaults to score >= 0.5)
        message: Human-readable message (defaults to feedback)
        metadata: Additional result metadata
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized CriticResult
    """
    if passed is None:
        passed = score >= 0.5

    if message is None:
        message = feedback

    return CriticResult(
        score=score,
        feedback=feedback,
        passed=passed,
        message=message,
        metadata=metadata or {},
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_chain_result(
    output: str,
    prompt: str,
    validation_results: Optional[List[Any]] = None,  # Accept any ValidationResult-like objects
    passed: bool = True,
    message: str = "Chain execution completed",
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 1.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    execution_time: float = 0.0,
    attempt_count: int = 1,
    processing_time_ms: Optional[float] = None,
) -> ChainResult:
    """
    Create a standardized chain result.

    This function accepts ValidationResult objects from either sifaka.core.results
    or sifaka.interfaces.chain.models, ensuring type compatibility across the codebase.

    Args:
        output: The generated output
        prompt: The original prompt
        validation_results: Results of validation (from either ValidationResult class)
        passed: Whether the chain execution passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        execution_time: Execution time in seconds
        attempt_count: Number of generation attempts
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized ChainResult
    """
    # Handle None case
    if validation_results is None:
        validation_results = []

    # Check if any validation result is not a core ValidationResult
    if any(not isinstance(vr, ValidationResult) for vr in validation_results):
        # If any validation result is not a core ValidationResult, use the conversion method
        return ChainResult.from_interface_validation_results(
            output=output,
            prompt=prompt,
            validation_results=validation_results,
            passed=passed,
            message=message,
            metadata=metadata,
            score=score,
            issues=issues,
            suggestions=suggestions,
            execution_time=execution_time,
            attempt_count=attempt_count,
            processing_time_ms=processing_time_ms,
        )
    else:
        # All validation results are core ValidationResults or the list is empty
        return ChainResult(
            output=output,
            prompt=prompt,
            validation_results=validation_results,
            passed=passed,
            message=message,
            metadata=metadata or {},
            score=score,
            issues=issues or [],
            suggestions=suggestions or [],
            execution_time=execution_time,
            attempt_count=attempt_count,
            processing_time_ms=processing_time_ms or 0.0,
        )


def create_retrieval_result(
    query: str,
    documents: List[Any],
    processed_query: Optional[str] = None,
    total_results: Optional[int] = None,
    passed: bool = True,
    message: str = "Retrieval completed",
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 1.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> RetrievalResult:
    """
    Create a standardized retrieval result.

    Args:
        query: The query used for retrieval
        documents: The retrieved documents
        processed_query: The processed query (after preprocessing)
        total_results: The total number of results found
        passed: Whether the retrieval passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized RetrievalResult
    """
    if total_results is None:
        total_results = len(documents)

    return RetrievalResult(
        query=query,
        documents=documents,
        processed_query=processed_query,
        total_results=total_results,
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_model_result(
    output: OutputType,
    passed: bool = True,
    message: str = "Model operation completed",
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 1.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> ModelResult[OutputType]:
    """
    Create a standardized model result.

    Args:
        output: The generated output
        passed: Whether the model operation passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized ModelResult
    """
    return ModelResult(
        output=output,
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_generation_result(
    text: str,
    model_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    passed: bool = True,
    message: str = "Generation completed",
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 1.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
) -> GenerationResult:
    """
    Create a standardized generation result.

    Args:
        text: Generated text
        model_name: Name of the model
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total number of tokens
        passed: Whether the generation passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized GenerationResult
    """
    return GenerationResult(
        output=text,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def create_error_result(
    message: str,
    component_name: Optional[str] = None,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "error",
) -> Union[RuleResult, ErrorResult]:
    """
    Create a standardized error result.

    Args:
        message: Human-readable error message
        component_name: Name of the component that created the result
        error_type: Type of error
        metadata: Additional result metadata
        severity: Severity level of the error

    Returns:
        Standardized ErrorResult or RuleResult for error condition
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {"error": True, "severity": severity}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use ErrorResult if error_type is provided
    if error_type:
        return ErrorResult(
            error_type=error_type,
            error_message=message,
            passed=False,
            message=message,
            metadata=final_metadata,
            score=0.0,
        )

    # Otherwise use RuleResult
    return RuleResult(
        passed=False,
        message=message,
        metadata=final_metadata,
        severity=severity,
        score=0.0,
    )


def merge_metadata(*metadata_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries.

    Args:
        *metadata_dicts: Metadata dictionaries to merge

    Returns:
        Merged metadata dictionary
    """
    result: Dict[str, Any] = {}

    # Process metadata dictionaries in order
    for metadata in metadata_dicts:
        if metadata:
            result.update(metadata)

    return result


# Export all public classes and functions
__all__ = [
    # Classes
    "BaseResult",
    "RuleResult",
    "ClassificationResult",
    "CriticResult",
    "ValidationResult",
    "ModelResult",
    "GenerationResult",
    "ErrorResult",
    "ChainResult",
    "RetrievalResult",
    # Functions
    "create_base_result",
    "create_rule_result",
    "create_classification_result",
    "create_critic_result",
    "create_chain_result",
    "create_retrieval_result",
    "create_model_result",
    "create_generation_result",
    "create_error_result",
    "merge_metadata",
]
