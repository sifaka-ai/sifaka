"""Result types for Sifaka operations.

This module defines the result classes returned by various Sifaka components,
including validation results, improvement results, and the overall chain result.
These classes provide structured access to operation outcomes and additional
information about the operations.

The module includes three main result classes:
- ValidationResult: Returned by validators to indicate whether text meets criteria
- ImprovementResult: Returned by critics to provide improved text and details
- Result: Returned by the Chain class to provide the final outcome of execution

Example:
    ```python
    from sifaka import Chain
    from sifaka.validators import length
    from sifaka.critics.reflexion import create_reflexion_critic

    # Create and run a chain
    result = (Chain()
        .with_model("openai:gpt-4")
        .with_prompt("Write a short story about a robot.")
        .validate_with(length(min_words=50, max_words=500))
        .improve_with(create_reflexion_critic(model="openai:gpt-4"))
        .run())

    # Access result properties
    if result.passed:
        print(f"Chain execution succeeded in {result.execution_time_ms:.2f}ms")
        print(f"Final text: {result.text}")

        # Access improvement details
        for i, improvement in enumerate(result.improvement_results):
            print(f"Improvement {i+1}:")
            print(f"- Original length: {len(improvement.original_text)}")
            print(f"- Improved length: {len(improvement.improved_text)}")
            print(f"- Changes made: {improvement.changes_made}")
    else:
        print("Chain execution failed validation:")
        for issue in result.all_issues:
            print(f"- {issue}")
        for suggestion in result.all_suggestions:
            print(f"- Suggestion: {suggestion}")
    ```
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from sifaka.interfaces import ImprovementResult as ImprovementResultProtocol


@dataclass
class ValidationResult:
    """Result of a validation operation on text.

    This class represents the outcome of validating text against specific criteria.
    It includes information about whether the validation passed, any issues found,
    and suggestions for improvement.

    Validators return instances of this class to indicate whether text meets
    their criteria and to provide details about any validation failures.

    Attributes:
        passed (bool): Whether the validation passed (True) or failed (False).
        message (str): Human-readable message describing the validation result.
        _details (Dict[str, Any]): Internal storage for additional details about the validation.
        score (Optional[float]): Normalized score between 0.0 and 1.0, if applicable.
        issues (Optional[List[str]]): List of identified issues if validation failed.
        suggestions (Optional[List[str]]): List of suggestions for addressing the issues.

    Example:
        ```python
        # Creating a validation result
        result = ValidationResult(
            passed=False,
            message="Text is too short",
            score=0.3,
            issues=["Text has only 30 words, minimum required is 50"],
            suggestions=["Add at least 20 more words to meet the minimum length requirement"]
        )

        # Using the result
        if not result.passed:
            print(f"Validation failed: {result.message}")
            for issue in result.issues:
                print(f"- {issue}")
            for suggestion in result.suggestions:
                print(f"- Suggestion: {suggestion}")
        ```
    """

    passed: bool
    message: str = ""
    _details: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context.

        This method enables using ValidationResult objects in boolean expressions,
        where the result evaluates to True if validation passed and False if it failed.

        Returns:
            bool: The value of the passed attribute.

        Example:
            ```python
            result = validator.validate(text)
            if result:  # Equivalent to if result.passed
                print("Validation passed")
            else:
                print("Validation failed")
            ```
        """
        return self.passed

    @property
    def details(self) -> Dict[str, Any]:
        """Get additional details about the validation result.

        This property provides access to additional details about the validation
        that may be useful for debugging or advanced processing. The content of
        the details dictionary varies depending on the validator implementation.

        Returns:
            Dict[str, Any]: A dictionary containing additional details about the
                validation result.
        """
        return self._details


@dataclass
class ImprovementResult(ImprovementResultProtocol):
    """Result of an improvement operation on text.

    This class represents the outcome of improving text using a critic or other
    improvement mechanism. It includes both the original and improved text,
    information about whether changes were made, and additional details about
    the improvement process.

    Critics return instances of this class to provide the improved text and
    details about the improvement process.

    This class implements the ImprovementResultProtocol, which defines the
    interface that all improvement results must follow.

    Attributes:
        _original_text (str): The original text before improvement.
        _improved_text (str): The improved text after improvement.
        _changes_made (bool): Whether any changes were made to the text.
        message (str): Human-readable message describing the improvement.
        _details (Dict[str, Any]): Internal storage for additional details about the improvement.
        processing_time_ms (Optional[float]): Processing time in milliseconds, if available.

    Example:
        ```python
        # Creating an improvement result
        result = ImprovementResult(
            _original_text="This is the original text.",
            _improved_text="This is the improved text with better clarity.",
            _changes_made=True,
            message="Improved text clarity",
            processing_time_ms=123.45
        )

        # Using the result
        if result.changes_made:
            print(f"Text was improved: {result.message}")
            print(f"Original: {result.original_text}")
            print(f"Improved: {result.improved_text}")
            print(f"Processing time: {result.processing_time_ms:.2f}ms")
        else:
            print("No changes were made to the text")
        ```
    """

    _original_text: str
    _improved_text: str
    _changes_made: bool
    message: str = ""
    _details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context.

        This method enables using ImprovementResult objects in boolean expressions,
        where the result evaluates to True if changes were made to the text and
        False if no changes were made.

        Returns:
            bool: The value of the _changes_made attribute.

        Example:
            ```python
            text, result = critic.improve(text)
            if result:  # Equivalent to if result.changes_made
                print("Text was improved")
            else:
                print("No changes were made to the text")
            ```
        """
        return self._changes_made

    @property
    def passed(self) -> bool:
        """Indicate whether the improvement operation completed successfully.

        This property always returns True for improvement operations, as they are
        considered to "pass" in the sense that they complete their task, regardless
        of whether they actually made changes to the text.

        This property is required by the ImprovementResultProtocol to maintain
        consistency with the ValidationResult interface.

        Returns:
            bool: Always True for improvement operations.
        """
        return True  # Improvement operations always "pass" in the sense that they complete

    @property
    def details(self) -> Dict[str, Any]:
        """Get additional details about the improvement process.

        This property provides access to additional details about the improvement
        that may be useful for debugging or advanced processing. The content of
        the details dictionary varies depending on the critic implementation.

        Returns:
            Dict[str, Any]: A dictionary containing additional details about the
                improvement process.
        """
        return self._details

    @property
    def original_text(self) -> str:
        """Get the original text before improvement.

        This property provides access to the original text that was passed to
        the improve method, allowing comparison with the improved text.

        Returns:
            str: The original text before improvement.
        """
        return self._original_text

    @property
    def improved_text(self) -> str:
        """Get the improved text after improvement.

        This property provides access to the improved text that was generated
        by the critic. If no changes were made, this will be identical to the
        original text.

        Returns:
            str: The improved text after improvement.
        """
        return self._improved_text

    @property
    def changes_made(self) -> bool:
        """Indicate whether any changes were made to the text.

        This property indicates whether the critic actually made any changes
        to the text during the improvement process. It's useful for determining
        whether the improvement had any effect.

        Returns:
            bool: True if changes were made to the text, False otherwise.
        """
        return self._changes_made


@dataclass
class Result:
    """Result of a chain execution with validation and improvement details.

    This class represents the complete outcome of executing a Chain, including
    the final text, validation status, and detailed results from all validators
    and improvers that were applied.

    The Chain.run() method returns an instance of this class, providing a
    comprehensive view of the execution process and its outcome.

    The result can be used in boolean context to check if all validations passed:
    ```python
    result = chain.run()
    if result:  # Equivalent to if result.passed
        print("All validations passed")
    ```

    Attributes:
        text (str): The final text after all validations and improvements.
        passed (bool): Whether all validations passed (True) or at least one failed (False).
        validation_results (List[ValidationResult]): Results of all validations that were performed.
        improvement_results (List[ImprovementResult]): Results of all improvements that were applied.
        metadata (Dict[str, Any]): Additional metadata about the execution.
        execution_time_ms (Optional[float]): Total execution time in milliseconds, if available.

    Example:
        ```python
        from sifaka import Chain
        from sifaka.validators import length, prohibited_content

        # Create and run a chain
        result = (Chain()
            .with_model("openai:gpt-4")
            .with_prompt("Write a short story about a robot.")
            .validate_with(length(min_words=50, max_words=500))
            .validate_with(prohibited_content(prohibited=["violent", "harmful"]))
            .run())

        # Check if all validations passed
        if result.passed:
            print(f"Chain execution succeeded in {result.execution_time_ms:.2f}ms")
            print(f"Final text ({len(result.text)} chars):")
            print(result.text)
        else:
            # Find which validation failed
            failed_validation = next(v for v in result.validation_results if not v.passed)
            print(f"Validation failed: {failed_validation.message}")

            # Show all issues and suggestions
            for issue in result.all_issues:
                print(f"- Issue: {issue}")
            for suggestion in result.all_suggestions:
                print(f"- Suggestion: {suggestion}")
        ```
    """

    text: str
    passed: bool
    validation_results: List[ValidationResult]
    improvement_results: List[ImprovementResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None

    def __bool__(self) -> bool:
        """Allow using the result in boolean context.

        This method enables using Result objects in boolean expressions,
        where the result evaluates to True if all validations passed and
        False if any validation failed.

        Returns:
            bool: The value of the passed attribute.

        Example:
            ```python
            result = chain.run()
            if result:  # Equivalent to if result.passed
                print("Chain execution succeeded")
            else:
                print("Chain execution failed validation")
            ```
        """
        return self.passed

    @property
    def has_issues(self) -> bool:
        """Check if any validation has reported issues.

        This property checks all validation results to see if any of them
        have reported issues. It's useful for quickly determining if there
        are any specific issues that need to be addressed, even if the
        overall validation passed.

        Returns:
            bool: True if any validation has reported issues, False otherwise.
        """
        for result in self.validation_results:
            if result.issues and len(result.issues) > 0:
                return True
        return False

    @property
    def all_issues(self) -> List[str]:
        """Get all issues from all validations as a single list.

        This property collects all issues reported by all validations into
        a single list, making it easy to display or process all issues at once.

        Returns:
            List[str]: A list of all issues from all validations. If no issues
                were reported, returns an empty list.
        """
        issues = []
        for result in self.validation_results:
            if result.issues:
                issues.extend(result.issues)
        return issues

    @property
    def all_suggestions(self) -> List[str]:
        """Get all suggestions from all validations as a single list.

        This property collects all suggestions provided by all validations into
        a single list, making it easy to display or process all suggestions at once.

        Returns:
            List[str]: A list of all suggestions from all validations. If no
                suggestions were provided, returns an empty list.
        """
        suggestions = []
        for result in self.validation_results:
            if result.suggestions:
                suggestions.extend(result.suggestions)
        return suggestions
