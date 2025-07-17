"""Validation component for enforcing text quality requirements.

This module manages the execution of validators during the text improvement
process. Validators act as quality gates, ensuring that generated text meets
specific requirements before proceeding or completing.

## Role in the Engine:

Validators serve three key purposes:
1. **Quality Gates**: Prevent low-quality text from being accepted
2. **Requirement Enforcement**: Ensure business rules are met
3. **Feedback Generation**: Provide specific guidance for improvement

## Validation Flow:

1. Validators run after each text generation
2. Failed validations are formatted as requirements for the generator
3. The improvement loop continues until all validators pass
4. Results are tracked for analysis and debugging

## Design Principles:

- **Fail-Safe**: Errors in validators don't crash the process
- **Traceable**: All validation results are recorded
- **Memory-Bounded**: Old results are pruned to prevent growth
- **Extensible**: Easy to add custom validators

## Usage:

    >>> runner = ValidationRunner()
    >>> validators = [LengthValidator(min_length=100)]
    >>>
    >>> all_passed = await runner.run_validators(
    ...     text="Current text",
    ...     result=sifaka_result,
    ...     validators=validators
    ... )
    >>>
    >>> if not all_passed:
    ...     print("Text needs improvement")

## Error Handling:

Validator exceptions are caught and converted to failed validations,
ensuring the improvement process continues even if a validator fails.
"""

from typing import List

from ..interfaces import Validator
from ..models import SifakaResult


class ValidationRunner:
    """Orchestrates validator execution and result collection.

    The ValidationRunner is responsible for executing all configured
    validators against the current text and collecting their results.
    It provides error handling, result tracking, and memory management.

    Key responsibilities:
    - Execute validators in sequence
    - Handle validator errors gracefully
    - Track validation results in the SifakaResult
    - Manage memory bounds for long-running sessions
    - Provide clear pass/fail status

    Example:
        >>> runner = ValidationRunner()
        >>>
        >>> # Run multiple validators
        >>> validators = [
        ...     LengthValidator(min_length=50, max_length=500),
        ...     ContentValidator(required_terms=["AI", "benefits"]),
        ...     FormatValidator(min_sentences=3)
        ... ]
        >>>
        >>> passed = await runner.run_validators(
        ...     text="Improved text here...",
        ...     result=result,
        ...     validators=validators
        ... )
        >>>
        >>> # Check specific validation results
        >>> for val in result.validations:
        ...     if not val.passed:
        ...         print(f"{val.validator}: {val.details}")

    The runner ensures that even if individual validators fail,
    the overall process continues with appropriate error tracking.
    """

    async def run_validators(
        self, text: str, result: SifakaResult, validators: List[Validator]
    ) -> bool:
        """Execute all validators and collect their results.

        Runs each validator in sequence, collecting results and tracking
        overall pass/fail status. Validator errors are caught and converted
        to failed validations to ensure process continuity.

        Args:
            text: The current text to validate. This is typically either
                the original input or an improved version from the generator.
            result: SifakaResult object where validation results are stored.
                The runner adds results using result.add_validation().
            validators: List of Validator instances to execute. Each validator
                checks a specific quality criterion. Empty list is allowed
                and returns True.

        Returns:
            True if all validators pass, False if any validator fails
            or encounters an error. This determines whether improvement
            should continue.

        Error Handling:
            If a validator raises an exception, it's caught and recorded
            as a failed validation with the error message. This ensures
            one faulty validator doesn't break the entire process.

        Example:
            >>> # All validators pass
            >>> passed = await runner.run_validators(
            ...     "This text meets all requirements.",
            ...     result,
            ...     [length_validator, content_validator]
            ... )
            >>> assert passed == True
            >>>
            >>> # One validator fails
            >>> passed = await runner.run_validators(
            ...     "Too short",
            ...     result,
            ...     [LengthValidator(min_length=100)]
            ... )
            >>> assert passed == False
        """
        if not validators:
            return True

        all_passed = True

        for validator in validators:
            try:
                validation_result = await validator.validate(text, result)

                # Add to result
                result.add_validation(
                    validator=validation_result.validator,
                    passed=validation_result.passed,
                    score=validation_result.score,
                    details=validation_result.details,
                )

                if not validation_result.passed:
                    all_passed = False

            except Exception as e:
                # Create error validation
                result.add_validation(
                    validator=getattr(validator, "name", "unknown"),
                    passed=False,
                    score=0.0,
                    details=f"Validation error: {e!s}",
                )
                all_passed = False

        return all_passed

    def check_memory_bounds(
        self, result: SifakaResult, max_elements: int = 1000
    ) -> None:
        """Enforce memory limits on result collections.

        For long-running improvement sessions, the collections of
        generations, critiques, and validations can grow without bound.
        This method ensures they stay within reasonable limits by
        trimming old entries.

        Args:
            result: The SifakaResult to check and potentially trim.
                Collections are modified in-place if they exceed limits.
            max_elements: Maximum number of elements to keep in each
                collection (generations, critiques, validations).
                Defaults to 1000 which provides good history while
                preventing excessive memory use.

        Memory Management:
            When a collection exceeds max_elements, the oldest entries
            are removed to bring it back to the limit. This maintains
            the most recent and relevant history.

        Note:
            This method is typically called periodically by the engine
            during long improvement sessions. Most users don't need to
            call it directly.

        Example:
            >>> # After many iterations
            >>> print(f"Critiques before: {len(result.critiques)}")
            >>> runner.check_memory_bounds(result, max_elements=100)
            >>> print(f"Critiques after: {len(result.critiques)}")  # <= 100
        """
        # Trim old generations if needed
        if len(result.generations) > max_elements:
            # Convert to list, slice, and recreate deque
            from collections import deque

            result.generations = deque(
                list(result.generations)[-max_elements:], maxlen=max_elements
            )

        # Trim old critiques if needed
        if len(result.critiques) > max_elements:
            from collections import deque

            result.critiques = deque(
                list(result.critiques)[-max_elements:], maxlen=max_elements
            )

        # Trim old validations if needed
        if len(result.validations) > max_elements:
            from collections import deque

            result.validations = deque(
                list(result.validations)[-max_elements:], maxlen=max_elements
            )
