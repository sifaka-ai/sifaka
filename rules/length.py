from typing import Optional, Tuple
from sifaka.rules.base import Rule, RuleResult


class LengthRule(Rule):
    """
    Rule that checks if the output length falls within specified bounds.

    This rule is part of the Sifaka validation framework and implements length
    validation for text content. It supports three types of length constraints:
    - Minimum length: Ensures output is not too short
    - Maximum length: Ensures output is not too long
    - Exact length: Ensures output matches a specific length

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses a flexible constraint system supporting min/max/exact lengths
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on length validation
    - Includes helper methods for constraint validation and description generation

    Data Flow:
    1. User creates LengthRule with desired length constraints
    2. validate() method receives output text
    3. Length is calculated and checked against constraints
    4. Result is wrapped in RuleResult with relevant metadata
    5. RuleResult is returned to the caller
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exact_length: Optional[int] = None,
    ):
        """
        Initialize the rule with length constraints.

        Args:
            min_length: Minimum allowed length (inclusive).
                       If None, no minimum length is enforced.
            max_length: Maximum allowed length (inclusive).
                       If None, no maximum length is enforced.
            exact_length: Exact required length.
                        If specified, overrides min_length and max_length.

        Raises:
            ValueError: If length constraints are invalid, specifically:
                       - exact_length is negative
                       - exact_length is used with min_length or max_length
                       - min_length is greater than max_length
                       - min_length or max_length is negative
        """
        if exact_length is not None:
            if exact_length < 0:
                raise ValueError("exact_length must be non-negative")
            if min_length is not None or max_length is not None:
                raise ValueError("exact_length cannot be used with min_length or max_length")
        else:
            if min_length is not None and max_length is not None and min_length > max_length:
                raise ValueError("min_length cannot be greater than max_length")
            if min_length is not None and min_length < 0:
                raise ValueError("min_length must be non-negative")
            if max_length is not None and max_length < 0:
                raise ValueError("max_length must be non-negative")

        super().__init__(
            name="length_rule",
            description=self._generate_description(min_length, max_length, exact_length),
        )
        self.min_length = min_length
        self.max_length = max_length
        self.exact_length = exact_length

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output length meets the specified constraints.

        This method implements the core validation logic by:
        1. Calculating the output length
        2. Checking against the specified constraints
        3. Constructing a detailed result message
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating if length meets constraints
                       - message: Human-readable validation result
                       - metadata: Additional validation details including lengths
        """
        try:
            length = len(output)
            passed, message = self._check_length(length)

            return RuleResult(
                passed=passed,
                message=message,
                metadata={
                    "length": length,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "exact_length": self.exact_length,
                },
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during length validation: {str(e)}",
                metadata={
                    "error": str(e),
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "exact_length": self.exact_length,
                },
            )

    def _generate_description(
        self,
        min_length: Optional[int],
        max_length: Optional[int],
        exact_length: Optional[int],
    ) -> str:
        """
        Generate a human-readable description of the length constraints.

        Args:
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            exact_length: Exact required length

        Returns:
            str: A description of the length constraints
        """
        if exact_length is not None:
            return f"Checks if output length is exactly {exact_length} characters"

        constraints = []
        if min_length is not None:
            constraints.append(f"at least {min_length}")
        if max_length is not None:
            constraints.append(f"at most {max_length}")

        return f"Checks if output length is {' and '.join(constraints)} characters"

    def _check_length(self, length: int) -> Tuple[bool, str]:
        """
        Check if the length meets the specified constraints.

        Args:
            length: The length to check

        Returns:
            Tuple[bool, str]: A tuple containing:
                            - bool: Whether the length meets the constraints
                            - str: A message describing the result
        """
        if self.exact_length is not None:
            passed = length == self.exact_length
            message = (
                f"Output length {length} {'matches' if passed else 'does not match'} "
                f"required length of {self.exact_length}"
            )
            return passed, message

        passed = True
        message_parts = []

        if self.min_length is not None:
            min_passed = length >= self.min_length
            passed &= min_passed
            message_parts.append(
                f"{'meets' if min_passed else 'does not meet'} minimum length of {self.min_length}"
            )

        if self.max_length is not None:
            max_passed = length <= self.max_length
            passed &= max_passed
            message_parts.append(
                f"{'meets' if max_passed else 'does not meet'} maximum length of {self.max_length}"
            )

        message = f"Output length {length} {' and '.join(message_parts)}"
        return passed, message
