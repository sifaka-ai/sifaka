from typing import Optional, Tuple, Dict, Any
from sifaka.rules.base import Rule, RuleResult
from pydantic import Field


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

    min_length: Optional[int] = Field(
        default=None, description="Minimum allowed length (inclusive)"
    )
    max_length: Optional[int] = Field(
        default=None, description="Maximum allowed length (inclusive)"
    )
    exact_length: Optional[int] = Field(default=None, description="Exact required length")

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with length constraints.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - min_length: Minimum allowed length (inclusive)
                   - max_length: Maximum allowed length (inclusive)
                   - exact_length: Exact required length
            **kwargs: Additional arguments

        Raises:
            ValueError: If length constraints are invalid, specifically:
                       - exact_length is negative
                       - exact_length is used with min_length or max_length
                       - min_length is greater than max_length
                       - min_length or max_length is negative
        """
        # First initialize the base class
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract length constraints from config
        config = config or {}
        min_length = config.get("min_length")
        max_length = config.get("max_length")
        exact_length = config.get("exact_length")

        # Validate constraints
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

        # Set the values using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "min_length", min_length)
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "exact_length", exact_length)

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
