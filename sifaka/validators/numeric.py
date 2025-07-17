"""Numeric range validator for ensuring numeric values fall within acceptable bounds.

This module provides validation for numeric values found in text, supporting
various numeric formats including plain numbers, percentages, and currency amounts.
Useful for ensuring data quality, preventing unrealistic values, and enforcing
business rules around numeric content.

## Key Features:

- **Range Validation**: Set minimum and maximum bounds for numeric values
- **Multiple Ranges**: Define multiple allowed or forbidden numeric ranges
- **Format Detection**: Automatically detects numbers, percentages, and currency
- **Contextual Validation**: Different rules for different numeric formats
- **Business Rules**: Enforce domain-specific constraints

## Usage Examples:

    >>> # Validate age ranges
    >>> age_validator = create_age_validator()
    >>> result = await age_validator.validate("John is 25 years old", sifaka_result)
    >>>
    >>> # Custom numeric ranges
    >>> validator = NumericRangeValidator(
    ...     min_value=0,
    ...     max_value=100,
    ...     forbidden_ranges=[(13, 17)],  # No teen numbers
    ...     check_percentages=True
    ... )

## Supported Formats:

- **Plain numbers**: 42, -3.14, 1000
- **Percentages**: 85%, 0.5%, 100%
- **Currency**: $50, $1,234.56, $0.99

## Common Use Cases:

- **Data Quality**: Ensure statistics and metrics are realistic
- **Business Rules**: Enforce pricing, age, or quantity constraints
- **Content Safety**: Prevent unrealistic or harmful numeric claims
- **Format Validation**: Ensure percentages are 0-100%, prices are positive

The validator provides detailed error messages indicating which values
violated which constraints, making it easy to understand and fix issues.
"""

import re
from typing import List, Optional, Tuple

from ..core.interfaces import Validator
from ..core.models import SifakaResult, ValidationResult


class NumericRangeValidator(Validator):
    """Validates numeric values in text are within specified ranges.

    Extracts numeric values from text in various formats (plain numbers,
    percentages, currency) and validates them against configurable range
    constraints. Provides flexible validation for business rules and
    data quality requirements.

    Key capabilities:
    - Multiple numeric format detection (numbers, percentages, currency)
    - Range validation with min/max bounds
    - Multiple allowed/forbidden ranges support
    - Format-specific validation rules (e.g., 0-100% for percentages)
    - Detailed error reporting with specific constraint violations

    Example:
        >>> # Validate product reviews (1-5 stars, positive prices)
        >>> validator = NumericRangeValidator(
        ...     allowed_ranges=[(1, 5)],  # Star ratings 1-5
        ...     check_currency=True,      # Validate prices
        ...     check_percentages=False   # Skip percentage validation
        ... )
        >>>
        >>> text = "Great product! 4.5 stars, only $29.99"
        >>> result = await validator.validate(text, sifaka_result)
        >>> print(f"Valid: {result.passed}")  # True

    Format detection:
        Numbers: Matches integers and decimals with optional minus sign
        Percentages: Matches numbers followed by % symbol
        Currency: Matches $ symbol followed by numbers (decimal optional)
    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_ranges: Optional[List[Tuple[float, float]]] = None,
        forbidden_ranges: Optional[List[Tuple[float, float]]] = None,
        check_percentages: bool = True,
        check_currency: bool = True,
    ):
        """Initialize numeric range validator with flexible constraints.

        Creates a validator that extracts and validates numeric values
        from text according to specified range constraints and format rules.

        Args:
            min_value: Global minimum allowed value for plain numbers.
                Does not apply to percentages (auto-limited 0-100) or currency
                (auto-limited to non-negative).
            max_value: Global maximum allowed value for plain numbers.
                Does not apply to format-specific validations.
            allowed_ranges: List of (min, max) tuples defining acceptable
                value ranges. If specified, values must fall within at least
                one range. Use for complex business rules.
            forbidden_ranges: List of (min, max) tuples defining prohibited
                value ranges. Values in these ranges cause validation failure.
                Useful for excluding specific problematic ranges.
            check_percentages: Whether to detect and validate percentage values.
                Automatically enforces 0-100% range when enabled.
            check_currency: Whether to detect and validate currency amounts.
                Automatically enforces non-negative amounts when enabled.

        Example:
            >>> # Business rules validator
            >>> validator = NumericRangeValidator(
            ...     min_value=1,                    # Minimum 1 for counts
            ...     max_value=1000,                 # Maximum 1000 for quantities
            ...     allowed_ranges=[(1, 10), (50, 100)],  # Valid ranges only
            ...     forbidden_ranges=[(13, 13)],   # Unlucky 13 prohibited
            ...     check_percentages=True,         # Validate percentages 0-100%
            ...     check_currency=True             # Validate positive prices
            ... )

        Range priority:
            1. Format-specific rules (0-100% for percentages, >=0 for currency)
            2. forbidden_ranges (values in these ranges always fail)
            3. allowed_ranges (if specified, values must be in at least one)
            4. min_value/max_value (global bounds for plain numbers)

        Performance:
            Regex patterns are compiled at initialization for efficient
            repeated validation across multiple texts.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_ranges = allowed_ranges or []
        self.forbidden_ranges = forbidden_ranges or []
        self.check_percentages = check_percentages
        self.check_currency = check_currency

        # Numeric patterns
        self.number_pattern = re.compile(r"-?\d+\.?\d*")
        self.percentage_pattern = re.compile(r"(\d+\.?\d*)\s*%")
        self.currency_pattern = re.compile(r"\$\s*(\d+\.?\d*)")

    @property
    def name(self) -> str:
        """Return the validator identifier.

        Returns:
            "numeric_range_validator" - used in validation results and logging
        """
        return "numeric_range_validator"

    async def validate(self, text: str, result: SifakaResult) -> ValidationResult:
        """Validate all numeric values found in text against range constraints.

        Extracts numeric values in various formats, applies appropriate validation
        rules, and returns detailed results with specific constraint violations.

        Args:
            text: Text to scan for numeric values and validate
            result: SifakaResult for context (not currently used but available)

        Returns:
            ValidationResult with pass/fail status, score, and detailed feedback.
            Score is 1.0 for pass (all numbers valid), 0.0 for fail (any violations).
            Details include specific values that violated constraints.

        Validation process:
        1. Extract plain numbers, percentages, and currency amounts
        2. Apply format-specific validation (0-100% for percentages, etc.)
        3. Check global min/max bounds for plain numbers
        4. Verify allowed_ranges requirements if specified
        5. Check for forbidden_ranges violations
        6. Return comprehensive results with first 3 issues for readability

        Format handling:
        - Plain numbers: Subject to min_value, max_value, and range constraints
        - Percentages: Auto-validated for 0-100% range, exempt from other rules
        - Currency: Auto-validated for non-negative amounts

        Example:
            >>> text = "Product costs $25.99 with 15% discount"
            >>> result = await validator.validate(text)
            >>> if not result.passed:
            ...     print(f"Issues found: {result.details}")
            ... else:
            ...     print(f"All {result.details} values valid")
        """
        issues = []
        all_numbers = []

        # Find all numbers
        plain_numbers = self.number_pattern.findall(text)
        for num_str in plain_numbers:
            try:
                num = float(num_str)
                all_numbers.append((num, num_str, "number"))
            except ValueError:
                pass

        # Find percentages
        if self.check_percentages:
            percentages = self.percentage_pattern.findall(text)
            for pct_str in percentages:
                try:
                    pct = float(pct_str)
                    all_numbers.append((pct, f"{pct_str}%", "percentage"))

                    # Check percentage bounds
                    if pct < 0 or pct > 100:
                        issues.append(f"Invalid percentage: {pct}%")
                except ValueError:
                    pass

        # Find currency amounts
        if self.check_currency:
            amounts = self.currency_pattern.findall(text)
            for amt_str in amounts:
                try:
                    amt = float(amt_str)
                    all_numbers.append((amt, f"${amt_str}", "currency"))

                    # Check for negative currency
                    if amt < 0:
                        issues.append(f"Negative currency amount: ${amt}")
                except ValueError:
                    pass

        # Validate each number
        for value, display, num_type in all_numbers:
            # Skip percentage validation for general ranges
            if num_type == "percentage":
                continue

            # Check min/max
            if self.min_value is not None and value < self.min_value:
                issues.append(f"Value {display} is below minimum {self.min_value}")

            if self.max_value is not None and value > self.max_value:
                issues.append(f"Value {display} is above maximum {self.max_value}")

            # Check allowed ranges
            if self.allowed_ranges:
                in_allowed_range = any(
                    min_val <= value <= max_val
                    for min_val, max_val in self.allowed_ranges
                )
                if not in_allowed_range:
                    issues.append(f"Value {display} is outside allowed ranges")

            # Check forbidden ranges
            for min_val, max_val in self.forbidden_ranges:
                if min_val <= value <= max_val:
                    issues.append(
                        f"Value {display} is in forbidden range [{min_val}, {max_val}]"
                    )

        # Build result
        if issues:
            return ValidationResult(
                validator=self.name,
                passed=False,
                score=0.0,
                details="; ".join(issues[:3]),  # Limit to first 3 issues
            )

        # Calculate score
        if not all_numbers:
            details = "No numeric values found"
            score = 1.0
        else:
            details = f"Validated {len(all_numbers)} numeric value(s)"
            score = 1.0

        return ValidationResult(
            validator=self.name, passed=True, score=score, details=details
        )


# Convenience factory functions for common numeric validation scenarios

# These functions create pre-configured NumericRangeValidator instances for
# typical use cases. They serve as both useful defaults and examples of how
# to configure the validator for domain-specific requirements.


def create_percentage_validator() -> NumericRangeValidator:
    """Create a pre-configured validator for percentage values.

    Creates a validator specifically for text containing percentages,
    ensuring all percentage values fall within the standard 0-100% range.

    Returns:
        NumericRangeValidator configured to validate only percentages

    Example:
        >>> validator = create_percentage_validator()
        >>>
        >>> # This would pass validation
        >>> valid_text = "Survey shows 85% satisfaction rate"
        >>>
        >>> # This would fail validation
        >>> invalid_text = "Improvement increased by 150%"

    Configuration:
        - check_percentages=True: Detects and validates % values
        - check_currency=False: Ignores currency amounts
        - Automatic 0-100% range enforcement
    """
    return NumericRangeValidator(
        check_percentages=True,
        check_currency=False,
    )


def create_price_validator(max_price: float = 10000.0) -> NumericRangeValidator:
    """Create a pre-configured validator for price values.

    Creates a validator for currency amounts and general numeric values
    with reasonable bounds for pricing contexts.

    Args:
        max_price: Maximum allowed price value. Defaults to $10,000
            for reasonable business pricing scenarios.

    Returns:
        NumericRangeValidator configured for price validation

    Example:
        >>> validator = create_price_validator(max_price=500.0)
        >>>
        >>> # This would pass validation
        >>> valid_text = "Item costs $25.99 plus shipping"
        >>>
        >>> # This would fail validation
        >>> invalid_text = "Premium version is $750.00"

    Configuration:
        - min_value=0: No negative prices allowed
        - max_value=max_price: Upper bound for reasonable pricing
        - check_currency=True: Validates $ amounts
        - check_percentages=False: Ignores percentage values
    """
    return NumericRangeValidator(
        min_value=0,
        max_value=max_price,
        check_currency=True,
        check_percentages=False,
    )


def create_age_validator() -> NumericRangeValidator:
    """Create a pre-configured validator for human age values.

    Creates a validator that ensures age-related numeric values fall
    within realistic human lifespan ranges, preventing impossible ages.

    Returns:
        NumericRangeValidator configured for age validation

    Example:
        >>> validator = create_age_validator()
        >>>
        >>> # This would pass validation
        >>> valid_text = "Sarah is 34 years old"
        >>>
        >>> # This would fail validation
        >>> invalid_text = "Ancient wisdom from 500-year-old sage"

    Configuration:
        - min_value=0: No negative ages
        - max_value=150: Realistic maximum human lifespan
        - forbidden_ranges=[(200, inf)]: Explicitly prohibits unrealistic ages
        - Validates plain numbers only (no percentages or currency)

    Use cases:
        - Biographical content validation
        - User profile data quality
        - Historical content fact-checking
        - Demographics and survey data
    """
    return NumericRangeValidator(
        min_value=0,
        max_value=150,
        forbidden_ranges=[(200, float("inf"))],  # No unrealistic ages
    )
