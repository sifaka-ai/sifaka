"""
Validation module for Sifaka.

This module is deprecated. Please use sifaka.core.validation instead.

Example:
    ```python
    from sifaka.core.validation import Validator, ValidationResult, ValidatorConfig
    from sifaka.rules import create_length_rule

    # Create rules
    rules = [create_length_rule(min_chars=10)]

    # Create validator
    validator = Validator(rules)

    # Validate output
    result = validator.validate("Short text")
    if not result.all_passed:
        errors = validator.get_error_messages(result)
        print(f"Validation failed: {errors}")
    ```
"""

from sifaka.core.validation import ValidationResult, Validator, ValidatorConfig

# Export public classes
__all__ = [
    "ValidationResult",
    "Validator",
    "ValidatorConfig",
]
