"""
Validation module for Sifaka.

This module provides components for validating outputs against rules. It includes:
- ValidationResult: A generic container for validation results
- Validator: A class that applies validation rules to outputs

The validation process follows these steps:
1. Initialize a Validator with a set of rules
2. Pass an output to the validate() method
3. Receive a ValidationResult containing:
   - The original output
   - Results for each rule
   - Overall validation status
4. Extract error messages if needed

Example:
    ```python
    from sifaka.validation import Validator
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

from .models import ValidationResult
from .validator import Validator, ValidatorConfig

# For backward compatibility
from .validator import Validator as BaseValidator

# Export public classes
__all__ = [
    "ValidationResult",
    "Validator",
    "ValidatorConfig",
    "BaseValidator",  # For backward compatibility
]
