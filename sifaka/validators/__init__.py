"""Validators for text quality checking."""

from .basic import LengthValidator, ContentValidator, FormatValidator
from .pattern import (
    PatternValidator,
    create_code_validator,
    create_citation_validator,
    create_structured_validator,
)
from .numeric import (
    NumericRangeValidator,
    create_percentage_validator,
    create_price_validator,
    create_age_validator,
)
from .composable import Validator, ComposableValidator, ValidatorBuilder

# Core validators always available
__all__ = [
    # Basic validators
    "LengthValidator",
    "ContentValidator",
    "FormatValidator",
    # Pattern and numeric validators
    "PatternValidator",
    "NumericRangeValidator",
    # Factory functions
    "create_code_validator",
    "create_citation_validator",
    "create_structured_validator",
    "create_percentage_validator",
    "create_price_validator",
    "create_age_validator",
    # Composable validators
    "Validator",
    "ComposableValidator",
    "ValidatorBuilder",
]

# Optional GuardrailsAI validator
try:
    from .guardrails import GuardrailsValidator  # noqa: F401

    __all__.append("GuardrailsValidator")
except ImportError:
    # GuardrailsAI not installed, that's okay
    pass
