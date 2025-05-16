"""
Validators for the Sifaka library.

This package provides validator classes that check if text meets specific criteria.
"""

from typing import Protocol, runtime_checkable, List

from ..types import ValidationResult


@runtime_checkable
class ValidatorProtocol(Protocol):
    """
    Protocol for validators that check if text meets specific criteria.

    Validators implement this protocol to check if text meets specific criteria such as
    length requirements, content guidelines, or style conventions.
    """

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against specific criteria.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with validation details
        """
        ...


# Import validator implementations
from .length import LengthValidator
from .content import ContentValidator
from .toxicity import ToxicityValidator
from .readability import ReadabilityValidator
from .grammar import GrammarValidator
from .similarity import SimilarityValidator
from .custom import CustomValidator
from .guardrails import GuardrailsValidator
from .sentiment import SentimentValidator
from .spam import SpamValidator
from .bias import BiasValidator

# Export validators
__all__ = [
    "ValidatorProtocol",
    "LengthValidator",
    "ContentValidator",
    "ToxicityValidator",
    "ReadabilityValidator",
    "GrammarValidator",
    "SimilarityValidator",
    "CustomValidator",
    "GuardrailsValidator",
    "SentimentValidator",
    "SpamValidator",
    "BiasValidator",
]
