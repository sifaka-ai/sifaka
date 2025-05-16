"""
Validation components for ensuring LLM outputs meet requirements.
"""

from sifaka.validators.length import LengthValidator, length, create_length_validator
from sifaka.validators.content import ContentValidator, prohibited_content, create_content_validator
from sifaka.validators.format import (
    FormatValidator,
    json_format,
    markdown_format,
    custom_format,
    create_json_format_validator,
    create_markdown_format_validator,
    create_custom_format_validator,
)
from sifaka.validators.critics import (
    Critic,
    ClarityAndCoherenceCritic,
    FactualAccuracyCritic,
)

__all__ = [
    # Basic validators
    "LengthValidator",
    "length",
    "create_length_validator",
    "ContentValidator",
    "prohibited_content",
    "create_content_validator",
    "FormatValidator",
    "json_format",
    "markdown_format",
    "custom_format",
    "create_json_format_validator",
    "create_markdown_format_validator",
    "create_custom_format_validator",
    # Critics
    "Critic",
    "ClarityAndCoherenceCritic",
    "FactualAccuracyCritic",
    "clarity",
    "factual_accuracy",
]


# Convenience functions for creating critics
def clarity(model: str = "openai:gpt-3.5-turbo", **options):
    """Create a clarity and coherence critic."""
    return ClarityAndCoherenceCritic(model, **options)


def factual_accuracy(model: str = "openai:gpt-3.5-turbo", **options):
    """Create a factual accuracy critic."""
    return FactualAccuracyCritic(model, **options)
