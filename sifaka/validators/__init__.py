"""Comprehensive validation system for ensuring text quality and compliance.

This package provides a rich ecosystem of validators that act as quality gates
during the text improvement process. Validators check if generated text meets
specific criteria and can prevent low-quality content from being accepted.

## Validation Philosophy:

Validators in Sifaka are designed to be:
- **Composable**: Combine multiple validators using logical operators
- **Extensible**: Easy to create custom validators for specific needs
- **Performance-focused**: Efficient validation with detailed feedback
- **Domain-agnostic**: Work across different content types and use cases

## Core Validator Categories:

### Basic Validators (`basic.py`)
**LengthValidator**: Character/word count validation
- Use for: Platform constraints, content guidelines, readability requirements
- Example: Blog posts (500-2000 chars), tweets (280 chars), abstracts (150 words)

**ContentValidator**: Required/forbidden terms validation
- Use for: Brand compliance, safety filters, topic coverage, content policies
- Example: Ensure "AI ethics" mentioned, forbid competitor names

**FormatValidator**: Text structure validation
- Use for: Document standards, consistent formatting, structural requirements
- Example: Require paragraphs, minimum sentences, proper line breaks

### Pattern Validators (`pattern.py`)
**PatternValidator**: Regex-based content validation
- Use for: Format validation, structural requirements, content standards
- Includes pre-built validators for code blocks, citations, structured documents

### Numeric Validators (`numeric.py`)
**NumericRangeValidator**: Numeric value constraint validation
- Use for: Data quality, business rules, realistic value enforcement
- Supports numbers, percentages, currency with range constraints

### Composable System (`composable.py`)
**ComposableValidator**: Fluent interface for complex validation logic
- Use for: Multi-criteria validation, conditional logic, readable validation rules
- Supports AND/OR/NOT operations with method chaining

### Advanced Integration (`guardrails.py`)
**GuardrailsValidator**: GuardrailsAI integration for AI safety
- Use for: Content moderation, PII detection, bias checking, safety compliance
- Access to 50+ pre-built validators from GuardrailsAI hub

## Usage Patterns:

    >>> # Simple validation
    >>> from sifaka.validators import LengthValidator, ContentValidator
    >>> validators = [
    ...     LengthValidator(min_length=100, max_length=500),
    ...     ContentValidator(required_terms=["AI", "benefits"])
    ... ]
    >>> result = await improve(text, validators=validators)

    >>> # Composable validation with logical operators
    >>> from sifaka.validators import Validator
    >>> length_ok = Validator.length(100, 500)
    >>> has_keywords = Validator.contains(["AI", "ML"])
    >>> combined = length_ok & has_keywords  # Both must pass
    >>>
    >>> # Complex validation builder
    >>> essay_validator = (Validator.create("essay")
    ...     .length(500, 2000)
    ...     .sentences(5, 25)
    ...     .contains(["thesis", "conclusion"])
    ...     .build())

    >>> # Safety and compliance validation
    >>> from sifaka.validators import GuardrailsValidator  # Requires guardrails-ai
    >>> safety_validator = GuardrailsValidator([
    ...     "toxic-language",
    ...     "detect-pii",
    ...     "profanity-free"
    ... ])

## Custom Validator Development:

    >>> from sifaka.core.interfaces import Validator
    >>> from sifaka.core.models import ValidationResult
    >>>
    >>> class SentimentValidator(Validator):
    ...     async def validate(self, text, result):
    ...         # Custom sentiment analysis logic
    ...         positive_score = analyze_sentiment(text)
    ...         passed = positive_score > 0.6
    ...         return ValidationResult(
    ...             validator="sentiment",
    ...             passed=passed,
    ...             score=positive_score,
    ...             details=f"Sentiment score: {positive_score}"
    ...         )

## Performance Guidelines:

- **Combine related checks**: Use ComposableValidator to group related validations
- **Order validators**: Place fast validators before slow ones for early termination
- **Cache expensive operations**: Store regex patterns, models, or API clients
- **Use appropriate granularity**: Balance validation thoroughness with performance needs

## Integration Tips:

Validators integrate seamlessly with the Sifaka improvement process:
1. Run after each text generation iteration
2. Failed validation triggers additional improvement cycles
3. Validation results are recorded in SifakaResult for analysis
4. Detailed feedback guides the improvement process

Choose validators based on your specific content requirements, performance
constraints, and quality standards.
"""

from .basic import ContentValidator, FormatValidator, LengthValidator
from .composable import ComposableValidator, Validator, ValidatorBuilder
from .numeric import (
    NumericRangeValidator,
    create_age_validator,
    create_percentage_validator,
    create_price_validator,
)
from .pattern import (
    PatternValidator,
    create_citation_validator,
    create_code_validator,
    create_structured_validator,
)

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
