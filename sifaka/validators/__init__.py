"""Validators for Sifaka.

This package provides validator classes that check if text meets specific criteria.
Validators are used in the Sifaka chain to ensure that generated text meets
specified requirements before it is returned to the user or passed to the next
stage of the chain.

Available validators:
- LengthValidator: Checks text length requirements
- RegexValidator: Validates text against regex patterns
- ContentValidator: Checks for prohibited content
- FormatValidator: Validates text format (JSON, Markdown, custom)
- ClassifierValidator: Uses ML classifiers for validation
- GuardrailsValidator: Integrates with GuardrailsAI

Example:
    ```python
    from sifaka.validators import (
        LengthValidator,
        ContentValidator,
        FormatValidator,
        ClassifierValidator,
        GuardrailsValidator,
    )
    from sifaka.core.thought import Thought

    # Create validators
    length_validator = LengthValidator(min_length=10, max_length=1000)
    content_validator = ContentValidator(prohibited=["spam", "harmful"])
    format_validator = FormatValidator(format_type="json")

    # Create a thought with text
    thought = Thought(prompt="Test prompt", text="This is test text.")

    # Validate
    length_result = length_validator.validate(thought)
    content_result = content_validator.validate(thought)
    format_result = format_validator.validate(thought)

    # Check results
    if length_result.passed and content_result.passed and format_result.passed:
        print("All validations passed!")
    ```
"""

# Import base validators
from sifaka.validators.base import LengthValidator, RegexValidator

# Import content validator
from sifaka.validators.content import (
    ContentValidator,
    create_content_validator,
    prohibited_content,
)

# Import format validator
from sifaka.validators.format import (
    FormatValidator,
    create_format_validator,
    json_format,
    markdown_format,
    custom_format,
)

# Import classifier validator
from sifaka.validators.classifier import (
    ClassifierValidator,
    Classifier,
    create_classifier_validator,
    classifier_validator,
)

# Import guardrails validator
from sifaka.validators.guardrails import (
    GuardrailsValidator,
    create_guardrails_validator,
    guardrails_validator,
)

__all__ = [
    # Base validators
    "LengthValidator",
    "RegexValidator",
    # Content validator
    "ContentValidator",
    "create_content_validator",
    "prohibited_content",
    # Format validator
    "FormatValidator",
    "create_format_validator",
    "json_format",
    "markdown_format",
    "custom_format",
    # Classifier validator
    "ClassifierValidator",
    "Classifier",
    "create_classifier_validator",
    "classifier_validator",
    # Guardrails validator
    "GuardrailsValidator",
    "create_guardrails_validator",
    "guardrails_validator",
]
