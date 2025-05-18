"""Validation components for ensuring LLM outputs meet requirements.

This module provides validators for checking if text meets specific criteria,
such as length, content, and format requirements. It includes both built-in
validators and utilities for creating custom validators.

The validators can be used with the Chain class to validate text generated
by language models before returning it to the user.

Example:
    ```python
    from sifaka import Chain
    from sifaka.validators import length, prohibited_content
    from sifaka.models.openai import OpenAIModel

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create validators
    length_validator = length(min_words=100, max_words=500)
    content_validator = prohibited_content(prohibited=["harmful", "offensive"])

    # Create a chain with validators
    chain = (Chain()
        .with_model(model)
        .with_prompt("Write a short story about a robot.")
        .validate_with(length_validator)
        .validate_with(content_validator)
    )

    # Run the chain
    result = chain.run()

    # Check if validation passed
    if result.passed:
        print("Validation passed!")
        print(result.text)
    else:
        print("Validation failed!")
        print(result.validation_results[0].message)
    ```
"""

from sifaka.validators.base import BaseValidator, ValidatorProtocol, safe_validate
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
from sifaka.validators.classifier import (
    ClassifierValidator,
    classifier_validator,
    create_classifier_validator,
)
from sifaka.validators.guardrails import (
    GuardrailsValidator,
    guardrails_validator,
    create_guardrails_validator,
)

# Critics have been moved to their own directory

__all__ = [
    # Base validator
    "BaseValidator",
    "ValidatorProtocol",
    "safe_validate",
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
    # Classifier validators
    "ClassifierValidator",
    "classifier_validator",
    "create_classifier_validator",
    # GuardrailsAI validators
    "GuardrailsValidator",
    "guardrails_validator",
    "create_guardrails_validator",
]
