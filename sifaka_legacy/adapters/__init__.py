"""
Adapters

Adapters for external libraries and services integration with Sifaka.

## Overview
This package provides adapter implementations that allow Sifaka to integrate with
various external libraries, services, and model providers. It follows the adapter
pattern to enable loose coupling between Sifaka and external systems.

## Components
1. **Chain Adapters**: Adapters for chain components (models, validators, improvers, formatters)
2. **Classifier Adapters**: Adapters for text classification systems
3. **Guardrails Adapters**: Adapters for using Guardrails validators
4. **PydanticAI Adapters**: Adapters for integrating with PydanticAI agents

## Usage Examples

### Chain Adapters
```python
from sifaka.adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model_provider = OpenAIProvider("gpt-3.5-turbo")
rule = create_length_rule(min_chars=10, max_chars=1000)
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create adapters
model = ModelAdapter(model_provider)
validator = ValidatorAdapter(rule)
improver = ImproverAdapter(critic)

# Use adapters
output = model.generate("Write a short story")
validation_result = validator.validate(output)
if not validation_result.passed:
    improved_output = improver.improve(output, [validation_result])
```

### Classifier Adapters
```python
from sifaka.adapters import ClassifierAdapter, create_classifier_rule
from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier

# Create a classifier
classifier = ToxicityClassifier()

# Create a rule from the classifier
rule = create_classifier_rule(
    classifier=classifier,
    valid_labels=["non_toxic"],
    threshold=0.8,
    name="toxicity_rule",
    description="Validates that text is not toxic"
)

# Use the rule for validation
result = rule.validate("This is a friendly message")
```

## Error Handling
- ImportError: Raised when optional dependencies are not available
- ConfigurationError: Raised when adapter configuration is invalid
- ValidationError: Raised when validation fails

## Configuration
Each adapter type has its own configuration options:
- Classifier adapters: Configure with classifier instance and validation parameters
- Guardrails adapters: Configure with validator instance and validation rules
- PydanticAI adapters: Configure with model and validation settings
"""

# Base adapter components
from sifaka.adapters.base import Adaptable, BaseAdapter, AdapterError, create_adapter

# Chain adapters
from sifaka.adapters.chain import (
    ModelAdapter,
    ValidatorAdapter,
    ImproverAdapter,
    FormatterAdapter,
)

# Classifier adapters
from sifaka.adapters.classifier import (
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_adapter,
    create_classifier_rule,
)

# Export types from sifaka.core.results for convenience
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig

# Try to import Guardrails adapters if available
try:
    from sifaka.adapters.guardrails import (
        GuardrailsAdapter,
        GuardrailsValidatorAdapter,
        GuardrailsRule,
        create_guardrails_adapter,
        create_guardrails_rule,
    )

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

# Try to import PydanticAI adapters if available
try:
    from sifaka.adapters.pydantic_ai import (
        SifakaPydanticAdapter,
        SifakaPydanticConfig,
        create_pydantic_adapter,
        create_pydantic_adapter_with_critic,
    )

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

__all__ = [
    # Base adapters
    "Adaptable",
    "BaseAdapter",
    "AdapterError",
    "create_adapter",
    # Chain adapters
    "ModelAdapter",
    "ValidatorAdapter",
    "ImproverAdapter",
    "FormatterAdapter",
    # Classifier adapters
    "ClassifierAdapter",
    "ClassifierRule",
    "create_classifier_adapter",
    "create_classifier_rule",
    # Classifier types
    "ClassificationResult",
    "ClassifierConfig",
]

# Add Guardrails adapter to exports if available
if GUARDRAILS_AVAILABLE:
    __all__.extend(
        [
            "GuardrailsAdapter",  # New standardized adapter
            "GuardrailsValidatorAdapter",  # Legacy adapter
            "GuardrailsRule",
            "create_guardrails_adapter",  # New standardized factory
            "create_guardrails_rule",  # Legacy factory
        ]
    )

# Add PydanticAI adapter to exports if available
if PYDANTIC_AI_AVAILABLE:
    __all__.extend(
        [
            "SifakaPydanticAdapter",
            "SifakaPydanticConfig",
            "create_pydantic_adapter",
            "create_pydantic_adapter_with_critic",
        ]
    )
