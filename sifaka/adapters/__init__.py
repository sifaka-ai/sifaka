"""
Adapters

Adapters for external libraries and services integration with Sifaka.

## Overview
This package provides adapter implementations that allow Sifaka to integrate with
various external libraries, services, and model providers. It follows the adapter
pattern to enable loose coupling between Sifaka and external systems.

## Components
1. **Classifier Adapters**: Adapters for text classification systems
2. **Guardrails Adapters**: Adapters for using Guardrails validators
3. **PydanticAI Adapters**: Adapters for integrating with PydanticAI agents

## Usage Examples
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
