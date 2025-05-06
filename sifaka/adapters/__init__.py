"""
Adapters for external libraries and services integration with Sifaka.

This package provides adapter implementations that allow Sifaka to integrate with
various external libraries, services, and model providers.

## Architecture Overview

The adapters module follows an adapter pattern to enable Sifaka to work with
external libraries and services without tight coupling:

1. **Protocol Definitions**: Define the expected interface for integration
2. **Adapter Components**: Implement wrappers that translate between systems
3. **Factory Functions**: Provide simple creation patterns for common use cases

## Component Types

Sifaka provides several types of adapters:

1. **Rules Adapters**: Adapters that convert external components to Sifaka rules
2. **Classifier Adapters**: Special adapters for text classification systems
3. **PydanticAI Adapters**: Adapters for integrating with PydanticAI agents

## Usage Patterns

The recommended way to use adapters is through factory functions:

```python
from sifaka.adapters import ClassifierAdapter, create_classifier_rule
from sifaka.classifiers.toxicity import ToxicityClassifier

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

For details on specific adapter implementations, see the respective module documentation.
"""

# Rules adapters
from sifaka.adapters.rules import (
    Adaptable,
    BaseAdapter,
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_rule,
)

# Export types from sifaka.classifiers.base for convenience
from sifaka.classifiers.base import ClassificationResult, ClassifierConfig

# Try to import Guardrails adapters if available
try:
    from sifaka.adapters.rules import (
        GuardrailsValidatorAdapter,
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
    # Rules adapters
    "Adaptable",
    "BaseAdapter",
    "ClassifierAdapter",
    "ClassifierRule",
    "create_classifier_rule",
    # Classifier types
    "ClassificationResult",
    "ClassifierConfig",
]

# Add Guardrails adapter to exports if available
if GUARDRAILS_AVAILABLE:
    __all__.extend(
        [
            "GuardrailsValidatorAdapter",
            "create_guardrails_rule",
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
