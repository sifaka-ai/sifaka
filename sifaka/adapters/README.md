# Sifaka Adapters

This package provides adapter implementations that allow Sifaka to integrate with various external libraries, services, and model providers. It follows the adapter pattern to enable loose coupling between Sifaka components and external systems.

## Architecture

The adapters architecture follows a component-based design:

```
Adapters
├── Base Components
│   ├── Adaptable (interface for adaptable objects)
│   ├── BaseAdapter (foundation class for all adapters)
│   └── AdapterError (standardized error handling)
├── Chain Adapters
│   ├── ModelAdapter (adapt model providers to Chain's Model interface)
│   ├── ValidatorAdapter (adapt rules to Chain's Validator interface)
│   ├── ImproverAdapter (adapt critics to Chain's Improver interface)
│   └── FormatterAdapter (adapt formatters to Chain's Formatter interface)
├── Classifier Adapters
│   ├── ClassifierAdapter (adapt any classifier to standardized interface)
│   └── ClassifierRule (convert classifier to rule for validation)
├── Integration Adapters
│   ├── GuardrailsAdapter (integrate with guardrails-ai validators)
│   └── SifakaPydanticAdapter (integrate with pydantic-ai for structured output)
└── Factory Functions
    ├── create_adapter (create adapter for any adaptable component)
    ├── create_classifier_rule (create rule from classifier)
    └── create_pydantic_adapter (create pydantic adapter with configuration)
```

## Core Components

- **BaseAdapter**: Foundation class that implements common adapter functionality
- **Chain Adapters**: Adapt components to work with Sifaka's chain system
- **Classifier Adapters**: Standardize classifier implementations and convert to rules
- **Integration Adapters**: Connect Sifaka with third-party validation systems
- **Factory Functions**: Simplify adapter creation with sensible defaults

## Usage

### Chain Adapters

Chain adapters allow you to use various components in the Sifaka chain system:

```python
from sifaka.chain import Chain
from sifaka.adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create original components
model_provider = OpenAIProvider("gpt-3.5-turbo")
rule = create_length_rule(min_chars=10, max_chars=1000)
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create adapters (usually done automatically by Chain)
model = ModelAdapter(model_provider)
validator = ValidatorAdapter(rule)
improver = ImproverAdapter(critic)

# Create and run chain
chain = Chain(
    model=model,
    validators=[validator],
    improver=improver,
    max_attempts=3
)
result = chain.run("Write a short story")
```

### Classifier Adapters

Classifier adapters allow you to use classifiers as validators in chains:

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.adapters import create_classifier_rule
from sifaka.classifiers import create_toxicity_classifier

# Create classifier and model
model = OpenAIProvider("gpt-3.5-turbo")
toxicity_classifier = create_toxicity_classifier()

# Create validator rule from classifier
toxicity_rule = create_classifier_rule(
    classifier=toxicity_classifier,
    valid_labels=["non_toxic"],
    threshold=0.8,
    name="toxicity_rule",
    description="Validates that text is not toxic"
)

# Create and run chain
chain = Chain(
    model=model,
    validators=[toxicity_rule],
    max_attempts=3
)
result = chain.run("Write a friendly story")
```

### Guardrails Integration

Connect Sifaka with guardrails-ai for structured validation:

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.adapters import create_guardrails_rule

# Import guardrails components
from guardrails import Guard
from guardrails.validators import ValidLength, NotEmpty

# Create model
model = OpenAIProvider("gpt-3.5-turbo")

# Create guardrails guard
guard = Guard().validate(
    ValidLength(min_length=10, max_length=1000),
    NotEmpty()
)

# Create rule from guardrails guard
guardrails_rule = create_guardrails_rule(
    guard=guard,
    name="guardrails_rule",
    description="Validates text using guardrails validators"
)

# Create and run chain
chain = Chain(
    model=model,
    validators=[guardrails_rule],
    max_attempts=3
)
result = chain.run("Write a short story")
```

### PydanticAI Integration

Integrate Sifaka with pydantic-ai for structured output:

```python
from pydantic import BaseModel, Field
from typing import List
from sifaka.models import OpenAIProvider
from sifaka.adapters import create_pydantic_adapter

# Define output schema
class Character(BaseModel):
    name: str = Field(description="Character name")
    age: int = Field(description="Character age")
    traits: List[str] = Field(description="Character traits")

# Create model provider
model_provider = OpenAIProvider("gpt-4")

# Create pydantic adapter
pydantic_adapter = create_pydantic_adapter(
    model=model_provider,
    output_class=Character,
    max_attempts=3
)

# Generate structured output
character = pydantic_adapter.generate(
    "Create a character for a fantasy story."
)

print(f"Name: {character.name}")
print(f"Age: {character.age}")
print(f"Traits: {', '.join(character.traits)}")
```

## Extending

### Creating a Custom Adapter

```python
from sifaka.adapters import BaseAdapter
from sifaka.core.results import ValidationResult
from typing import Any, List

class CustomSystemAdapter(BaseAdapter):
    """Adapter for custom external system integration."""

    def __init__(self, system: Any, **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.config = kwargs.get("config", {})

    def validate(self, text: str) -> ValidationResult:
        """Validate text using the custom system."""
        # Custom validation logic
        system_result = self.system.check_text(text)

        # Convert to Sifaka validation result
        return ValidationResult(
            passed=system_result.is_valid,
            message=system_result.message,
            score=system_result.confidence,
            issues=system_result.errors or [],
            suggestions=system_result.recommendations or []
        )
```

### Creating a Factory Function

```python
from typing import Any, Dict, List, Optional
from sifaka.adapters import BaseAdapter

def create_custom_adapter(
    system: Any,
    name: str = "custom_adapter",
    description: str = "Custom system adapter",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseAdapter:
    """
    Create a custom adapter for external system integration.

    Args:
        system: The external system to adapt
        name: Adapter name
        description: Adapter description
        config: Configuration parameters
        **kwargs: Additional adapter parameters

    Returns:
        A custom adapter instance
    """
    from .custom_adapter import CustomSystemAdapter

    return CustomSystemAdapter(
        system=system,
        name=name,
        description=description,
        config=config or {},
        **kwargs
    )
```

## Available Adapters

### Chain Adapters
- **ModelAdapter**: Adapts any text generation model to the Chain Model interface
- **ValidatorAdapter**: Adapts any validator/rule to the Chain Validator interface
- **ImproverAdapter**: Adapts any improver/critic to the Chain Improver interface
- **FormatterAdapter**: Adapts any formatter to the Chain Formatter interface

### Classifier Adapters
- **ClassifierAdapter**: Adapts any classifier to a standardized interface
- **ClassifierRule**: Converts any classifier to a Chain-compatible rule

### Integration Adapters
- **GuardrailsAdapter**: Integrates with guardrails-ai validators
- **SifakaPydanticAdapter**: Integrates with pydantic-ai for structured output