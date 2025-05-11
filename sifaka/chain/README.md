# Sifaka Chain

This package provides a simplified and more maintainable implementation of the chain system for orchestrating the validation and improvement flow between models, rules, and critics.

## Architecture

The chain architecture follows a simplified component-based design:

```
Chain
├── Engine (core execution logic)
│   ├── Executor (handles execution flow)
│   └── StateTracker (centralized state management)
├── Components (pluggable components)
│   ├── Model (text generation)
│   ├── Validator (rule-based validation)
│   ├── Improver (output improvement)
│   └── Formatter (result formatting)
└── Plugins (extension mechanism)
    ├── PluginRegistry (plugin discovery and registration)
    └── PluginLoader (dynamic plugin loading)
```

### Core Components

- **Chain**: Main user-facing class for running chains
- **Engine**: Core execution engine that coordinates the flow
- **StateTracker**: Centralized state management
- **Model**: Interface for text generation models
- **Validator**: Interface for output validators
- **Improver**: Interface for output improvers
- **Formatter**: Interface for result formatters
- **Plugin**: Interface for plugins

## Usage

### Basic Usage

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]

# Create chain
chain = Chain(
    model=model,
    validators=validators,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story about a robot.")

# Access the result
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
```

### Advanced Usage with Improver

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [
    create_length_rule(min_chars=10, max_chars=1000),
    create_toxicity_rule(threshold=0.7)
]
improver = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = Chain(
    model=model,
    validators=validators,
    improver=improver,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story about a robot.")

# Access the result
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
print(f"Issues: {result.issues}")
print(f"Suggestions: {result.suggestions}")
```

### Using Factory Functions

```python
from sifaka.chain import create_chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain using factory
chain = create_chain(
    model=model,
    validators=validators,
    improver=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story about a robot.")
```

## Extending

### Creating a Custom Model

```python
from sifaka.chain.interfaces import Model

class MyCustomModel(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        # Custom generation logic
        return f"Generated text for prompt: {prompt}"

    async def generate_async(self, prompt: str) -> str:
        # Custom async generation logic
        return f"Async generated text for prompt: {prompt}"
```

### Creating a Custom Validator

```python
from sifaka.chain.interfaces import Validator, ValidationResult

class MyCustomValidator(Validator):
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: str) -> ValidationResult:
        length = len(output)
        if length < self.min_length:
            return ValidationResult(
                passed=False,
                message=f"Output too short: {length} < {self.min_length}",
                score=0.0,
                issues=[f"Output length ({length}) is less than minimum ({self.min_length})"],
                suggestions=["Make the output longer"]
            )
        if length > self.max_length:
            return ValidationResult(
                passed=False,
                message=f"Output too long: {length} > {self.max_length}",
                score=0.0,
                issues=[f"Output length ({length}) is greater than maximum ({self.max_length})"],
                suggestions=["Make the output shorter"]
            )
        return ValidationResult(
            passed=True,
            message="Length validation passed",
            score=1.0
        )

    async def validate_async(self, output: str) -> ValidationResult:
        # For simple validators, async can just call the sync version
        return self.validate(output)
```

### Creating a Plugin

```python
from typing import Any, Dict
from sifaka.chain.interfaces import Plugin, Validator, ValidationResult

class MyValidatorPlugin(Plugin):
    @property
    def name(self) -> str:
        return "my_validator_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def component_type(self) -> str:
        return "validator"

    def create_component(self, config: Dict[str, Any]) -> Validator:
        min_length = config.get("min_length", 10)
        max_length = config.get("max_length", 1000)

        class LengthValidator(Validator):
            def validate(self, output: str) -> ValidationResult:
                length = len(output)
                if length < min_length:
                    return ValidationResult(
                        passed=False,
                        message=f"Output too short: {length} < {min_length}",
                        score=0.0
                    )
                if length > max_length:
                    return ValidationResult(
                        passed=False,
                        message=f"Output too long: {length} > {max_length}",
                        score=0.0
                    )
                return ValidationResult(
                    passed=True,
                    message="Length validation passed",
                    score=1.0
                )

            async def validate_async(self, output: str) -> ValidationResult:
                return self.validate(output)

        return LengthValidator()
```
