# Sifaka Chain

This package provides a sophisticated, component-based implementation of the chain system for orchestrating the validation and improvement flow between models, rules, and critics.

## Architecture

The chain architecture follows a refined component-based design:

```
Chain
├── Engine (core execution logic)
│   ├── RetryManager (handles retry strategies)
│   ├── CacheManager (manages result caching)
│   └── StateManager (centralized state management)
├── Components (pluggable components)
│   ├── Model (text generation)
│   ├── Validator (rule-based validation)
│   ├── Improver (output improvement)
│   └── Formatter (result formatting)
├── Factories (component creation)
│   ├── create_chain (standard chain creation)
│   ├── create_simple_chain (simplified creation with dependency injection)
│   └── create_backoff_chain (chain with backoff retry strategy)
└── Adapters (component compatibility)
    ├── ModelAdapter (adapt model providers)
    ├── ValidatorAdapter (adapt rules)
    └── ImproverAdapter (adapt critics)
```

### Core Components

- **Chain**: Main user-facing class for running chains with proper lifecycle management
- **Engine**: Core execution engine that coordinates the flow between components
- **StateManager**: Centralized state management for tracking execution
- **Model**: Interface for text generation models
- **Validator**: Interface for output validators
- **Improver**: Interface for output improvers
- **Formatter**: Interface for result formatters

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
print(f"Execution time: {result.execution_time:.2f}s")
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
print(f"Issues: {[r.issues for r in result.validation_results if not r.passed]}")
print(f"Suggestions: {[r.suggestions for r in result.validation_results if not r.passed]}")
```

### Using Factory Functions

```python
from sifaka.chain.factories import create_simple_chain, create_backoff_chain
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

# Create standard chain using factory
chain = create_simple_chain(
    model=model,
    rules=validators,
    critic=critic,
    max_attempts=3
)

# Create chain with backoff retry strategy
backoff_chain = create_backoff_chain(
    model=model,
    rules=validators,
    critic=critic,
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0
)

# Run chain
result = chain.run("Write a short story about a robot.")
```

## Extending

### Creating a Custom Model

```python
from sifaka.interfaces.chain.components import Model

class MyCustomModel(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        # Custom generation logic
        return f"Generated text for prompt: {prompt}"
```

### Creating a Custom Validator

```python
from sifaka.interfaces.chain.components import Validator
from sifaka.interfaces.chain.models import ValidationResult

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
```

### Creating a Custom Improver

```python
from typing import List
from sifaka.interfaces.chain.components import Improver
from sifaka.interfaces.chain.models import ValidationResult

class MyCustomImprover(Improver):
    def improve(self, output: str, validation_results: List[ValidationResult]) -> str:
        # Collect all issues
        issues = []
        for result in validation_results:
            if not result.passed and hasattr(result, "issues"):
                issues.extend(result.issues)

        # Simple improvement logic
        improved_output = output
        for issue in issues:
            if "too short" in issue:
                improved_output += " Additional content to make the output longer."
            if "too long" in issue:
                improved_output = improved_output[:len(improved_output)//2] + "..."

        return improved_output
```

### Error Handling and Monitoring

```python
from sifaka.chain import Chain
from sifaka.utils.errors import ChainError

try:
    # Create and run chain
    chain = Chain(model=model, validators=validators)
    result = chain.run("Write a short story")

    # Check statistics
    stats = chain.get_statistics()
    print(f"Execution count: {stats['execution_count']}")
    print(f"Average execution time: {stats['avg_execution_time']:.2f}s")

except ChainError as e:
    print(f"Chain execution failed: {str(e)}")
```
