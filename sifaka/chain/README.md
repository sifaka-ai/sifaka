# Sifaka Chain

This package provides components for orchestrating the validation and improvement flow between models, rules, and critics.

## Architecture

The chain architecture follows the Single Responsibility Principle by breaking down functionality into smaller, focused components:

```
ChainCore
├── PromptManager
├── ValidationManager
├── RetryStrategy
└── ResultFormatter
```

### Core Components

- **ChainCore**: Main interface that delegates to specialized components
- **PromptManager**: Manages prompt creation and management
- **ValidationManager**: Manages validation logic and rule management
- **RetryStrategy**: Handles retry logic with different strategies
- **ResultFormatter**: Handles formatting and processing of results

### Retry Strategies

- **SimpleRetryStrategy**: Simple retry strategy that retries a fixed number of times
- **BackoffRetryStrategy**: Retry strategy that increases the wait time between retries

## Usage

### Basic Usage

```python
from sifaka.chain import create_simple_chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create a model provider
model = OpenAIProvider(model_name="gpt-4")

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=100),
    create_toxicity_rule(),
]

# Create a chain
chain = create_simple_chain(
    model=model,
    rules=rules,
    max_attempts=3,
)

# Run the chain
result = chain.run("Write a short story about a robot.")

# Access the result
print(f"Output: {result.output}")
print(f"All rules passed: {result.all_passed}")
```

### Advanced Usage

```python
from sifaka.chain import (
    ChainCore,
    PromptManager,
    ValidationManager,
    BackoffRetryStrategy,
    ResultFormatter,
)
from sifaka.critics import create_prompt_critic
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule, create_toxicity_rule

# Create a model provider
model = OpenAIProvider(model_name="gpt-4")

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=100),
    create_toxicity_rule(),
]

# Create a critic
critic = create_prompt_critic(model)

# Create specialized components
validation_manager = ValidationManager(rules)
prompt_manager = PromptManager()
retry_strategy = BackoffRetryStrategy(
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0,
)
result_formatter = ResultFormatter()

# Create a chain
chain = ChainCore(
    model=model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter,
    critic=critic,
)

# Run the chain
result = chain.run("Write a short story about a robot.")
```

## Extending

### Creating a Custom Retry Strategy

```python
from sifaka.chain.strategies.retry import RetryStrategy

class MyCustomRetryStrategy(RetryStrategy):
    def __init__(self, max_attempts: int = 3):
        self._max_attempts = max_attempts
        
    def run(
        self,
        prompt: str,
        generator: Generator[OutputType],
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ) -> ChainResult[OutputType]:
        # Custom retry logic here
        pass
```

### Creating a Custom Prompt Manager

```python
from sifaka.chain.managers.prompt import PromptManager

class MyCustomPromptManager(PromptManager):
    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        # Custom prompt creation logic here
        return f"System: {feedback}\n\nUser: {original_prompt}"
```
