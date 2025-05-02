# Chain Architecture

The Chain module in Sifaka provides a flexible way to orchestrate the interaction between models, rules, critics, and other components.

## Overview

The Chain architecture in Sifaka follows the Single Responsibility Principle by breaking down functionality into smaller, focused components:

- **ChainCore**: Main interface that delegates to specialized components
- **PromptManager**: Manages prompt creation and feedback integration
- **ValidationManager**: Manages validation logic and rule application
- **RetryStrategy**: Handles retry logic with different strategies
- **ResultFormatter**: Handles formatting and processing of results

## Chain Components

```
ChainCore
├── PromptManager
├── ValidationManager
├── RetryStrategy
└── ResultFormatter
```

### Component Responsibilities

#### ChainCore
- Coordinates the processing flow
- Delegates to specialized components
- Manages the interaction between components
- Provides a unified interface

#### PromptManager
- Creates prompts for the model
- Integrates feedback from rules and critics into prompts
- Handles prompt templating and formatting

#### ValidationManager
- Manages rule validation
- Applies rules to model outputs
- Collects and organizes validation results

#### RetryStrategy
- Implements retry logic
- Determines when to retry after validation failures
- Handles backoff periods between retries

#### ResultFormatter
- Formats the final result
- Collects metadata from the process
- Creates structured output objects

## Factory Functions

For convenience, Sifaka provides factory functions to create common chain configurations:

- `create_simple_chain`: Creates a chain with a simple retry strategy
- `create_backoff_chain`: Creates a chain with an exponential backoff retry strategy
- `create_validation_chain`: Creates a chain focused on validation only
- `create_improvement_chain`: Creates a chain focused on improvement with a critic

## Configuration Options

### Chain Configuration
- `model`: Model provider to use
- `rules`: List of rules to apply
- `critic`: Optional critic for improvement
- `max_attempts`: Maximum number of retry attempts

### RetryStrategy Options
- `max_attempts`: Maximum number of retry attempts
- `initial_backoff`: Initial backoff period (for backoff strategy)
- `backoff_factor`: Factor to increase backoff by (for backoff strategy)
- `max_backoff`: Maximum backoff period (for backoff strategy)

## Usage Examples

### Basic Chain Usage

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.rules.formatting.length import create_length_rule

# Create components
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(temperature=0.7, max_tokens=500)
)

rules = [create_length_rule(min_chars=10, max_chars=100)]

# Create chain
chain = create_simple_chain(
    model=model,
    rules=rules,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story about a cat.")

# Check results
print(f"Output: {result.output}")
print(f"Validation passed: {result.all_passed}")
```

### Chain with Critic for Improvement

```python
from sifaka.chain import create_improvement_chain
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.critics import create_style_critic
from sifaka.rules.formatting.length import create_length_rule

# Create components
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(temperature=0.7, max_tokens=500)
)

critic = create_style_critic()
rules = [create_length_rule(min_chars=10, max_chars=100)]

# Create improvement chain
chain = create_improvement_chain(
    name="improvement_chain",
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Run chain
result = chain.process("Write a short story about a cat.")

# Check results
if result.improved_text:
    print(f"Improved text: {result.improved_text}")
else:
    print(f"Original text: {result.output}")
```

### Advanced Custom Chain

```python
from sifaka.chain import (
    ChainCore,
    PromptManager,
    ValidationManager,
    BackoffRetryStrategy,
    ResultFormatter
)
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.critics import create_prompt_critic, PromptCriticConfig
from sifaka.rules.formatting.length import create_length_rule

# Create components
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(temperature=0.7, max_tokens=500)
)

rules = [create_length_rule(min_chars=10, max_chars=100)]
critic = create_prompt_critic(llm_provider=model)

# Create specialized components
validation_manager = ValidationManager[str](rules)
prompt_manager = PromptManager()
retry_strategy = BackoffRetryStrategy[str](
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0
)
result_formatter = ResultFormatter[str]()

# Create chain
chain = ChainCore[str](
    model=model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter,
    critic=critic
)

# Run chain
result = chain.run("Write a short story about a cat.")
```

## Best Practices

1. **Chain Selection**
   - Use simple chains for basic validation
   - Use improvement chains when output quality is critical
   - Use backoff chains for rate-limited APIs

2. **Rule Organization**
   - Group related rules together
   - Use rule priorities appropriately
   - Start with the most important validations

3. **Model Configuration**
   - Choose appropriate temperature settings
   - Set reasonable max token limits
   - Configure models based on task requirements

4. **Error Handling**
   - Handle validation failures gracefully
   - Set appropriate retry limits
   - Log validation and improvement steps

## Extension

### Creating a Custom PromptManager

```python
from sifaka.chain.managers.prompt import PromptManager

class CustomPromptManager(PromptManager):
    """Custom prompt manager that adds a system message."""

    def create_prompt_with_feedback(self, original_prompt: str, feedback: str) -> str:
        """Create a new prompt with feedback."""
        return f"System: {feedback}\n\nUser: {original_prompt}"
```

### Creating a Custom RetryStrategy

```python
from sifaka.chain.strategies.retry import RetryStrategy

class CustomRetryStrategy(RetryStrategy):
    """Custom retry strategy."""

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

## Testing

See [Testing Guide](testing/chain_testing.md) for information on testing chain configurations.

## API Reference

For detailed API information, see [Chain API Reference](api/chain.md).