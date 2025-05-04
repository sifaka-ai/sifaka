# Chain API Reference

Chains are the orchestration components in Sifaka that coordinate model providers, rules, critics, and other components to generate and validate text. They implement the core feedback loop of the framework.

## Core Classes and Protocols

### ChainCore

`ChainCore` is the main implementation of the Chain interface, delegating to specialized components.

```python
from sifaka.chain.core import ChainCore
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic
from sifaka.chain.managers.validation import ValidationManager
from sifaka.chain.managers.prompt import PromptManager
from sifaka.chain.strategies.retry import RetryStrategy
from sifaka.chain.formatters.result import ResultFormatter

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic()

# Create managers
validation_manager = ValidationManager(rules=[rule])
prompt_manager = PromptManager()
retry_strategy = RetryStrategy(max_attempts=3)
result_formatter = ResultFormatter()

# Create the chain core
chain = ChainCore(
    model_provider=model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter,
    critic_core=critic
)
```

## Configuration

### ChainConfig

`ChainConfig` is the configuration class for chains.

```python
from sifaka.chain.config import ChainConfig

# Create a chain configuration
config = ChainConfig(
    max_attempts=3,
    trace_enabled=True,
    params={
        "system_prompt": "You are a helpful assistant.",
    }
)

# Access configuration values
print(f"Max attempts: {config.max_attempts}")
print(f"Trace enabled: {config.trace_enabled}")
print(f"System prompt: {config.params['system_prompt']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    max_attempts=5,
    params={"system_prompt": "You are an expert assistant."}
)
```

## Results

### ChainResult

`ChainResult` represents the result of a chain execution.

```python
from sifaka.chain.result import ChainResult
from sifaka.rules.base import RuleResult
from sifaka.critics.base import CriticMetadata

# Create rule results
rule_results = [
    RuleResult(
        passed=True,
        rule_name="length_rule",
        message="Text length is acceptable",
        metadata={"length": 100}
    ),
    RuleResult(
        passed=False,
        rule_name="toxicity_rule",
        message="Text contains toxic content",
        metadata={"toxicity_score": 0.8}
    )
]

# Create critic metadata
critic_metadata = CriticMetadata(
    score=0.7,
    feedback="Text needs improvement",
    issues=["Text contains toxic content"],
    suggestions=["Remove toxic language"]
)

# Create a chain result
result = ChainResult(
    output="This is the generated text",
    prompt="Generate a short text",
    rule_results=rule_results,
    critic_metadata=critic_metadata,
    attempts=2,
    max_attempts=3,
    processing_time_ms=150
)

# Access result values
print(f"Output: {result.output}")
print(f"Prompt: {result.prompt}")
print(f"Rule results: {result.rule_results}")
print(f"Critic metadata: {result.critic_metadata}")
print(f"Attempts: {result.attempts} of {result.max_attempts}")
print(f"Processing time: {result.processing_time_ms} ms")
```

## Chain Components

Chains in Sifaka are composed of several specialized components:

### ValidationManager

`ValidationManager` manages rule validation.

```python
from sifaka.chain.managers.validation import ValidationManager
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.toxicity import create_toxicity_rule

# Create rules
length_rule = create_length_rule(min_chars=50, max_chars=200)
toxicity_rule = create_toxicity_rule(threshold=0.7)

# Create a validation manager
validation_manager = ValidationManager(rules=[length_rule, toxicity_rule])

# Validate text
results = validation_manager.validate("This is a test")
passed = validation_manager.all_passed(results)
print(f"All rules passed: {passed}")
```

### PromptManager

`PromptManager` manages prompt handling.

```python
from sifaka.chain.managers.prompt import PromptManager

# Create a prompt manager
prompt_manager = PromptManager(
    system_prompt="You are a helpful assistant.",
    prompt_template="{prompt}"
)

# Format a prompt
formatted_prompt = prompt_manager.format_prompt("Generate a short text")
print(f"Formatted prompt: {formatted_prompt}")
```

### RetryStrategy

`RetryStrategy` manages retry logic.

```python
from sifaka.chain.strategies.retry import RetryStrategy

# Create a retry strategy
retry_strategy = RetryStrategy(max_attempts=3)

# Execute with retry
result = retry_strategy.execute(
    lambda: "Success" if retry_strategy.current_attempt > 1 else None,
    lambda result: result is not None
)
print(f"Result: {result}")
print(f"Attempts: {retry_strategy.current_attempt}")
```

### ResultFormatter

`ResultFormatter` formats chain results.

```python
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.chain.result import ChainResult
from sifaka.rules.base import RuleResult

# Create a result formatter
formatter = ResultFormatter()

# Format a chain result
result = ChainResult(
    output="This is the generated text",
    prompt="Generate a short text",
    rule_results=[
        RuleResult(
            passed=True,
            rule_name="length_rule",
            message="Text length is acceptable",
            metadata={"length": 100}
        )
    ],
    attempts=1,
    max_attempts=3
)
formatted_result = formatter.format(result)
print(f"Formatted result: {formatted_result}")
```

## Chain Types

Sifaka provides several types of chains:

### SimpleChain

`SimpleChain` is a basic chain implementation with a single model, rules, and critic.

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic()

# Create a simple chain
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic,
    max_attempts=3
)
```

### BackoffChain

`BackoffChain` is a chain that implements exponential backoff for retries.

```python
from sifaka.chain import create_backoff_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic()

# Create a backoff chain
chain = create_backoff_chain(
    model=model,
    rules=[rule],
    critic=critic,
    max_attempts=3,
    initial_delay_ms=100,
    max_delay_ms=1000
)
```

## Usage Examples

### Basic Chain Usage

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to meet the length requirements."
)

# Create a chain
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic,
    max_attempts=3
)

# Run the chain
result = chain.run("Write a short description of a sunset.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

### Chain with Multiple Rules

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.content.toxicity import create_toxicity_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
length_rule = create_length_rule(min_chars=50, max_chars=200)
toxicity_rule = create_toxicity_rule(threshold=0.7)
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to meet the requirements."
)

# Create a chain with multiple rules
chain = create_simple_chain(
    model=model,
    rules=[length_rule, toxicity_rule],
    critic=critic,
    max_attempts=3
)

# Run the chain
result = chain.run("Write a short story about a conflict.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```

### Chain with Custom Components

```python
from sifaka.chain.core import ChainCore
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic
from sifaka.chain.managers.validation import ValidationManager
from sifaka.chain.managers.prompt import PromptManager
from sifaka.chain.strategies.retry import RetryStrategy
from sifaka.chain.formatters.result import ResultFormatter

# Create components
model = create_openai_chat_provider(model_name="gpt-4")
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic()

# Create custom managers
validation_manager = ValidationManager(
    rules=[rule],
    validate_empty=False,
    empty_text_result="empty"
)
prompt_manager = PromptManager(
    system_prompt="You are a helpful assistant.",
    prompt_template="Please respond to the following: {prompt}"
)
retry_strategy = RetryStrategy(
    max_attempts=5,
    should_retry=lambda result, attempt: not all(r.passed for r in result.rule_results) and attempt < 5
)
result_formatter = ResultFormatter(
    include_rule_results=True,
    include_critic_metadata=True
)

# Create the chain with custom components
chain = ChainCore(
    model_provider=model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter,
    critic_core=critic
)

# Run the chain
result = chain.run("Write a short description of a sunset.")
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
```
