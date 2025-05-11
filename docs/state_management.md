# State Management in Sifaka

This document describes how state is managed throughout the Sifaka framework.

## State Management Principles

Sifaka follows these principles for state management:

1. **Immutability**: State objects are immutable to prevent accidental changes
2. **Centralization**: State is managed through dedicated state objects
3. **Type Safety**: State objects use Pydantic models for type safety
4. **Encapsulation**: State access is controlled through properties
5. **Lifecycle Management**: Clear initialization and cleanup processes

## Core State Management

### Chain State

The `ChainCore` class uses a dedicated state management system:

```python
from sifaka.chain.core import ChainCore
from sifaka.chain.managers.validation import ValidationManager
from sifaka.chain.managers.prompt import PromptManager
from sifaka.chain.strategies.retry import SimpleRetryStrategy
from sifaka.chain.adapters import FormatterAdapter

# Create components
validation_manager = ValidationManager(rules=[...])
prompt_manager = PromptManager()
retry_strategy = SimpleRetryStrategy(max_attempts=3)
result_formatter = FormatterAdapter(formatter=None)

# Create chain core with state management
chain_core = ChainCore(
    model=model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager,
    retry_strategy=retry_strategy,
    result_formatter=result_formatter
)

# Access state through properties
print(f"Model: {chain_core.model}")
print(f"Validation Manager: {chain_core.validation_manager}")
print(f"Prompt Manager: {chain_core.prompt_manager}")
print(f"Retry Strategy: {chain_core.retry_strategy}")
print(f"Result Formatter: {chain_core.result_formatter}")
print(f"Critic: {chain_core.critic}")
```

### State Components

The chain state includes these components:

1. **Model Provider**
   - Handles text generation
   - Manages model-specific state
   - Tracks token usage and costs

2. **Validation Manager**
   - Manages rule validation state
   - Tracks validation results
   - Handles rule priorities

3. **Prompt Manager**
   - Manages prompt state
   - Tracks prompt history
   - Handles prompt modifications

4. **Retry Strategy**
   - Manages retry state
   - Tracks attempt counts
   - Handles backoff timing

5. **Result Formatter**
   - Manages result state
   - Tracks formatting options
   - Handles output formatting

6. **Critic (Optional)**
   - Manages improvement state
   - Tracks critique history
   - Handles improvement attempts

## State Lifecycle

### 1. Initialization

```python
# Initialize chain state
chain_core.initialize()

# State is ready for use
print(f"Initialized: {chain_core.initialized}")
```

### 2. State Updates

```python
# Update configuration
chain_core.update_config({
    "max_attempts": 5,
    "trace_enabled": True
})

# State is updated through properties
print(f"New max attempts: {chain_core.retry_strategy.max_attempts}")
```

### 3. Cleanup

```python
# Clean up state
chain_core.cleanup()

# State is cleaned up
print(f"Initialized: {chain_core.initialized}")
```

## State Access Patterns

### 1. Property Access

```python
# Access state through properties
model = chain_core.model
validation_manager = chain_core.validation_manager
prompt_manager = chain_core.prompt_manager
retry_strategy = chain_core.retry_strategy
result_formatter = chain_core.result_formatter
critic = chain_core.critic
```

### 2. Configuration Access

```python
# Access configuration
config = chain_core.config

# Update configuration
chain_core.update_config({
    "max_attempts": 5,
    "trace_enabled": True
})
```

### 3. Component State

```python
# Access component state
validation_state = chain_core.validation_manager.state
prompt_state = chain_core.prompt_manager.state
retry_state = chain_core.retry_strategy.state
```

## Best Practices

1. **Use Property Access**
   - Access state through properties
   - Don't modify state directly
   - Use update methods for changes

2. **Initialize Properly**
   - Call initialize() before use
   - Check initialized state
   - Handle initialization errors

3. **Clean Up Resources**
   - Call cleanup() when done
   - Release resources properly
   - Handle cleanup errors

4. **Handle State Updates**
   - Use update_config() for changes
   - Validate updates
   - Handle update errors

5. **Monitor State**
   - Track state changes
   - Log important updates
   - Handle state errors

## Examples

### Basic Chain State

```python
from sifaka.chain import ChainOrchestrator
from sifaka.chain.config import ChainConfig

# Create chain with state
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    config=ChainConfig(
        max_attempts=3,
        trace_enabled=True
    )
)

# Initialize state
chain.initialize()

# Use chain
result = chain.run("Write a story")

# Clean up state
chain.cleanup()
```

### Chain with Critic State

```python
from sifaka.chain import ChainOrchestrator
from sifaka.critics import create_prompt_critic

# Create critic with state
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor."
)

# Create chain with critic state
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    critic=critic
)

# Initialize state
chain.initialize()

# Use chain with critic
result = chain.run("Write a poem")

# Clean up state
chain.cleanup()
```

### Chain with Custom State

```python
from sifaka.chain import ChainOrchestrator
from sifaka.chain.config import ChainConfig

# Create chain with custom state
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    config=ChainConfig(
        max_attempts=3,
        trace_enabled=True,
        params={
            "custom_state": "value",
            "state_tracking": True
        }
    )
)

# Initialize state
chain.initialize()

# Access custom state
print(f"Custom state: {chain.config.params.get('custom_state')}")

# Use chain
result = chain.run("Write a story")

# Clean up state
chain.cleanup()
```
