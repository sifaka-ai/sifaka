# Implementation Notes: LAC (LLM-Based Actor-Critic) Critic

This document provides implementation details and notes for the LAC (LLM-Based Actor-Critic) Critic in the Sifaka project.

## Overview

The LAC Critic implements the LLM-Based Actor-Critic approach from the paper [Language Feedback Improves Language Model-based Decision Making](https://arxiv.org/abs/2403.03692). It combines language feedback and value scoring to improve language model-based decision making, providing both qualitative feedback and quantitative assessment of text quality.

## Implementation Details

### State Management

The LAC Critic follows the standardized StateManager pattern used in other Sifaka critics:

- Uses a `StateManager` with `CriticState` to store all mutable state
- Stores configuration values in the state's cache dictionary
- Accesses state through the state manager

```python
# State management using StateManager
_state_manager = PrivateAttr(default_factory=create_critic_state)

def __init__(self, config: LACCriticConfig, llm_provider: Any) -> None:
    # Initialize base class
    super().__init__(config)

    # Initialize state
    state = self._state_manager.get_state()

    # Store components in state
    state.cache = {
        "feedback_critic": FeedbackCritic(config=feedback_config, llm_provider=llm_provider),
        "value_critic": ValueCritic(config=value_config, llm_provider=llm_provider),
        "system_prompt": config.system_prompt,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    state.model = llm_provider
    state.initialized = True
```

### Configuration

The LAC Critic uses three dedicated configuration classes:

1. `FeedbackCriticConfig` - For the feedback component
2. `ValueCriticConfig` - For the value component
3. `LACCriticConfig` - For the combined LAC critic

```python
class FeedbackCriticConfig(PromptCriticConfig):
    feedback_prompt_template: str = Field(
        default=DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        description="Template for feedback prompts",
    )

class ValueCriticConfig(PromptCriticConfig):
    value_prompt_template: str = Field(
        default=DEFAULT_VALUE_PROMPT_TEMPLATE,
        description="Template for value prompts",
    )

class LACCriticConfig(CriticConfig):
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the model",
        min_length=1,
    )
    temperature: float = Field(
        default=0.7, description="Temperature for model generation", ge=0.0, le=1.0
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens for model generation", gt=0)
    feedback_prompt_template: str = Field(
        default=DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        description="Template for feedback prompts",
    )
    value_prompt_template: str = Field(
        default=DEFAULT_VALUE_PROMPT_TEMPLATE,
        description="Template for value prompts",
    )
```

### Core Components

The LAC Critic consists of three main components:

1. **FeedbackCritic**: Provides natural language feedback for text
2. **ValueCritic**: Estimates numeric values (e.g., probability of success) for text
3. **LACCritic**: Combines both feedback and value critics

Each component implements the standard critic methods:
- `validate`: Check if text meets quality standards
- `improve`: Improve text based on feedback
- `critique`: Analyze text and provide feedback
- `improve_with_feedback`: Improve text based on provided feedback

### Core Algorithm

The core algorithm for the LAC Critic is implemented in the `run` method:

1. Generate natural language feedback using the FeedbackCritic
2. Generate a numeric value using the ValueCritic
3. Combine both signals into a single result

```python
def run(self, task: str, response: str) -> Dict[str, Any]:
    """
    Generate feedback and value for a response to a task.
    """
    self._check_input(response)

    state = self._state_manager.get_state()

    # Get feedback and value critics
    feedback_critic = state.cache.get("feedback_critic")
    value_critic = state.cache.get("value_critic")

    # Generate feedback and value
    feedback = feedback_critic.run(task, response)
    value = value_critic.run(task, response)

    # Return results
    return {
        "feedback": feedback,
        "value": value,
    }
```

The `improve` method uses both feedback and value to generate an improved response:

```python
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Improve text based on feedback and value.
    """
    # Get task from metadata
    task = self._get_task_from_metadata(metadata)

    # Generate feedback and value
    result = self.run(task, text)
    feedback = result["feedback"]
    value = result["value"]

    # Create improvement prompt
    prompt = (
        f"Task:\n{task}\n\n"
        f"Original response:\n{text}\n\n"
        f"Feedback:\n{feedback}\n\n"
        f"Quality score: {value:.2f} (on a scale of 0 to 1)\n\n"
        f"Improved response:"
    )

    state = self._state_manager.get_state()

    # Generate improved response
    improved_text = state.model.generate(
        prompt,
        system_prompt=state.cache.get("system_prompt", ""),
        temperature=state.cache.get("temperature", 0.7),
        max_tokens=state.cache.get("max_tokens", 1000),
    ).strip()

    return improved_text
```

### Factory Functions

The LAC Critic provides three factory functions for easy creation:

1. `create_feedback_critic`: Creates a FeedbackCritic
2. `create_value_critic`: Creates a ValueCritic
3. `create_lac_critic`: Creates a LACCritic

```python
def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    value_prompt_template: str = DEFAULT_VALUE_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic:
    # Implementation details...
```

## Integration with Sifaka

The LAC Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added default configurations:
   - `DEFAULT_FEEDBACK_CONFIG`
   - `DEFAULT_VALUE_CONFIG`
   - `DEFAULT_LAC_CONFIG`
4. Provided examples in:
   - `examples/critics/lac_critic_demo.py`
   - `examples/critics/lac_comparison_demo.py`

## Testing

The LAC Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of text
3. Feedback generation
4. Value estimation
5. Text improvement using both feedback and value
6. Factory function behavior

## Future Improvements

Potential future improvements for the LAC Critic include:

1. Adding support for more sophisticated value estimation techniques
2. Implementing a more robust parsing of feedback
3. Adding support for multi-step improvement with different prompts at each step
4. Implementing a more sophisticated scoring mechanism for feedback
5. Adding support for tracking the history of improvements
6. Addressing the warning about prompt tokens exceeding max_tokens
7. Improving the tracing functionality to reduce unnecessary warnings

## References

- [Language Feedback Improves Language Model-based Decision Making](https://arxiv.org/abs/2403.03692)
- [Sifaka Critics Documentation](../components/critics.md)
