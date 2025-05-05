# Implementation Notes: Prompt Critic

This document provides implementation details and notes for the Prompt Critic in the Sifaka project.

## Overview

The Prompt Critic is a fundamental critic implementation that uses language models to evaluate, validate, and improve text outputs. It serves as the foundation for many other critic types in Sifaka.

## Implementation Details

### State Management

The Prompt Critic follows the standard state management pattern used in Sifaka critics:

- Uses a `CriticState` object to store all mutable state
- Stores configuration values in the state's cache dictionary
- Accesses state through direct state access

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
self._state.response_parser = ResponseParser()
self._state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)
self._state.cache = {
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
    "critique_service": CritiqueService(
        llm_provider=llm_provider,
        prompt_manager=self._state.prompt_manager,
        response_parser=self._state.response_parser,
    ),
}
self._state.initialized = True
```

### Configuration

The Prompt Critic uses a dedicated configuration class:

```python
class PromptCriticConfig(CriticConfig):
    system_prompt: str = Field(
        default="You are an expert at evaluating and improving text.",
        description="System prompt for the model",
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for model generation",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for model generation",
        gt=0,
    )
    validation_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for validation prompts",
    )
    critique_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for critique prompts",
    )
    improvement_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for improvement prompts",
    )
```

### Core Methods

The Prompt Critic implements three core methods:

1. **validate**: Determines if text meets quality standards
2. **critique**: Analyzes text and provides detailed feedback
3. **improve**: Generates an improved version of text based on feedback

Each method delegates to a `CritiqueService` that handles the interaction with the language model:

```python
def validate(self, text: str) -> bool:
    """Validate text quality."""
    # Ensure initialized
    if not self._state.initialized:
        raise RuntimeError("PromptCritic not properly initialized")

    if not isinstance(text, str) or not text.strip():
        return False

    # Get critique service from state
    critique_service = self._state.cache.get("critique_service")
    if not critique_service:
        raise RuntimeError("Critique service not initialized")

    # Delegate to critique service
    return critique_service.validate(text)
```

### Factory Function

The Prompt Critic provides a factory function for easy creation:

```python
def create_prompt_critic(
    llm_provider: LanguageModel,
    name: str = "factory_critic",
    description: str = "Evaluates and improves text using language models",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    validation_prompt_template: Optional[str] = None,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], PromptCriticConfig]] = None,
    **kwargs: Any,
) -> PromptCritic:
    """
    Create a prompt critic with the given parameters.
    """
    # Implementation details...
```

## Integration with Sifaka

The Prompt Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added a default configuration `DEFAULT_PROMPT_CONFIG`
4. Provided comprehensive tests in `tests/critics/test_prompt.py`
5. Provided examples in `examples/critics/prompt_critic_example.py`

## Component Interactions

The Prompt Critic interacts with several components:

1. **Language Model Provider**: Generates responses based on prompts
2. **Prompt Manager**: Creates and formats prompts for different operations
3. **Response Parser**: Parses responses from language models
4. **Memory Manager**: Stores past interactions for potential future use
5. **Critique Service**: Coordinates the critique and improvement process

This component-based architecture allows for flexible customization and extension of the critic's capabilities.

## Testing

The Prompt Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of text
3. Critique generation
4. Text improvement
5. Factory function behavior
6. Error handling

## Future Improvements

Potential future improvements for the Prompt Critic include:

1. Adding support for more sophisticated prompt templates
2. Implementing a more robust parsing of responses
3. Adding support for few-shot examples in prompts
4. Implementing a more sophisticated scoring mechanism
5. Adding support for tracking the history of improvements

## References

- [Sifaka Critics Documentation](../components/critics.md)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
