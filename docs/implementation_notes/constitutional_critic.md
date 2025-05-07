# Implementation Notes: Constitutional Critic

This document provides implementation details and notes for the Constitutional Critic in the Sifaka project.

## Overview

The Constitutional Critic implements a Constitutional AI approach, which evaluates responses against a set of human-written principles (a "constitution") and provides natural language feedback when violations are detected.

Based on the paper [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073).

## Implementation Details

### State Management

The Constitutional Critic follows the standardized StateManager pattern used in Sifaka critics:

- Uses a `StateManager` with `CriticState` to store all mutable state
- Stores configuration values in the state's cache dictionary
- Accesses state through the state manager

```python
# State management using StateManager
_state_manager = PrivateAttr(default_factory=create_critic_state)

def __init__(self, config: ConstitutionalCriticConfig, llm_provider: Any) -> None:
    # Initialize base class
    super().__init__(config)

    # Initialize state
    state = self._state_manager.get_state()

    # Store components in state
    state.model = llm_provider
    state.cache = {
        "principles": config.principles,
        "critique_prompt_template": config.critique_prompt_template,
        "improvement_prompt_template": config.improvement_prompt_template,
        "system_prompt": config.system_prompt,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    state.initialized = True
```

### Configuration

The Constitutional Critic uses a dedicated configuration class that extends `CriticConfig`:

```python
class ConstitutionalCriticConfig(CriticConfig):
    principles: List[str] = Field(
        ...,
        description="List of principles that responses should adhere to",
        min_items=1,
    )
    system_prompt: str = Field(
        default="You are an expert at evaluating content against principles.",
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
    critique_prompt_template: str = Field(
        default=(
            "Please evaluate the following response against these principles:\n\n"
            "PRINCIPLES:\n{principles}\n\n"
            "TASK:\n{task}\n\n"
            "RESPONSE:\n{response}\n\n"
            "Provide your evaluation in the following format:\n"
            "SCORE: [number between 0 and 1]\n"
            "FEEDBACK: [your general feedback]\n"
            "ISSUES:\n- [issue 1]\n- [issue 2]\n"
            "SUGGESTIONS:\n- [suggestion 1]\n- [suggestion 2]"
        ),
        description="Template for critique prompts",
    )
    improvement_prompt_template: str = Field(
        default=(
            "Please revise the following response to better align with these principles:\n\n"
            "PRINCIPLES:\n{principles}\n\n"
            "TASK:\n{task}\n\n"
            "ORIGINAL RESPONSE:\n{response}\n\n"
            "CRITIQUE:\n{critique}\n\n"
            "REVISED RESPONSE:"
        ),
        description="Template for improvement prompts",
    )
```

### Principles Management

The Constitutional Critic manages principles through a dedicated method:

```python
def _format_principles(self) -> str:
    """Format principles for inclusion in prompts."""
    state = self._state_manager.get_state()
    principles = state.cache.get("principles", [])
    if not principles:
        return "No principles defined."

    return "\n".join(f"{i+1}. {principle}" for i, principle in enumerate(principles))
```

### Core Methods

The Constitutional Critic implements three core methods:

1. **validate**: Determines if a response adheres to principles
2. **critique**: Evaluates a response against principles and provides feedback
3. **improve**: Generates an improved response based on critique

Each method interacts directly with the language model:

```python
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze a response against the principles and provide detailed feedback.
    """
    # Ensure initialized
    state = self._state_manager.get_state()
    if not state.initialized:
        raise RuntimeError("ConstitutionalCritic not properly initialized")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    # Get task from metadata
    task = self._get_task_from_metadata(metadata)

    # Format principles
    principles_text = self._format_principles()

    # Create critique prompt
    prompt = state.cache.get("critique_prompt_template", "").format(
        principles=principles_text,
        task=task,
        response=text,
    )

    # Generate critique
    critique_text = state.model.generate(
        prompt,
        system_prompt=state.cache.get("system_prompt", ""),
        temperature=state.cache.get("temperature", 0.7),
        max_tokens=state.cache.get("max_tokens", 1000),
    ).strip()

    # Parse critique
    # Implementation details...
```

### Factory Function

The Constitutional Critic provides a factory function for easy creation:

```python
def create_constitutional_critic(
    llm_provider: Any,
    principles: List[str],
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at evaluating content against principles.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic with the given parameters.
    """
    # Implementation details...
```

## Integration with Sifaka

The Constitutional Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added a default configuration `DEFAULT_CONSTITUTIONAL_CONFIG`
4. Provided comprehensive tests in `tests/critics/test_constitutional.py`
5. Provided examples in `examples/critics/constitutional_critic_example.py`

## Component Interactions

The Constitutional Critic interacts with several components:

1. **Language Model Provider**: Generates responses based on prompts
2. **Principles**: Define the standards that responses should meet
3. **Metadata**: Provides context for evaluation (e.g., task description)

This architecture allows for explicit control over the behavior of language models by defining clear principles that responses should follow.

## Testing

The Constitutional Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of responses against principles
3. Critique generation
4. Response improvement
5. Factory function behavior
6. Error handling

## Future Improvements

Potential future improvements for the Constitutional Critic include:

1. Adding support for hierarchical principles
2. Implementing a more robust parsing of critiques
3. Adding support for principle-specific feedback
4. Implementing a more sophisticated scoring mechanism
5. Adding support for tracking the history of improvements

## References

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Sifaka Critics Documentation](../components/critics.md)
