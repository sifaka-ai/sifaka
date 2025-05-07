# Implementation Notes: Reflexion Critic

This document provides implementation details and notes for the Reflexion Critic in the Sifaka project.

## Overview

The Reflexion Critic implements the Reflexion approach, which enables language model agents to learn from feedback without requiring weight updates. It maintains reflections in memory to improve future text generation.

## Implementation Details

### State Management

The Reflexion Critic follows the standardized StateManager pattern used in Sifaka critics:

- Uses a `StateManager` with `CriticState` to store all mutable state
- Stores configuration values in the state's cache dictionary
- Accesses state through the state manager

```python
# State management using StateManager
_state_manager = PrivateAttr(default_factory=create_critic_state)

def __init__(self, config: ReflexionCriticConfig, llm_provider: Any, prompt_factory: Any = None) -> None:
    # Initialize base class
    super().__init__(config)

    # Initialize state
    state = self._state_manager.get_state()

    # Import required components
    from .managers.prompt_factories import ReflexionCriticPromptManager
    from .managers.response import ResponseParser
    from .managers.memory import MemoryManager
    from .services.critique import CritiqueService

    # Store components in state
    state.model = llm_provider
    state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
    state.response_parser = ResponseParser()
    state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)
    state.cache = {
        "system_prompt": config.system_prompt,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reflection_depth": config.reflection_depth,
        "critique_service": CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
            memory_manager=state.memory_manager,
        ),
    }
    state.initialized = True
```

### Configuration

The Reflexion Critic uses a dedicated configuration class that extends `PromptCriticConfig`:

```python
class ReflexionCriticConfig(PromptCriticConfig):
    memory_buffer_size: int = Field(
        default=5,
        description="Size of the memory buffer for storing reflections",
        gt=0,
    )
    reflection_depth: int = Field(
        default=1,
        description="Number of past reflections to consider",
        ge=0,
    )
```

### Memory Management

The Reflexion Critic uses a `MemoryManager` to store and retrieve past reflections:

```python
class MemoryManager:
    """Manages memory for reflection-based critics."""

    def __init__(self, buffer_size: int = 5):
        """Initialize the memory manager."""
        self.buffer_size = buffer_size
        self.memory = []

    def add_to_memory(self, item: str) -> None:
        """Add an item to memory."""
        self.memory.append(item)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def get_memory(self, max_items: Optional[int] = None) -> List[str]:
        """Get items from memory."""
        if max_items is None or max_items >= len(self.memory):
            return self.memory.copy()
        return self.memory[-max_items:]

    def clear_memory(self) -> None:
        """Clear all memory."""
        self.memory = []
```

### Prompt Management

The Reflexion Critic uses a specialized prompt manager that incorporates past reflections:

```python
class ReflexionCriticPromptManager(PromptManager):
    """Manages prompts for reflexion critics."""

    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        """Create a prompt for improving text with feedback and reflections."""
        # Get reflections from memory
        reflections = self.memory_manager.get_memory(self.config.reflection_depth)

        # Format reflections if available
        reflections_text = ""
        if reflections:
            reflections_text = "PREVIOUS REFLECTIONS:\n" + "\n".join(reflections) + "\n\n"

        # Create improvement prompt
        return f"""Please improve the following text based on the feedback and previous reflections:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}

        {reflections_text}IMPROVED TEXT:"""
```

### Core Methods

The Reflexion Critic implements three core methods:

1. **validate**: Determines if text meets quality standards
2. **critique**: Analyzes text and provides detailed feedback
3. **improve**: Generates an improved version of text based on feedback and reflections

Each method delegates to a `CritiqueService` that handles the interaction with the language model:

```python
def improve(self, text: str, feedback: str = None) -> str:
    """Improve text based on feedback and reflections."""
    self._check_input(text)
    feedback_str = self._format_feedback(feedback)
    state = self._state_manager.get_state()
    return state.cache["critique_service"].improve(text, feedback_str)
```

### Factory Function

The Reflexion Critic provides a factory function for easy creation:

```python
def create_reflexion_critic(
    llm_provider: LanguageModel,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
    config: Optional[Union[Dict[str, Any], ReflexionCriticConfig]] = None,
) -> ReflexionCritic:
    """Create a reflexion critic with the given parameters."""
    # Implementation details...
```

## Integration with Sifaka

The Reflexion Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added a default configuration `DEFAULT_REFLEXION_CONFIG`
4. Provided comprehensive tests in `tests/critics/test_reflexion.py`
5. Provided examples in `examples/critics/reflexion_critic_example.py`

## Component Interactions

The Reflexion Critic interacts with several components:

1. **Language Model Provider**: Generates responses based on prompts
2. **Prompt Manager**: Creates and formats prompts that incorporate reflections
3. **Response Parser**: Parses responses from language models
4. **Memory Manager**: Stores and retrieves past reflections
5. **Critique Service**: Coordinates the critique and improvement process

This component-based architecture allows for flexible customization and extension of the critic's capabilities.

## Testing

The Reflexion Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of text
3. Critique generation
4. Text improvement with reflections
5. Memory management
6. Factory function behavior
7. Error handling

## Future Improvements

Potential future improvements for the Reflexion Critic include:

1. Adding support for more sophisticated memory management
2. Implementing a more robust parsing of reflections
3. Adding support for different types of reflections
4. Implementing a more sophisticated scoring mechanism
5. Adding support for tracking the history of improvements

## References

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Sifaka Critics Documentation](../components/critics.md)
