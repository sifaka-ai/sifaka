# Implementation Notes: Prompt Critic

This document provides implementation details and notes for the Prompt Critic in the Sifaka project.

## Overview

The Prompt Critic uses language models to evaluate, validate, and improve text outputs based on rule violations. It analyzes text for clarity, ambiguity, completeness, and effectiveness using a language model to generate feedback and validation scores.

## Architecture

The PromptCritic follows a component-based architecture with clear separation of concerns:

1. **Core Components**
   - **PromptCritic**: Main class that implements the critic interfaces
   - **CritiqueService**: Service that handles the core critique functionality
   - **PromptManager**: Manages prompt creation and formatting
   - **ResponseParser**: Parses and validates model responses
   - **MemoryManager**: Manages history of improvements and critiques

## Implementation Details

### Component Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Factory initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Text improvement
   - Feedback processing

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

### State Management

The PromptCritic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()
self._state.initialized = False

# Store components in state
self._state.model = llm_provider
self._state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
self._state.response_parser = ResponseParser()
self._state.memory_manager = MemoryManager(buffer_size=10)

# Create services and store in state cache
self._state.cache["critique_service"] = CritiqueService(
    llm_provider=llm_provider,
    prompt_manager=self._state.prompt_manager,
    response_parser=self._state.response_parser,
    memory_manager=self._state.memory_manager,
)

# Mark as initialized
self._state.initialized = True
```

### Core Methods

The PromptCritic implements these core methods:

1. **Text Improvement**:
```python
def improve(self, text: str, feedback: str = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str
def improve_with_history(self, text: str, feedback: str = None) -> Tuple[str, List[Dict[str, Any]]]
async def aimprove(self, text: str, feedback: str = None) -> str
```

2. **Feedback Loop**:
```python
def close_feedback_loop(self, text: str, generator_response: str, feedback: str = None) -> Tuple[str, Dict[str, Any]]
async def aclose_feedback_loop(self, text: str, generator_response: str, feedback: str = None) -> Tuple[str, Dict[str, Any]]
```

3. **Validation and Critique**:
```python
def validate(self, text: str) -> bool
def critique(self, text: str) -> dict
```

### Factory Function

The PromptCritic provides a factory function for easy creation:

```python
def create_prompt_critic(
    llm_provider: Any,
    name: str = "prompt_critic",
    description: str = "A critic that uses prompts to improve text",
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    prompt_factory: Optional[Any] = None,
    config: Optional[PromptCriticConfig] = None,
) -> PromptCritic
```

## Usage Example

```python
from sifaka.critics.implementations.prompt import create_prompt_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a prompt critic
critic = create_prompt_critic(llm_provider=provider)

# Use the critic to improve text
text = "The quick brown fox jumps over the lazy dog."
improved_text = critic.improve(text, "Make the text more descriptive")

# Get improvement history
improved_text, history = critic.improve_with_history(text)
```

## Error Handling

The PromptCritic handles these error cases:

1. **Initialization Errors**
   - Missing required parameters
   - Invalid provider type
   - Invalid configuration values

2. **Validation Errors**
   - Empty text
   - Invalid feedback format
   - Uninitialized critic

3. **Generation Errors**
   - Model provider failures
   - Invalid prompt formatting
   - Response parsing errors

## Testing

The PromptCritic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Text improvement functionality
3. Feedback loop closure
4. History tracking
5. Async method behavior
6. Error handling
7. Memory management

## Future Improvements

Potential future improvements for the PromptCritic include:

1. Adding support for more sophisticated prompt templates
2. Implementing parallel processing for multiple improvements
3. Adding support for custom improvement strategies
4. Implementing more advanced memory management
5. Adding support for streaming responses

## References

- [Sifaka Critics Documentation](../components/critics.md)
- [Prompt Engineering Best Practices](../best_practices/prompt_engineering.md)
