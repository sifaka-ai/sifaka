# Implementation Notes: Reflexion Critic

This document provides implementation details and notes for the Reflexion Critic in the Sifaka project.

## Overview

The Reflexion Critic implements the Reflexion approach, which enables language model agents to learn from feedback without requiring weight updates. It maintains reflections in memory to improve future text generation.

## Architecture

The ReflexionCritic follows a component-based architecture with memory-augmented generation:

1. **Core Components**
   - **ReflexionCritic**: Main class that implements the critic interfaces
   - **CritiqueService**: Service that handles the core critique functionality
   - **ReflexionCriticPromptManager**: Creates prompts with reflection context
   - **ResponseParser**: Parses and validates model responses
   - **MemoryManager**: Manages the memory buffer of past reflections

## Implementation Details

### State Management

The ReflexionCritic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
self._state.response_parser = ResponseParser()
self._state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

# Create service and store in state cache
self._state.cache["critique_service"] = CritiqueService(
    llm_provider=llm_provider,
    prompt_manager=self._state.prompt_manager,
    response_parser=self._state.response_parser,
    memory_manager=self._state.memory_manager,
)

# Mark as initialized
self._state.initialized = True
```

### Prompt Templates

The ReflexionCritic uses specialized prompt templates for different operations:

1. **Validation Prompt**:
```
Please validate the following text:

TEXT TO VALIDATE:
{text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
VALID: [true/false]
REASON: [reason for validation result]

VALIDATION:
```

2. **Critique Prompt**:
```
Please critique the following text:

TEXT TO CRITIQUE:
{text}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
SCORE: [number between 0 and 1]
FEEDBACK: [your general feedback]
ISSUES:
- [issue 1]
- [issue 2]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

CRITIQUE:
```

3. **Improvement Prompt**:
```
Please improve the following text:

TEXT TO IMPROVE:
{text}

FEEDBACK:
{feedback}

PREVIOUS REFLECTIONS:
{reflections}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
IMPROVED_TEXT: [improved text]

IMPROVEMENT:
```

4. **Reflection Prompt**:
```
Please reflect on the following text improvement process:

ORIGINAL TEXT:
{text}

FEEDBACK RECEIVED:
{feedback}

IMPROVED TEXT:
{improved_text}

Reflect on what went well, what went wrong, and what could be improved in future iterations.
Focus on specific patterns, mistakes, or strategies that could be applied to similar tasks.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
REFLECTION: [your reflection]

REFLECTION:
```

### Core Methods

The ReflexionCritic implements these core methods:

1. **Validation and Critique**:
```python
def validate(self, text: str) -> bool
def critique(self, text: str) -> dict
async def avalidate(self, text: str) -> bool
async def acritique(self, text: str) -> dict
```

2. **Text Improvement**:
```python
def improve(self, text: str, feedback: str = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str
async def aimprove(self, text: str, feedback: str = None) -> str
async def aimprove_with_feedback(self, text: str, feedback: str) -> str
```

### Factory Function

The ReflexionCritic provides a factory function for easy creation:

```python
def create_reflexion_critic(
    llm_provider: Any,
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
) -> ReflexionCritic
```

## Usage Example

```python
from sifaka.critics.implementations.reflexion import create_reflexion_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a reflexion critic
critic = create_reflexion_critic(llm_provider=provider)

# Improve text with feedback
text = "This is a sample technical document."
feedback = "The text needs more detail and better structure."
improved_text = critic.improve(text, feedback)
```

## Error Handling

The ReflexionCritic handles these error cases:

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

The ReflexionCritic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Text validation
3. Feedback generation
4. Text improvement with reflections
5. Memory management
6. Async method behavior
7. Error handling

## Future Improvements

Potential future improvements for the ReflexionCritic include:

1. Adding support for more sophisticated reflection strategies
2. Implementing parallel processing for multiple improvements
3. Adding support for custom reflection templates
4. Implementing more advanced memory management
5. Adding support for streaming responses

## References

- [Sifaka Critics Documentation](../components/critics.md)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
