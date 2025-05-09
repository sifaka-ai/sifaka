# Implementation Notes: Self-Refine Critic

This document provides implementation details and notes for the Self-Refine Critic in the Sifaka project.

## Overview

The Self-Refine Critic implements the Self-Refine approach, which enables language models to iteratively critique and revise their own outputs without requiring external feedback. The critic uses the same language model to generate critiques and revisions in multiple rounds.

Based on: [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

## Architecture

The SelfRefineCritic follows a component-based architecture with iterative refinement:

1. **Core Components**
   - **SelfRefineCritic**: Main class that implements the critic interfaces
   - **PromptManager**: Creates prompts for critique and revision
   - **ResponseParser**: Parses and validates model responses
   - **MemoryManager**: Manages history of critiques and revisions

## Implementation Details

### Lifecycle Management

The SelfRefineCritic manages its lifecycle through three main phases:

1. **Initialization**
   - Validates configuration
   - Sets up language model provider
   - Initializes state
   - Allocates resources

2. **Operation**
   - Validates text
   - Critiques text
   - Improves text through multiple iterations
   - Tracks improvements

3. **Cleanup**
   - Releases resources
   - Resets state
   - Logs final status

### State Management

The SelfRefineCritic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.cache = {
    "max_iterations": config.max_iterations,
    "critique_prompt_template": config.critique_prompt_template or (
        "Please critique the following response to the task. "
        "Focus on accuracy, clarity, and completeness.\n\n"
        "Task:\n{task}\n\n"
        "Response:\n{response}\n\n"
        "Critique:"
    ),
    "revision_prompt_template": config.revision_prompt_template or (
        "Please revise the following response based on the critique.\n\n"
        "Task:\n{task}\n\n"
        "Response:\n{response}\n\n"
        "Critique:\n{critique}\n\n"
        "Revised response:"
    ),
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
}
self._state.initialized = True
```

### Core Methods

The SelfRefineCritic implements these core methods:

1. **Validation and Critique**:
```python
def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
async def acritique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

2. **Text Improvement**:
```python
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str
async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
async def aimprove_with_feedback(self, text: str, feedback: str) -> str
```

### Factory Function

The SelfRefineCritic provides a factory function for easy creation:

```python
def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    max_iterations: int = None,
    critique_prompt_template: Optional[str] = None,
    revision_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRefineCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRefineCritic
```

## Usage Example

```python
from sifaka.critics.implementations.self_refine import create_self_refine_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a self-refine critic
critic = create_self_refine_critic(
    llm_provider=provider,
    max_iterations=3
)

# Use the critic to improve text
task = "Write a concise explanation of quantum computing."
initial_output = "Quantum computing uses quantum bits."
improved_output = critic.improve(initial_output, {"task": task})
```

## Error Handling

The SelfRefineCritic handles these error cases:

1. **Initialization Errors**
   - Missing required parameters
   - Invalid provider type
   - Invalid configuration values

2. **Validation Errors**
   - Empty text
   - Missing task in metadata
   - Uninitialized critic

3. **Generation Errors**
   - Model provider failures
   - Invalid prompt formatting
   - Response parsing errors

## Testing

The SelfRefineCritic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Text validation
3. Critique generation
4. Iterative improvement
5. Async method behavior
6. Error handling
7. Memory management

## Future Improvements

Potential future improvements for the SelfRefineCritic include:

1. Adding support for more sophisticated critique strategies
2. Implementing parallel processing for multiple iterations
3. Adding support for custom critique templates
4. Implementing more advanced memory management
5. Adding support for streaming responses

## References

- [Sifaka Critics Documentation](../components/critics.md)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
