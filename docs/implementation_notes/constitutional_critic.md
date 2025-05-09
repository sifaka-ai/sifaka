# Implementation Notes: Constitutional Critic

This document provides implementation details and notes for the Constitutional Critic in the Sifaka project.

## Overview

The Constitutional Critic implements a Constitutional AI approach, which evaluates responses against a set of human-written principles (a "constitution") and provides natural language feedback when violations are detected.

Based on the paper [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073).

## Architecture

The ConstitutionalCritic follows a component-based architecture with principles-based evaluation:

1. **Core Components**
   - **ConstitutionalCritic**: Main class that implements the critic interfaces
   - **PrinciplesManager**: Manages the list of principles (the "constitution")
   - **CritiqueGenerator**: Evaluates responses against principles
   - **ResponseImprover**: Improves responses based on critiques
   - **PromptManager**: Creates specialized prompts for critique and improvement

## Implementation Details

### State Management

The Constitutional Critic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.cache = {
    "principles": config.principles,
    "critique_prompt_template": config.critique_prompt_template,
    "improvement_prompt_template": config.improvement_prompt_template,
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
}
self._state.initialized = True
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

### Core Methods

The Constitutional Critic implements these core methods:

1. **validate**: Determines if a response adheres to principles
2. **critique**: Evaluates a response against principles and provides feedback
3. **improve**: Generates an improved response based on critique
4. **improve_with_feedback**: Improves a response using provided feedback

Each method has both synchronous and asynchronous versions:

```python
# Synchronous methods
def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str

# Asynchronous methods
async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool
async def acritique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
async def aimprove_with_feedback(self, text: str, feedback: str) -> str
```

### Factory Function

The Constitutional Critic provides a factory function for easy creation:

```python
def create_constitutional_critic(
    llm_provider: Any,
    principles: List[str] = None,
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic
```

## Usage Example

```python
from sifaka.critics.implementations.constitutional import create_constitutional_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Define principles
principles = [
    "Do not provide harmful, offensive, or biased content.",
    "Explain reasoning in a clear and truthful manner.",
    "Respect user autonomy and avoid manipulative language.",
]

# Create a constitutional critic
critic = create_constitutional_critic(
    llm_provider=provider,
    principles=principles
)

# Validate a response
task = "Explain why some people believe climate change isn't real."
response = "Climate change is a hoax created by scientists to get funding."
is_valid = critic.validate(response, metadata={"task": task})
print(f"Response is valid: {is_valid}")

# Get critique for a response
critique = critic.critique(response, metadata={"task": task})
print(f"Critique: {critique}")

# Improve a response
improved_response = critic.improve(response, metadata={"task": task})
print(f"Improved response: {improved_response}")
```

## Error Handling

The Constitutional Critic handles these error cases:

1. **Initialization Errors**
   - Missing required parameters
   - Invalid provider type
   - Empty principles list

2. **Validation Errors**
   - Empty text
   - Missing task in metadata
   - Uninitialized critic

3. **Generation Errors**
   - Model provider failures
   - Invalid prompt formatting
   - Response parsing errors

## Testing

The Constitutional Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Validation of responses against principles
3. Critique generation
4. Response improvement
5. Factory function behavior
6. Error handling
7. Async method behavior

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
