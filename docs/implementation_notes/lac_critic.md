# Implementation Notes: LAC Critic

This document provides implementation details and notes for the LAC (LLM-Based Actor-Critic) Critic in the Sifaka project.

## Overview

The LAC Critic implements the LLM-Based Actor-Critic approach, which combines language feedback and value scoring to improve language model-based decision making.

Based on: [Language Feedback Improves Language Model-based Decision Making](https://arxiv.org/abs/2403.03692)

## Architecture

The LAC Critic combines two specialized critics:

1. **FeedbackCritic**: Produces natural language feedback for a model's response
2. **ValueCritic**: Estimates a numeric value for a model's response

These critics work together in the `LACCritic` class to provide both qualitative and quantitative feedback.

## Implementation Details

### State Management

Each critic uses direct state management with a `CriticState` object:

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.cache = {
    "feedback_prompt_template": config.feedback_prompt_template,
    "value_prompt_template": config.value_prompt_template,
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
}
self._state.initialized = True
```

### Configuration

The LAC Critic uses three configuration classes:

1. **FeedbackCriticConfig**:
```python
class FeedbackCriticConfig(CriticConfig):
    feedback_prompt_template: str = Field(
        default=DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        description="Template for feedback prompts",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
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
```

2. **ValueCriticConfig**:
```python
class ValueCriticConfig(CriticConfig):
    value_prompt_template: str = Field(
        default=DEFAULT_VALUE_PROMPT_TEMPLATE,
        description="Template for value prompts",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
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
    min_score: float = Field(
        default=0.0,
        description="Minimum possible score",
        ge=0.0,
        le=1.0,
    )
    max_score: float = Field(
        default=1.0,
        description="Maximum possible score",
        ge=0.0,
        le=1.0,
    )
```

3. **LACCriticConfig**:
```python
class LACCriticConfig(CriticConfig):
    feedback_critic_config: FeedbackCriticConfig
    value_critic_config: ValueCriticConfig
```

### Core Methods

Each critic implements these core methods:

1. **FeedbackCritic**:
```python
def run(self, task: str, response: str) -> str
async def arun(self, task: str, response: str) -> str
def validate(self, text: str) -> bool
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def improve_with_feedback(self, text: str, feedback: str) -> str
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

2. **ValueCritic**:
```python
def run(self, task: str, response: str) -> float
def validate(self, text: str) -> bool
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

3. **LACCritic**:
```python
def run(self, task: str, response: str) -> Dict[str, Any]
def validate(self, text: str) -> bool
def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str
def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

### Factory Functions

The LAC Critic provides three factory functions:

1. **create_feedback_critic**:
```python
def create_feedback_critic(
    llm_provider: Any,
    name: str = "feedback_critic",
    description: str = "Provides natural language feedback for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic
```

2. **create_value_critic**:
```python
def create_value_critic(
    llm_provider: Any,
    name: str = "value_critic",
    description: str = "Provides numeric value scoring for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    value_prompt_template: str = None,
    min_score: float = None,
    max_score: float = None,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic
```

3. **create_lac_critic**:
```python
def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    value_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic
```

## Usage Example

```python
from sifaka.critics.implementations.lac import create_lac_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a LAC critic
critic = create_lac_critic(llm_provider=provider)

# Use the critic to improve text
task = "Summarize the causes of World War I in 3 bullet points."
response = provider.generate(f"Task:\n{task}")
results = critic.critique(response, {"task": task})

print("Feedback:", results["feedback"])
print("Value Score:", results["value"])
```

## Error Handling

The LAC Critic handles these error cases:

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

The LAC Critic includes comprehensive tests that verify:

1. Initialization of all three critic types
2. Feedback generation
3. Value scoring
4. Combined feedback and scoring
5. Factory function behavior
6. Error handling
7. Async method behavior

## Future Improvements

Potential future improvements for the LAC Critic include:

1. Adding support for more sophisticated value scoring
2. Implementing feedback aggregation
3. Adding support for multi-step feedback
4. Implementing feedback history tracking
5. Adding support for custom scoring ranges

## References

- [Language Feedback Improves Language Model-based Decision Making](https://arxiv.org/abs/2403.03692)
- [Sifaka Critics Documentation](../components/critics.md)
