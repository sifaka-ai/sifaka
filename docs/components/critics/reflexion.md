# Reflexion Critic

The Reflexion Critic implements the Reflexion approach for critics, which enables language model agents to learn from feedback without requiring weight updates. It maintains reflections in memory to improve future text generation.

## Overview

The Reflexion Critic follows a memory-based approach:

1. **Memory Management**: Maintains a buffer of past reflections and feedback
2. **Reflection**: Uses past reflections to inform current text generation
3. **Improvement**: Generates improved text based on feedback and reflections

This approach allows the model to "learn" from past feedback and improve its outputs over time, without requiring any changes to the model weights.

## Usage

### Basic Usage

```python
from sifaka.critics import create_reflexion_critic
from sifaka.models.openai import create_openai_provider

# Create a language model provider
provider = create_openai_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Create a reflexion critic
critic = create_reflexion_critic(
    llm_provider=provider,
    name="example_critic",
    description="A critic for improving technical documentation",
    system_prompt="You are an expert technical writer with experience in creating clear, concise documentation.",
    memory_buffer_size=5,
    reflection_depth=2
)

# Define a task and initial output
text = "This is a brief explanation of the feature."

# Get critique of the text
critique = critic.critique(text)
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")
print(f"Issues: {critique['issues']}")
print(f"Suggestions: {critique['suggestions']}")

# Improve the text using the critic
improved_text = critic.improve(text, critique['feedback'])
```

### Configuration Options

The Reflexion Critic supports the following configuration options:

- `system_prompt`: System prompt for the model (default: "You are an expert editor that learns from past feedback.")
- `temperature`: Temperature for model generation (default: 0.7)
- `max_tokens`: Maximum tokens for model generation (default: 1000)
- `memory_buffer_size`: Size of the memory buffer for storing reflections (default: 5)
- `reflection_depth`: Number of past reflections to consider (default: 1)

### Custom Prompt Templates

You can customize the prompt templates used for critique and improvement:

```python
from sifaka.critics.models import ReflexionCriticConfig

config = ReflexionCriticConfig(
    name="custom_critic",
    description="A critic with custom prompts",
    system_prompt="You are an expert editor.",
    memory_buffer_size=10,
    reflection_depth=3,
    validation_prompt_template="Is the following text valid? Text: {text}",
    critique_prompt_template="Please critique the following text: {text}",
    improvement_prompt_template="Please improve this text: {text} based on this feedback: {feedback}"
)

critic = create_reflexion_critic(
    llm_provider=provider,
    config=config
)
```

## Advanced Usage

### Improving with Specific Feedback

You can provide specific feedback to improve text:

```python
feedback = "The explanation is too technical. Please simplify it for a general audience."
improved_text = critic.improve_with_feedback(text, feedback)
```

### Asynchronous Operations

The Reflexion Critic supports asynchronous operations:

```python
import asyncio

async def validate_and_improve_text():
    is_valid = await critic.avalidate(text)
    if not is_valid:
        critique = await critic.acritique(text)
        improved_text = await critic.aimprove(text, critique['feedback'])
        return improved_text
    return text

improved_text = asyncio.run(validate_and_improve_text())
```

## Implementation Details

The Reflexion Critic uses a memory-based architecture:

1. **Memory Manager**: Stores and retrieves past reflections and feedback
2. **Prompt Manager**: Creates prompts that incorporate past reflections
3. **Response Parser**: Parses responses from language models
4. **Critique Service**: Coordinates the critique and improvement process

This architecture allows the critic to "learn" from past feedback and improve its outputs over time.

## References

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Sifaka Critics Documentation](../critics.md)
