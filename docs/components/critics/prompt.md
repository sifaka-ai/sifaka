# Prompt Critic

The Prompt Critic is a fundamental critic implementation that uses language models to evaluate, validate, and improve text outputs. It serves as the foundation for many other critic types in Sifaka.

## Overview

The Prompt Critic uses carefully crafted prompts to guide language models in providing feedback and improvements for text. It follows a simple but effective approach:

1. **Validate**: Determine if the text meets quality standards
2. **Critique**: Analyze the text and provide detailed feedback
3. **Improve**: Generate an improved version of the text based on feedback

This approach allows for flexible text evaluation and improvement using the capabilities of modern language models.

## Usage

### Basic Usage

```python
from sifaka.critics import create_prompt_critic
from sifaka.models.openai import create_openai_provider

# Create a language model provider
provider = create_openai_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Create a prompt critic
critic = create_prompt_critic(
    llm_provider=provider,
    name="example_critic",
    description="A critic for improving technical documentation",
    system_prompt="You are an expert technical writer with experience in creating clear, concise documentation."
)

# Validate text
is_valid = critic.validate("This is a brief explanation of the feature.")

# Get critique of text
critique = critic.critique("This is a brief explanation of the feature.")
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")
print(f"Issues: {critique['issues']}")
print(f"Suggestions: {critique['suggestions']}")

# Improve text based on feedback
improved_text = critic.improve("This is a brief explanation of the feature.", critique['feedback'])
```

### Configuration Options

The Prompt Critic supports the following configuration options:

- `system_prompt`: System prompt for the model (default: "You are an expert at evaluating and improving text.")
- `temperature`: Temperature for model generation (default: 0.7)
- `max_tokens`: Maximum tokens for model generation (default: 1000)
- `min_confidence`: Minimum confidence threshold (default: 0.7)
- `max_attempts`: Maximum improvement attempts (default: 3)

### Custom Prompt Templates

You can customize the prompt templates used for validation, critique, and improvement:

```python
from sifaka.critics.models import PromptCriticConfig

config = PromptCriticConfig(
    name="custom_critic",
    description="A critic with custom prompts",
    system_prompt="You are an expert editor.",
    validation_prompt_template="Is the following text valid? Text: {text}",
    critique_prompt_template="Please critique the following text: {text}",
    improvement_prompt_template="Please improve this text: {text} based on this feedback: {feedback}"
)

critic = create_prompt_critic(
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

The Prompt Critic supports asynchronous operations:

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

The Prompt Critic uses a component-based architecture:

1. **PromptManager**: Creates and formats prompts for different operations
2. **ResponseParser**: Parses responses from language models
3. **CritiqueService**: Coordinates the critique and improvement process

This architecture allows for flexible customization and extension of the critic's capabilities.

## References

- [Sifaka Critics Documentation](../critics.md)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
