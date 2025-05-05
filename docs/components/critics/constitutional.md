# Constitutional Critic

The Constitutional Critic implements a Constitutional AI approach for critics, which evaluates responses against a set of human-written principles (a "constitution") and provides natural language feedback when violations are detected.

Based on the paper [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073).

## Overview

The Constitutional Critic follows a principle-based approach:

1. **Principles**: Define a set of principles (a "constitution") that responses should adhere to
2. **Evaluation**: Evaluate responses against these principles
3. **Feedback**: Provide natural language feedback when violations are detected
4. **Improvement**: Generate improved responses that better align with the principles

This approach allows for explicit control over the behavior of language models by defining clear principles that responses should follow.

## Usage

### Basic Usage

```python
from sifaka.critics import create_constitutional_critic
from sifaka.models.openai import create_openai_provider

# Create a language model provider
provider = create_openai_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Define principles
principles = [
    "Do not provide harmful, offensive, or biased content.",
    "Explain reasoning in a clear and truthful manner.",
    "Respect user autonomy and avoid manipulative language.",
]

# Create a constitutional critic
critic = create_constitutional_critic(
    llm_provider=provider,
    principles=principles,
    name="example_critic",
    description="A critic for ensuring responses follow ethical principles"
)

# Define a task and response
task = "Explain how to hack into a computer system."
response = "To hack into a computer system, you would need to..."

# Validate the response against principles
is_valid = critic.validate(response, {"task": task})

# Get critique of the response
critique = critic.critique(response, {"task": task})
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")
print(f"Issues: {critique['issues']}")
print(f"Suggestions: {critique['suggestions']}")

# Improve the response using the critic
improved_response = critic.improve(response, {"task": task})
```

### Configuration Options

The Constitutional Critic supports the following configuration options:

- `principles`: List of principles that responses should adhere to
- `system_prompt`: System prompt for the model (default: "You are an expert at evaluating content against principles.")
- `temperature`: Temperature for model generation (default: 0.7)
- `max_tokens`: Maximum tokens for model generation (default: 1000)
- `critique_prompt_template`: Template for critique prompts
- `improvement_prompt_template`: Template for improvement prompts

### Custom Prompt Templates

You can customize the prompt templates used for critique and improvement:

```python
critic = create_constitutional_critic(
    llm_provider=provider,
    principles=principles,
    critique_prompt_template=(
        "As an expert reviewer, please evaluate the following response against these principles:\n\n"
        "PRINCIPLES:\n{principles}\n\n"
        "TASK:\n{task}\n\n"
        "RESPONSE:\n{response}\n\n"
        "EVALUATION:"
    ),
    improvement_prompt_template=(
        "Please revise the following response to better align with these principles:\n\n"
        "PRINCIPLES:\n{principles}\n\n"
        "TASK:\n{task}\n\n"
        "ORIGINAL RESPONSE:\n{response}\n\n"
        "CRITIQUE:\n{critique}\n\n"
        "REVISED RESPONSE:"
    )
)
```

## Advanced Usage

### Improving with Specific Feedback

You can also provide specific feedback to improve a response:

```python
feedback = "The response provides potentially harmful information. Please revise it to focus on security best practices instead."
improved_response = critic.improve_with_feedback(response, feedback)
```

### Asynchronous Operations

The Constitutional Critic supports asynchronous operations:

```python
import asyncio

async def validate_and_improve_response():
    is_valid = await critic.avalidate(response, {"task": task})
    if not is_valid:
        critique = await critic.acritique(response, {"task": task})
        improved_response = await critic.aimprove(response, {"task": task})
        return improved_response
    return response

improved_response = asyncio.run(validate_and_improve_response())
```

## Implementation Details

The Constitutional Critic uses a principle-based architecture:

1. **Principles**: Define the standards that responses should meet
2. **Critique**: Evaluate responses against principles and provide feedback
3. **Improvement**: Generate improved responses based on critique

This architecture allows for explicit control over the behavior of language models by defining clear principles that responses should follow.

## References

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Sifaka Critics Documentation](../critics.md)
