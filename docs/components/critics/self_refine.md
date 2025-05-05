# Self-Refine Critic

The Self-Refine Critic implements the Self-Refine approach for iterative self-improvement of text. This critic uses the same language model to critique and revise its own outputs in multiple iterations, leading to progressively improved results.

Based on the paper [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651).

## Overview

The Self-Refine Critic follows a simple but effective approach:

1. **Critique**: The model critiques its own output, identifying issues and suggesting improvements.
2. **Revise**: The model revises the output based on the critique.
3. **Repeat**: Steps 1 and 2 are repeated for a specified number of iterations or until no further improvements are needed.

This approach allows the model to iteratively improve its output without requiring external feedback or multiple models.

## Usage

### Basic Usage

```python
from sifaka.critics import create_self_refine_critic
from sifaka.models.openai import create_openai_provider

# Create a language model provider
provider = create_openai_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Create a self-refine critic
critic = create_self_refine_critic(
    llm_provider=provider,
    name="example_critic",
    description="A critic for improving explanations",
    max_iterations=3,
    system_prompt="You are an expert at explaining complex concepts clearly and concisely."
)

# Define a task and initial output
task = "Explain quantum computing to a high school student."
initial_output = "Quantum computing uses qubits instead of regular bits. Qubits can be in multiple states at once due to superposition."

# Improve the output using the critic
improved_output = critic.improve(initial_output, {"task": task})

# Get critique of the improved output
critique = critic.critique(improved_output, {"task": task})
```

### Configuration Options

The Self-Refine Critic supports the following configuration options:

- `max_iterations`: Maximum number of refinement iterations (default: 3)
- `system_prompt`: System prompt for the model (default: "You are an expert at critiquing and revising content.")
- `temperature`: Temperature for model generation (default: 0.7)
- `max_tokens`: Maximum tokens for model generation (default: 1000)
- `critique_prompt_template`: Template for critique prompts
- `revision_prompt_template`: Template for revision prompts

### Custom Prompt Templates

You can customize the prompt templates used for critique and revision:

```python
critic = create_self_refine_critic(
    llm_provider=provider,
    critique_prompt_template=(
        "As an expert reviewer, please critique the following response:\n\n"
        "Task:\n{task}\n\n"
        "Response:\n{response}\n\n"
        "Critique:"
    ),
    revision_prompt_template=(
        "Please revise the following response based on the critique:\n\n"
        "Task:\n{task}\n\n"
        "Original Response:\n{response}\n\n"
        "Critique:\n{critique}\n\n"
        "Revised Response:"
    )
)
```

## Advanced Usage

### Improving with Specific Feedback

You can also provide specific feedback to improve text:

```python
feedback = "The explanation is too technical. Please simplify it for a high school student."
improved_text = critic.improve_with_feedback(text, feedback)
```

### Asynchronous Operations

The Self-Refine Critic supports asynchronous operations:

```python
import asyncio

async def improve_text():
    result = await critic.aimprove(text, {"task": task})
    return result

improved_text = asyncio.run(improve_text())
```

## Implementation Details

The Self-Refine Critic uses a simple but effective algorithm:

1. Generate a critique of the current text.
2. If the critique indicates no issues, return the current text.
3. Generate a revised version of the text based on the critique.
4. If the revised text is the same as the current text, return the current text.
5. Update the current text to the revised text.
6. Repeat steps 1-5 for a specified number of iterations or until no further improvements are needed.

This approach allows the model to iteratively improve its output without requiring external feedback or multiple models.

## References

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Sifaka Critics Documentation](../critics.md)
