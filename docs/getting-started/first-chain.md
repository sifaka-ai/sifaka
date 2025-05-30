# Your First Sifaka Chain

Learn how to create and run your first Sifaka chain in just a few minutes.

## 🚨 **Current Status (v0.2.1)**

**Important**: This guide reflects current limitations in Sifaka 0.2.1:
- **⚠️ HuggingFace Models**: Temporarily disabled due to PydanticAI dependency conflicts
- **⚠️ Guardrails AI**: Temporarily disabled due to griffe version incompatibility
- **✅ Supported Models**: OpenAI, Anthropic, Gemini, Ollama, Mock models work fully

## Chain Types

Sifaka offers two chain implementations:

- **🚀 PydanticAI Chain**: Modern, tool-focused, simple setup (**recommended for new projects**)
- **🏗️ Traditional Chain**: Feature-rich, production-ready (**deprecated but still available**)

This guide covers **both approaches**, starting with the recommended **PydanticAI Chain**.

## What You'll Build

By the end of this guide, you'll have created a working Sifaka chain that:
- Generates text using an AI model
- Validates the output for quality
- Improves the text through iterative feedback
- Tracks the complete process

## Prerequisites

- Sifaka installed ([installation guide](installation.md))
- An API key for OpenAI or Anthropic (or use our mock model for testing)

## Quick Start (30 seconds)

The fastest way to get started is with QuickStart:

```python
from sifaka.quickstart import QuickStart

# Create a chain with one line
chain = QuickStart.for_production(
    "openai:gpt-4",  # Requires OPENAI_API_KEY in environment
    "Write a short story about a robot learning to help humans.",
    validators=["length"],
    critics=["reflexion"]
)

# Run it
result = chain.run()
print(f"Generated text: {result.text}")
```

## 🚀 PydanticAI Chain (Recommended)

The modern approach using PydanticAI agents with tool calling:

### Step 1: Create a PydanticAI Agent

```python
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic
from sifaka.models import create_model

# Create PydanticAI agent
agent = Agent("openai:gpt-4", system_prompt="You are a creative writer.")

# Add tools (optional)
@agent.tool_plain
def search_web(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"
```

### Step 2: Create Sifaka Chain

```python
# Create Sifaka components
validator = LengthValidator(min_length=50, max_length=500)
critic = ReflexionCritic(model=create_model("openai:gpt-4"))

# Create hybrid chain
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic],
    always_apply_critics=True
)
```

### Step 3: Run the Chain

```python
# Run with runtime prompt
result = chain.run("Write a short story about a robot learning to help humans.")
print(f"Generated text: {result.text}")
print(f"Iterations: {result.iteration}")
```

## 🏗️ Traditional Chain (Deprecated)

> **⚠️ Note**: Traditional Chain is deprecated as of v0.2.0. Use PydanticAI Chain for new projects. This section is maintained for existing users.

### Step 1: Import Sifaka

```python
from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic
```

### Step 2: Create a Model

```python
# Option 1: OpenAI (requires OPENAI_API_KEY)
model = create_model("openai:gpt-4")

# Option 2: Anthropic (requires ANTHROPIC_API_KEY)
model = create_model("anthropic:claude-3-sonnet")

# Option 3: Mock model (no API key needed)
model = create_model("mock:test-model")
```

### Step 3: Create Validators (Optional)

Validators check if the generated text meets your requirements:

```python
# Ensure text is between 50-500 characters
length_validator = LengthValidator(min_length=50, max_length=500)
```

### Step 4: Create Critics (Optional)

Critics provide feedback to improve the text:

```python
# Reflexion critic analyzes and suggests improvements
critic = ReflexionCritic(model=model)
```

### Step 5: Build the Chain

```python
chain = Chain(
    model=model,
    prompt="Write a short story about a robot learning to help humans.",
    max_improvement_iterations=2
)

# Add validators and critics
chain.validate_with(length_validator)
chain.improve_with(critic)
```

### Step 6: Run the Chain

```python
# Execute the chain
result = chain.run()

# Access the results
print(f"Final text: {result.text}")
print(f"Iterations: {result.iteration}")
print(f"Validation passed: {result.validation_results}")
```

## Complete Example

Here's a complete working example you can run:

```python
#!/usr/bin/env python3
"""My first Sifaka chain."""

from sifaka import Chain
from sifaka.models import create_model
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic

def main():
    # Create model (using mock for this example)
    model = create_model("mock:demo-model")

    # Create validator
    length_validator = LengthValidator(min_length=50, max_length=500)

    # Create critic
    critic = ReflexionCritic(model=model)

    # Build chain
    chain = Chain(
        model=model,
        prompt="Write a short story about a robot learning to help humans.",
        max_improvement_iterations=2
    )

    # Configure chain
    chain.validate_with(length_validator)
    chain.improve_with(critic)

    # Run chain
    print("Running your first Sifaka chain...")
    result = chain.run()

    # Show results
    print(f"\n✅ Chain completed!")
    print(f"📝 Generated text ({len(result.text)} chars):")
    print("-" * 50)
    print(result.text)
    print("-" * 50)
    print(f"🔄 Iterations: {result.iteration}")
    print(f"✓ Validation: {'Passed' if result.validation_results else 'No validators'}")

if __name__ == "__main__":
    main()
```

## Using QuickStart Shortcuts

For common patterns, use QuickStart methods:

```python
from sifaka.quickstart import QuickStart

# Development (fast, uses mock model)
dev_chain = QuickStart.for_development()

# Production (with validation and improvement)
prod_chain = QuickStart.for_production(
    "openai:gpt-4",
    "Your prompt here",
    validators=["length"],
    critics=["reflexion"]
)

# Research (comprehensive setup)
research_chain = QuickStart.for_research(
    "anthropic:claude-3-sonnet",
    "Analyze the impact of AI on scientific research"
)
```

## Understanding the Output

When you run a chain, you get a `Thought` object with:

- **`text`**: The final generated text
- **`iteration`**: Number of improvement iterations
- **`validation_results`**: Results from validators
- **`critic_feedback`**: Feedback from critics
- **`history`**: Complete evolution of the thought
- **`chain_id`**: Unique identifier for this run

## Next Steps

Now that you've created your first chain:

1. **[Chain Selection Guide](../guides/chain-selection.md)** - Learn when to use PydanticAI vs Traditional chains
2. **[Learn basic concepts](basic-concepts.md)** - Understand Thoughts, Models, and the architecture
3. **[Explore examples](../../examples/)** - See more complex use cases
4. **[Add storage](../guides/storage-setup.md)** - Set up Redis or Milvus for persistence
5. **[Feedback summarization](../feedback-summarizer.md)** - Enhance critics with automatic feedback summarization
6. **[Custom models](../guides/custom-models.md)** - Create your own model integrations

## Troubleshooting

**Chain doesn't run?**
- Check your API keys are set in environment variables
- Try the mock model first: `create_model("mock:test-model")`

**Import errors?**
- Ensure you installed the right extras: `pip install sifaka[openai]`
- See [import problems](../troubleshooting/import-problems.md)

**API errors?**
- Verify your API keys are valid
- Check [configuration errors](../troubleshooting/configuration-errors.md)

## What's Next?

You've successfully created your first Sifaka chain! The framework provides much more:

- **Multiple model providers** (OpenAI, Anthropic, Google Gemini, HuggingFace, Ollama)
- **Rich validation** (length, content, format, ML classifiers)
- **Advanced critics** (Reflexion, Constitutional AI, Self-RAG)
- **Feedback summarization** (automatic summarization of critic and validation feedback)
- **Persistent storage** (Redis, Milvus, file-based)
- **Complete observability** with audit trails

Continue with the [basic concepts guide](basic-concepts.md) to understand how it all works together.
