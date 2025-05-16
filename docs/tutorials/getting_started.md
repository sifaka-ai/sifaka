# Getting Started with Sifaka

This tutorial will guide you through the basics of using Sifaka to build reliable LLM applications.

## Installation

First, install Sifaka using pip:

```bash
pip install sifaka
```

For full functionality, you'll also want to install the dependencies for the model providers you plan to use:

```bash
# For OpenAI models
pip install openai tiktoken

# For Anthropic models
pip install anthropic

# For Google Gemini models
pip install google-generativeai
```

## Basic Usage

Let's start with a simple example of using Sifaka to generate text:

```python
import sifaka

# Create a simple chain
result = (sifaka.Chain()
    .with_model("openai:gpt-4")  # Replace with your preferred model
    .with_prompt("Write a short story about a robot.")
    .run())

print(result.text)
```

This example:
1. Creates a new Chain
2. Configures it to use the OpenAI GPT-4 model
3. Sets the prompt to "Write a short story about a robot."
4. Runs the chain
5. Prints the generated text

## Adding Validation

Now, let's add validation to ensure the generated text meets certain criteria:

```python
import sifaka
from sifaka.validators import length, content

# Create a chain with validation
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(content(required_terms=["robot", "AI"]))
    .run())

if result.passed:
    print("Validation passed!")
    print(result.text)
else:
    print("Validation failed:")
    for i, validation_result in enumerate(result.validation_results):
        if not validation_result.passed:
            print(f"  {i+1}. {validation_result.message}")
```

This example:
1. Creates a chain with the same model and prompt
2. Adds a length validator that requires between 50 and 200 words
3. Adds a content validator that requires the terms "robot" and "AI"
4. Runs the chain
5. Checks if validation passed
6. If validation passed, prints the text
7. If validation failed, prints the validation error messages

## Adding Improvement

Now, let's add improvement to enhance the quality of the generated text:

```python
import sifaka
from sifaka.validators import length, clarity

# Create a chain with validation and improvement
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .improve_with(clarity())
    .run())

print("Final text:")
print(result.text)

print("\nImprovement details:")
for i, improvement_result in enumerate(result.improvement_results):
    print(f"Improvement {i+1}:")
    print(f"  Changes made: {improvement_result.changes_made}")
    print(f"  Message: {improvement_result.message}")
```

This example:
1. Creates a chain with the same model and prompt
2. Adds a length validator
3. Adds a clarity improver that enhances the clarity and coherence of the text
4. Runs the chain
5. Prints the final text
6. Prints details about the improvements made

## Configuring Models

You can configure the model with additional options:

```python
import sifaka

# Create a chain with model options
result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .with_options(
        temperature=0.7,  # Controls randomness
        max_tokens=500,   # Maximum tokens to generate
        system_message="You are a creative writing assistant that specializes in science fiction."
    )
    .run())

print(result.text)
```

This example:
1. Creates a chain with the same model and prompt
2. Adds options for the model:
   - `temperature`: Controls randomness (0.0 to 1.0)
   - `max_tokens`: Maximum tokens to generate
   - `system_message`: A system message for chat models
3. Runs the chain
4. Prints the generated text

## Using Different Model Providers

Sifaka supports multiple model providers. Here's how to use different providers:

```python
import sifaka
import os

# Set API keys (or use environment variables)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# Using OpenAI
openai_result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())

# Using Anthropic
anthropic_result = (sifaka.Chain()
    .with_model("anthropic:claude-3-opus")
    .with_prompt("Write a short story about a robot.")
    .run())

# Using Google Gemini
gemini_result = (sifaka.Chain()
    .with_model("gemini:gemini-pro")
    .with_prompt("Write a short story about a robot.")
    .run())

print("OpenAI result:")
print(openai_result.text[:100] + "...\n")

print("Anthropic result:")
print(anthropic_result.text[:100] + "...\n")

print("Google Gemini result:")
print(gemini_result.text[:100] + "...")
```

This example:
1. Sets API keys for different providers
2. Creates chains with different model providers
3. Runs each chain
4. Prints the first 100 characters of each result

## Error Handling

It's important to handle errors when working with LLMs:

```python
import sifaka
from sifaka.errors import ModelError, ValidationError, ChainError

try:
    result = (sifaka.Chain()
        .with_model("openai:gpt-4")
        .with_prompt("Write a short story about a robot.")
        .run())
    
    print(result.text)
except ModelError as e:
    print(f"Model error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except ChainError as e:
    print(f"Chain error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

This example:
1. Creates a chain
2. Runs the chain inside a try-except block
3. Catches different types of errors that might occur

## Next Steps

Now that you've learned the basics of Sifaka, you can explore more advanced features:

- [Using Critics](using_critics.md): Learn how to use LLM-based critics to validate and improve text
- [Custom Validators](custom_validators.md): Create your own validators for specific requirements
- [Advanced Chain Configuration](advanced_chain.md): Explore advanced options for configuring chains

For more information, see the [API Reference](../api/index.md) and [Architecture Overview](../architecture/index.md).
