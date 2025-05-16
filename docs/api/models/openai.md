# OpenAI Model

This page documents the OpenAI model implementation in Sifaka.

## Overview

The `OpenAIModel` class provides an implementation of the Model protocol for OpenAI models. It supports both the ChatCompletion and Completion APIs, and includes features like token counting and error handling.

## Installation

To use the OpenAI model, you need to install the OpenAI Python package:

```bash
pip install openai tiktoken
```

## Basic Usage

```python
from sifaka.models import create_model

# Create an OpenAI model using the factory function
model = create_model("openai", "gpt-4", api_key="your-api-key")

# Generate text
response = model.generate("Write a short story about a robot.")
print(response)

# Count tokens
token_count = model.count_tokens(response)
print(f"Token count: {token_count}")
```

## Direct Initialization

You can also initialize the OpenAI model directly:

```python
from sifaka.models.openai import OpenAIModel

# Create an OpenAI model
model = OpenAIModel(
    model_name="gpt-4",
    api_key="your-api-key",
    organization="your-organization-id"  # Optional
)

# Generate text
response = model.generate("Write a short story about a robot.")
print(response)
```

## API Reference

### Constructor

```python
OpenAIModel(
    model_name: str,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    **options: Any
)
```

Initialize the OpenAI model.

**Parameters:**
- `model_name`: The name of the OpenAI model to use.
- `api_key`: The OpenAI API key to use. If not provided, it will be read from the OPENAI_API_KEY environment variable.
- `organization`: The OpenAI organization ID to use.
- `**options`: Additional options to pass to the OpenAI client.

**Raises:**
- `ConfigurationError`: If the OpenAI package is not installed.
- `ModelError`: If the API key is not provided and not available in the environment.

### Methods

#### `generate`

```python
generate(self, prompt: str, **options: Any) -> str
```

Generate text from a prompt.

**Parameters:**
- `prompt`: The prompt to generate text from.
- `**options`: Additional options to pass to the OpenAI API.
  - `temperature`: Controls randomness. Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic.
  - `max_tokens`: Maximum number of tokens to generate.
  - `top_p`: Controls diversity via nucleus sampling.
  - `frequency_penalty`: Reduces repetition of token sequences.
  - `presence_penalty`: Reduces repetition of topics.
  - `stop`: Sequences where the API will stop generating further tokens.
  - `system_message`: A system message to include at the beginning of the conversation.
  - `use_completion_api`: Force using the completion API instead of chat API.

**Returns:**
- The generated text.

**Raises:**
- `ModelAPIError`: If there is an error communicating with the OpenAI API.

#### `count_tokens`

```python
count_tokens(self, text: str) -> int
```

Count tokens in text.

**Parameters:**
- `text`: The text to count tokens in.

**Returns:**
- The number of tokens in the text.

**Raises:**
- `ModelError`: If there is an error counting tokens.

## Advanced Usage

### Using System Messages

You can include a system message when generating text with chat models:

```python
from sifaka.models import create_model

model = create_model("openai", "gpt-4", api_key="your-api-key")

response = model.generate(
    "Write a short story about a robot.",
    system_message="You are a creative writing assistant that specializes in science fiction."
)

print(response)
```

### Forcing the Completion API

For models that support both the chat and completion APIs, you can force the use of the completion API:

```python
from sifaka.models import create_model

model = create_model("openai", "gpt-3.5-turbo", api_key="your-api-key")

response = model.generate(
    "Write a short story about a robot.",
    use_completion_api=True
)

print(response)
```

### Error Handling

The OpenAI model implementation handles various API errors and provides meaningful error messages:

```python
from sifaka.models import create_model
from sifaka.errors import ModelAPIError

try:
    model = create_model("openai", "gpt-4", api_key="your-api-key")
    response = model.generate("Write a short story about a robot.")
    print(response)
except ModelAPIError as e:
    print(f"API error: {e}")
```

## Environment Variables

The OpenAI model implementation supports the following environment variables:

- `OPENAI_API_KEY`: The API key to use for OpenAI API calls.
- `OPENAI_ORGANIZATION`: The organization ID to use for OpenAI API calls.

If these environment variables are set, you don't need to provide the corresponding parameters when creating the model:

```python
import os
from sifaka.models import create_model

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_ORGANIZATION"] = "your-organization-id"

# Create model without explicitly providing API key or organization
model = create_model("openai", "gpt-4")
```
