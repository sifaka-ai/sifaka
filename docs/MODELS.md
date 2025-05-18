# Models Documentation

Models are integrations with various language model providers (like OpenAI, Anthropic, Google Gemini) that handle text generation in the Sifaka framework. Each model implementation follows the Model protocol, which defines a consistent interface for all models.

## Overview

The key responsibilities of models include:
- Generating text from prompts
- Counting tokens in text
- Handling API communication with the provider

Models can be created either directly through their constructors or through the create_model factory function, which uses the registry to create the appropriate model based on the provider prefix.

## Supported Model Providers

Sifaka supports the following model providers:

### OpenAI

OpenAI models include GPT-3.5, GPT-4, and other models available through the OpenAI API.

```python
from sifaka.models.openai import OpenAIModel, create_openai_model

# Create a model directly
model1 = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Create a model using the factory function
model2 = create_openai_model(model_name="gpt-4", api_key="your-api-key")

# Create a model using the generic factory function
from sifaka.models import create_model
model3 = create_model("openai:gpt-4", api_key="your-api-key")
```

### Anthropic

Anthropic models include Claude and other models available through the Anthropic API.

```python
from sifaka.models.anthropic import AnthropicModel, create_anthropic_model

# Create a model directly
model1 = AnthropicModel(model_name="claude-3-opus-20240229", api_key="your-api-key")

# Create a model using the factory function
model2 = create_anthropic_model(model_name="claude-3-opus-20240229", api_key="your-api-key")

# Create a model using the generic factory function
from sifaka.models import create_model
model3 = create_model("anthropic:claude-3-opus-20240229", api_key="your-api-key")
```

### Google Gemini

Google Gemini models include Gemini Pro and other models available through the Google AI API.

```python
from sifaka.models.gemini import GeminiModel, create_gemini_model

# Create a model directly
model1 = GeminiModel(model_name="gemini-pro", api_key="your-api-key")

# Create a model using the factory function
model2 = create_gemini_model(model_name="gemini-pro", api_key="your-api-key")

# Create a model using the generic factory function
from sifaka.models import create_model
model3 = create_model("gemini:gemini-pro", api_key="your-api-key")
```

### Mock Model

Sifaka includes a mock model for testing purposes.

```python
from sifaka.models import create_model

# Create a mock model
model = create_model("mock:test-model")
```

## Using Models

Models can be used directly to generate text and count tokens:

```python
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Generate text
response = model.generate(
    "Write a short story about a robot.",
    temperature=0.7,
    max_tokens=500,
    system_message="You are a creative writer."
)
print(response)

# Count tokens
token_count = model.count_tokens("This is a test.")
print(f"Token count: {token_count}")
```

Models can also be configured with additional options:

```python
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Configure the model with new options
model.configure(
    temperature=0.5,
    max_tokens=1000,
    presence_penalty=0.2
)

# Generate text with the new configuration
response = model.generate("Write a short story about a robot.")
print(response)
```

## Using Models with Chain

Models are typically used with the Chain class to generate text:

```python
from sifaka import Chain
from sifaka.models.openai import OpenAIModel

# Create a model
model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

# Create a chain with the model
chain = (Chain()
    .with_model(model)
    .with_prompt("Write a short story about a robot.")
)

# Run the chain
result = chain.run()
print(result.text)
```

You can also use string-based model specification with the Chain class:

```python
from sifaka import Chain

# Create a chain with string-based model specification
chain = (Chain()
    .with_model("openai:gpt-4")  # Will use OPENAI_API_KEY environment variable
    .with_prompt("Write a short story about a robot.")
)

# Run the chain
result = chain.run()
print(result.text)
```

## Model Protocol

All model implementations must follow the Model protocol, which defines the following interface:

```python
class Model(Protocol):
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            **options: Additional options to pass to the model
            
        Returns:
            The generated text
        """
        ...
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: The text to count tokens in
            
        Returns:
            The number of tokens in the text
        """
        ...
```

## Creating Custom Model Implementations

You can create custom model implementations by implementing the Model protocol:

```python
from sifaka.models.base import Model
from typing import Any

class MyCustomModel:
    def __init__(self, model_name: str, api_key: str = None, **options: Any):
        self.model_name = model_name
        self.api_key = api_key
        self.options = options
        # Initialize your custom model here
        
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt using your custom model."""
        # Merge options with default options
        merged_options = {**self.options, **options}
        
        # Implement your custom text generation logic here
        # This is just a placeholder example
        return f"Generated text from {self.model_name} for prompt: {prompt}"
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using your custom tokenizer."""
        # Implement your custom token counting logic here
        # This is just a placeholder example
        return len(text.split())
        
    def configure(self, **options: Any) -> None:
        """Configure the model with new options."""
        # Update options
        self.options.update(options)
        
        # Handle special options like api_key
        if "api_key" in options:
            self.api_key = options["api_key"]
```

You can also register your custom model with the registry system:

```python
from sifaka.registry import register_model

@register_model("my_provider")
def create_my_custom_model(model_name: str, **options: Any) -> MyCustomModel:
    """Create a custom model instance."""
    return MyCustomModel(model_name=model_name, **options)
```

## Model Options

Models support various options that control the generation process. Common options include:

- **temperature**: Controls randomness in generation (0.0 to 1.0)
- **max_tokens**: Maximum number of tokens to generate
- **top_p**: Controls diversity via nucleus sampling (0.0 to 1.0)
- **frequency_penalty**: Reduces repetition of token sequences (-2.0 to 2.0)
- **presence_penalty**: Reduces repetition of topics (-2.0 to 2.0)
- **stop**: Sequences where the model should stop generating
- **system_message**: A system message to include at the beginning of the conversation (for chat models)

Different model providers may support different options or have different names for similar options. Consult the documentation for each model provider for details.

## API Keys

Most model providers require an API key to access their services. You can provide the API key in several ways:

1. **Directly in the constructor**:
   ```python
   model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")
   ```

2. **Through environment variables**:
   ```python
   # Set environment variables
   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   
   # Create model without explicitly providing API key
   model = OpenAIModel(model_name="gpt-4")
   ```

3. **Using a .env file**:
   ```
   # .env file
   OPENAI_API_KEY=your-api-key
   ```
   
   ```python
   # Load environment variables from .env file
   from dotenv import load_dotenv
   load_dotenv()
   
   # Create model without explicitly providing API key
   model = OpenAIModel(model_name="gpt-4")
   ```

## Best Practices

1. **Use environment variables or .env files** for API keys instead of hardcoding them
2. **Handle API errors gracefully** by catching exceptions and providing meaningful error messages
3. **Use appropriate temperature settings** for your use case (lower for deterministic outputs, higher for creative outputs)
4. **Set reasonable max_tokens limits** to control the length of generated text
5. **Use system messages** to guide the model's behavior (for chat models)
6. **Consider token usage** when designing prompts and processing responses
