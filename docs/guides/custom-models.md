# Custom Models Guide

Learn how to create custom model integrations for Sifaka, enabling support for new AI providers or specialized model configurations.

## Overview

Sifaka's model system is designed for extensibility. You can create custom models by implementing the `Model` protocol or extending the `BaseModelImplementation` class for common functionality.

## Model Protocol

All models must implement the `Model` protocol:

```python
from typing import Any, Protocol
from sifaka.core.thought import Thought

class Model(Protocol):
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        ...

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container with context."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...
```

## Quick Start: Simple Custom Model

Here's a minimal custom model implementation:

```python
from typing import Any
from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought

class MyCustomModel:
    """Simple custom model implementation."""

    def __init__(self, model_name: str, **options: Any):
        self.model_name = model_name
        self.options = options

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        # Your custom generation logic here
        return f"Generated response for: {prompt[:50]}..."

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container."""
        # Access context from the thought
        context = ""
        if thought.pre_generation_context:
            context = "\n".join([doc.content for doc in thought.pre_generation_context])

        # Build prompt with context
        full_prompt = f"{context}\n\n{thought.prompt}" if context else thought.prompt

        # Generate text
        generated_text = self.generate(full_prompt, **options)

        return generated_text, full_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens (simple word-based approximation)."""
        return len(text.split())

# Use your custom model
model = MyCustomModel("my-model")
```

## Using BaseModelImplementation

For production models, extend `BaseModelImplementation` for built-in error handling, API key management, and logging:

```python
from typing import Any, Optional
from sifaka.models.shared import BaseModelImplementation
from sifaka.core.thought import Thought

class MyProductionModel(BaseModelImplementation):
    """Production-ready custom model with full error handling."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.example.com",
        **options: Any
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            provider_name="MyProvider",
            env_var_name="MY_PROVIDER_API_KEY",
            required_packages=["requests"],  # List required packages
            **options
        )
        self.base_url = base_url
        self.client = self._create_client()

    def _create_client(self):
        """Create API client with proper configuration."""
        import requests
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        return session

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Implement the actual generation logic."""
        # This method is called by the base class after validation
        response = self.client.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "model": self.model_name,
                **options
            }
        )
        response.raise_for_status()
        return response.json()["text"]

    def count_tokens(self, text: str) -> int:
        """Implement token counting for your provider."""
        # Use your provider's tokenizer or approximation
        return len(text.split())

# Factory function for easy creation
def create_my_model(model_name: str, **kwargs) -> MyProductionModel:
    """Create a MyProvider model instance."""
    return MyProductionModel(model_name=model_name, **kwargs)
```

## Integration with Factory System

To integrate with Sifaka's `create_model()` factory, you have two options:

### Option 1: Register with Factory (Recommended)

Extend the factory function in your code:

```python
from sifaka.models.base import create_model as _original_create_model

def create_model(model_spec: str, **kwargs):
    """Extended factory function with custom provider support."""
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
        if provider == "myprovider":
            return create_my_model(model_name=model_name, **kwargs)

    # Fall back to original factory
    return _original_create_model(model_spec, **kwargs)

# Now you can use: create_model("myprovider:my-model")
```

### Option 2: Direct Instantiation

Use your model directly without the factory:

```python
from sifaka.agents import create_pydantic_chain
from pydantic_ai import Agent

# Create your custom model
model = MyProductionModel("my-model", api_key="your-key")

# Create PydanticAI agent with custom model
agent = Agent(model, system_prompt="You are a helpful assistant.")

# Use in modern PydanticAI chain
chain = create_pydantic_chain(
    agent=agent,
    validators=[],
    critics=[]
)
```

## Advanced Features

### Context-Aware Generation

Access retrieved context from the Thought container:

```python
def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
    """Generate with full context awareness."""

    # Build context from retrieved documents
    context_parts = []

    # Pre-generation context (e.g., relevant documents)
    if thought.pre_generation_context:
        context_parts.append("Relevant context:")
        for doc in thought.pre_generation_context:
            context_parts.append(f"- {doc.content}")

    # Post-generation context (e.g., feedback from previous iterations)
    if thought.post_generation_context:
        context_parts.append("Additional context:")
        for doc in thought.post_generation_context:
            context_parts.append(f"- {doc.content}")

    # Combine context with prompt
    context_str = "\n".join(context_parts)
    full_prompt = f"{context_str}\n\nTask: {thought.prompt}" if context_str else thought.prompt

    # Generate with context
    generated_text = self._generate_impl(full_prompt, **options)

    return generated_text, full_prompt
```

### Async Support

Add async methods for better performance:

```python
import asyncio
from typing import Any

class AsyncCustomModel(BaseModelImplementation):
    """Custom model with async support."""

    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Async generation implementation."""
        # Use async HTTP client
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, "model": self.model_name, **options},
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as response:
                result = await response.json()
                return result["text"]

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._generate_async(prompt, **options))
```

### Error Handling

The base class provides comprehensive error handling:

```python
class RobustCustomModel(BaseModelImplementation):
    """Model with custom error handling."""

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Generation with custom error handling."""
        try:
            response = self.client.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, **options},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["text"]

        except requests.exceptions.Timeout:
            # Handle specific errors with helpful messages
            self._handle_api_error(
                error=TimeoutError("Request timed out"),
                operation="generation",
                suggestions=[
                    "Try reducing the max_tokens parameter",
                    "Check your network connection",
                    "Consider using a different model"
                ]
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limiting
                self._handle_api_error(
                    error=e,
                    operation="generation",
                    suggestions=[
                        "Wait before retrying",
                        "Reduce request frequency",
                        "Check your API quota"
                    ]
                )
            else:
                # Other HTTP errors
                self._handle_api_error(error=e, operation="generation")
```

## Testing Your Custom Model

Create comprehensive tests for your model:

```python
import pytest
from sifaka.core.thought import Thought

def test_custom_model_generation():
    """Test basic text generation."""
    model = MyCustomModel("test-model")

    result = model.generate("Write a story")
    assert isinstance(result, str)
    assert len(result) > 0

def test_custom_model_with_thought():
    """Test generation with Thought container."""
    model = MyCustomModel("test-model")
    thought = Thought(prompt="Write a story")

    text, prompt_used = model.generate_with_thought(thought)
    assert isinstance(text, str)
    assert isinstance(prompt_used, str)
    assert len(text) > 0

def test_token_counting():
    """Test token counting functionality."""
    model = MyCustomModel("test-model")

    count = model.count_tokens("Hello world")
    assert isinstance(count, int)
    assert count > 0

def test_integration_with_chain():
    """Test model works in a PydanticAI Chain."""
    from sifaka.agents import create_pydantic_chain
    from pydantic_ai import Agent

    model = MyCustomModel("test-model")
    agent = Agent(model, system_prompt="You are a test assistant.")
    chain = create_pydantic_chain(
        agent=agent,
        validators=[],
        critics=[]
    )

    result = chain.run("Test prompt")
    assert result.text is not None
    assert len(result.text) > 0
```

## Best Practices

### 1. Configuration Management
```python
# Use environment variables for API keys
import os

class SecureModel(BaseModelImplementation):
    def __init__(self, model_name: str, **options):
        api_key = options.get('api_key') or os.getenv('MY_PROVIDER_API_KEY')
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            provider_name="MyProvider",
            env_var_name="MY_PROVIDER_API_KEY",
            **options
        )
```

### 2. Proper Token Counting
```python
def count_tokens(self, text: str) -> int:
    """Accurate token counting using provider's tokenizer."""
    try:
        # Use provider-specific tokenizer if available
        return self.tokenizer.count_tokens(text)
    except Exception:
        # Fallback to word-based approximation
        return len(text.split())
```

### 3. Validation and Normalization
```python
def _validate_generate_params(self, prompt: str, **options: Any) -> dict:
    """Validate and normalize generation parameters."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Normalize parameters
    normalized = {}
    if 'temperature' in options:
        temp = float(options['temperature'])
        normalized['temperature'] = max(0.0, min(2.0, temp))

    if 'max_tokens' in options:
        normalized['max_tokens'] = max(1, int(options['max_tokens']))

    return normalized
```

## Next Steps

- **[Custom validators guide](custom-validators.md)** - Create domain-specific validation
- **[Storage setup guide](storage-setup.md)** - Configure persistent storage
- **[Performance tuning guide](performance-tuning.md)** - Optimize your models
- **[API reference](../api/API_REFERENCE.md)** - Complete technical documentation

Your custom model is now ready to use with all of Sifaka's features including validation, criticism, and storage!
