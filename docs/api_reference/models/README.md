# Model Providers API Reference

Model Providers are components in Sifaka that connect to language model services for text generation. They handle the communication with external APIs, token counting, and response processing.

## Core Classes and Protocols

### ModelProvider

`ModelProvider` is the abstract base class for all model providers in Sifaka.

```python
from sifaka.models.base import ModelProvider, ModelConfig
from typing import Dict, Any, Optional

class MyModelProvider(ModelProvider[ModelConfig]):
    """Custom model provider implementation."""
    
    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        super().__init__(model_name, config)
    
    def generate(self, prompt: str) -> str:
        """Generate text using the model."""
        # Implementation details
        return f"Response to: {prompt}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the text."""
        # Implementation details
        return len(text.split())
```

### APIClient

`APIClient` is the interface for communicating with language model services.

```python
from sifaka.models.base import APIClient, ModelConfig

class MyAPIClient(APIClient):
    """Custom API client implementation."""
    
    def __init__(self, api_key: str):
        """Initialize the client."""
        self.api_key = api_key
    
    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to the language model service."""
        # Implementation details
        return f"Response to: {prompt}"
```

### TokenCounter

`TokenCounter` is the interface for counting tokens in text.

```python
from sifaka.models.base import TokenCounter

class MyTokenCounter(TokenCounter):
    """Custom token counter implementation."""
    
    def __init__(self, model: str):
        """Initialize the token counter."""
        self.model = model
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the text."""
        # Implementation details
        return len(text.split())
```

## Configuration

### ModelConfig

`ModelConfig` is the configuration class for model providers.

```python
from sifaka.models.base import ModelConfig

# Create a model configuration
config = ModelConfig(
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key",
    trace_enabled=True,
    params={
        "top_p": 0.9,
        "frequency_penalty": 0.5,
    }
)

# Access configuration values
print(f"Temperature: {config.temperature}")
print(f"Max tokens: {config.max_tokens}")
print(f"Top P: {config.params['top_p']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    temperature=0.5,
    params={"top_p": 0.8}
)
```

## Model Provider Types

Sifaka provides several types of model providers:

### OpenAIProvider

`OpenAIProvider` connects to OpenAI's API for text generation.

```python
from sifaka.models.openai import create_openai_chat_provider

# Create an OpenAI provider
provider = create_openai_chat_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)
```

### AnthropicProvider

`AnthropicProvider` connects to Anthropic's API for text generation.

```python
from sifaka.models.anthropic import create_anthropic_provider

# Create an Anthropic provider
provider = create_anthropic_provider(
    model_name="claude-3-opus-20240229",
    api_key="your-anthropic-api-key",
    temperature=0.7,
    max_tokens=1000
)
```

### GeminiProvider

`GeminiProvider` connects to Google's Gemini API for text generation.

```python
from sifaka.models.gemini import create_gemini_provider

# Create a Gemini provider
provider = create_gemini_provider(
    model_name="gemini-pro",
    api_key="your-gemini-api-key",
    temperature=0.7,
    max_tokens=1000
)
```

### MockProvider

`MockProvider` is a mock provider for testing.

```python
from sifaka.models.mock import create_mock_provider

# Create a mock provider
provider = create_mock_provider(
    responses={"Hello": "Hi there!"},
    default_response="I don't know how to respond to that."
)
```

## Component Architecture

Model providers in Sifaka follow a component-based architecture:

```
ModelProviderCore
├── ClientManager
├── TokenCounterManager
├── TracingManager
└── GenerationService
```

### ClientManager

`ClientManager` manages API client creation and lifecycle.

```python
from sifaka.models.managers import ClientManager
from sifaka.models.base import APIClient, ModelConfig

class MyClientManager(ClientManager):
    """Custom client manager implementation."""
    
    def create_client(self, config: ModelConfig) -> APIClient:
        """Create an API client."""
        return MyAPIClient(api_key=config.api_key)
```

### TokenCounterManager

`TokenCounterManager` manages token counting functionality.

```python
from sifaka.models.managers import TokenCounterManager
from sifaka.models.base import TokenCounter

class MyTokenCounterManager(TokenCounterManager):
    """Custom token counter manager implementation."""
    
    def create_counter(self, model: str) -> TokenCounter:
        """Create a token counter."""
        return MyTokenCounter(model=model)
```

### TracingManager

`TracingManager` manages tracing and logging.

```python
from sifaka.models.managers import TracingManager
from sifaka.utils.tracing import Tracer

class MyTracingManager(TracingManager):
    """Custom tracing manager implementation."""
    
    def __init__(self, tracer: Tracer):
        """Initialize the tracing manager."""
        self.tracer = tracer
    
    def trace(self, event: str, data: dict):
        """Trace an event."""
        self.tracer.trace(event, data)
```

### GenerationService

`GenerationService` handles text generation and error handling.

```python
from sifaka.models.managers import GenerationService
from sifaka.models.base import APIClient, ModelConfig

class MyGenerationService(GenerationService):
    """Custom generation service implementation."""
    
    def generate(self, prompt: str, client: APIClient, config: ModelConfig) -> str:
        """Generate text using the model."""
        try:
            return client.send_prompt(prompt, config)
        except Exception as e:
            # Handle error
            return f"Error: {str(e)}"
```

## Usage Examples

### Basic Model Provider Usage

```python
from sifaka.models.openai import create_openai_chat_provider

# Create a model provider
provider = create_openai_chat_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Generate text
text = provider.generate("Write a short story about a robot learning to paint.")
print(f"Generated text: {text}")

# Count tokens
token_count = provider.count_tokens("This is a test")
print(f"Token count: {token_count}")
```

### Custom Model Provider Implementation

```python
from sifaka.models.base import ModelProvider, ModelConfig, APIClient, TokenCounter
from typing import Dict, Any, Optional

class SimpleModelProvider(ModelProvider[ModelConfig]):
    """A simple model provider implementation."""
    
    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        super().__init__(model_name, config)
        self._responses = {
            "Hello": "Hi there!",
            "How are you?": "I'm doing well, thank you!",
            "What is your name?": "My name is SimpleModel.",
        }
        self._default_response = "I don't know how to respond to that."
    
    def _create_client(self) -> APIClient:
        """Create the API client."""
        # Simple implementation that doesn't actually call an API
        return SimpleAPIClient()
    
    def _create_token_counter(self) -> TokenCounter:
        """Create the token counter."""
        return SimpleTokenCounter()
    
    def generate(self, prompt: str) -> str:
        """Generate text using the model."""
        # Simple implementation that returns predefined responses
        return self._responses.get(prompt, self._default_response)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the text."""
        # Simple implementation that counts words
        return len(text.split())

class SimpleAPIClient(APIClient):
    """A simple API client implementation."""
    
    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to the language model service."""
        # Simple implementation that doesn't actually call an API
        responses = {
            "Hello": "Hi there!",
            "How are you?": "I'm doing well, thank you!",
            "What is your name?": "My name is SimpleModel.",
        }
        return responses.get(prompt, "I don't know how to respond to that.")

class SimpleTokenCounter(TokenCounter):
    """A simple token counter implementation."""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the text."""
        # Simple implementation that counts words
        return len(text.split())

# Create the model provider
provider = SimpleModelProvider(
    model_name="simple-model",
    config=ModelConfig(temperature=0.7, max_tokens=100)
)

# Use the model provider
text = provider.generate("Hello")
print(f"Generated text: {text}")
```

### Using Model Providers with Chains

Model providers are typically used in chains to generate text:

```python
from sifaka.chain import create_simple_chain
from sifaka.models.openai import create_openai_chat_provider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics.prompt import create_prompt_critic

# Create components
model = create_openai_chat_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)
rule = create_length_rule(min_chars=50, max_chars=200)
critic = create_prompt_critic(
    system_prompt="You are an expert editor. Improve the text to meet the length requirements."
)

# Create a chain
chain = create_simple_chain(
    model=model,
    rules=[rule],
    critic=critic
)

# Run the chain
result = chain.run("Write a short description of a sunset.")
print(f"Output: {result.output}")
```
