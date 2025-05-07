# HuggingFaceProvider API Reference

> **Note:** Comprehensive documentation for this component is coming soon.

`HuggingFaceProvider` is a model provider that connects to Hugging Face's models for text generation. It allows you to use models from the Hugging Face Hub directly within Sifaka.

## Overview

The `HuggingFaceProvider` enables integration with Hugging Face's extensive collection of open-source models. This provider will support both local model inference and API-based access to Hugging Face's hosted inference endpoints.

## Basic Usage

```python
from sifaka.models.huggingface import create_huggingface_provider

# Create a HuggingFace provider
provider = create_huggingface_provider(
    model_name="gpt2",  # or any other model from Hugging Face
    temperature=0.7,
    max_tokens=1000,
    api_key="your-huggingface-api-key"  # Optional, for accessing gated models
)

# Generate text
response = provider.generate("Explain quantum computing in simple terms.")
print(f"Response: {response}")

# Count tokens
token_count = provider.count_tokens("How many tokens is this?")
print(f"Token count: {token_count}")
```

## Features (Coming Soon)

- Support for both local model inference and API-based access
- Integration with Hugging Face's tokenizers for accurate token counting
- Support for various model architectures (GPT, T5, BERT, etc.)
- Configurable generation parameters
- Streaming support for real-time text generation
- Batched inference for improved performance

## Architecture

The `HuggingFaceProvider` will follow Sifaka's component-based architecture:

```
HuggingFaceProvider
├── HuggingFaceClient (APIClient implementation)
├── HuggingFaceTokenCounter (TokenCounter implementation)
└── Standard Sifaka provider components
```

## Configuration Options

The provider will support standard Sifaka model configuration options plus Hugging Face-specific parameters:

```python
from sifaka.models.huggingface import create_huggingface_provider

provider = create_huggingface_provider(
    model_name="gpt2",
    temperature=0.7,
    max_tokens=1000,
    api_key="your-huggingface-api-key",  # Optional
    use_local=True,  # Whether to use local model or API
    device="cuda",  # Device to run the model on (for local inference)
    params={
        # Model-specific parameters
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
)
```

## Error Handling

The provider will implement comprehensive error handling for:

- Model loading errors
- Inference errors
- Token counting errors
- API communication errors
- Resource constraints (memory, compute)

## Integration with Sifaka

The `HuggingFaceProvider` will be fully compatible with Sifaka's chain architecture:

```python
from sifaka.chain import create_simple_chain
from sifaka.models.huggingface import create_huggingface_provider
from sifaka.rules.formatting.length import create_length_rule

# Create components
model = create_huggingface_provider(
    model_name="gpt2",
    temperature=0.7,
    max_tokens=1000
)
rule = create_length_rule(min_chars=50, max_chars=200)

# Create a chain
chain = create_simple_chain(
    model=model,
    rules=[rule]
)

# Run the chain
result = chain.run("Write a short description of a sunset.")
print(f"Output: {result.output}")
```

## Coming Soon

Full documentation for the `HuggingFaceProvider` is under development and will include:

- Comprehensive API reference
- Detailed examples for different use cases
- Performance optimization guidelines
- Troubleshooting and best practices
- Integration examples with popular Hugging Face models
