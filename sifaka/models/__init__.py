"""Models for Sifaka.

This module provides implementations of the Model protocol for various
language model providers, including OpenAI, Anthropic, and others.

It also provides factory functions for creating model instances based on
provider and model name.

Example:
    ```python
    from sifaka.models import create_model

    # Create a model using the factory function
    model = create_model("openai:gpt-4", api_key="your-api-key")

    # Generate text
    response = model.generate(
        "Write a short story about a robot.",
        temperature=0.7,
        max_tokens=500
    )
    print(response)

    # Count tokens
    token_count = model.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

from typing import List

from sifaka.models.base import create_model

# Import models with error handling
__all__: List[str] = ["create_model"]

# OpenAI
try:
    from sifaka.models.openai import OPENAI_AVAILABLE

    if OPENAI_AVAILABLE:
        __all__.extend(["OpenAIModel", "create_openai_model"])
except ImportError:
    pass

# Anthropic
try:
    from sifaka.models.anthropic import ANTHROPIC_AVAILABLE

    if ANTHROPIC_AVAILABLE:
        __all__.extend(["AnthropicModel", "create_anthropic_model"])
except ImportError:
    pass

# Gemini
try:
    from sifaka.models.gemini import GEMINI_AVAILABLE

    if GEMINI_AVAILABLE:
        __all__.extend(["GeminiModel", "create_gemini_model"])
except ImportError:
    pass
