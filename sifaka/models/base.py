"""Base model interface for Sifaka.

This module defines the Model protocol that all model implementations must follow
and provides a factory function for creating model instances based on provider
and model name.

The Model protocol requires the following methods:
- generate: Generate text from a prompt
- generate_with_thought: Generate text using a Thought container
- count_tokens: Count tokens in text

Example:
    ```python
    from sifaka.models.base import create_model

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

import importlib
import logging
from typing import Any, Dict, Optional, Type, Union

from sifaka.core.interfaces import Model, Retriever
from sifaka.core.thought import Document, Thought
from sifaka.utils.error_handling import ConfigurationError, ModelError
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


def create_model(
    model_spec: str,
    retriever: Optional[Retriever] = None,
    **kwargs: Any,
) -> Model:
    """Create a model instance based on provider and model name.

    This factory function creates a model instance based on the provider and model name.
    The model_spec can be in the format "provider:model_name" or just "model_name".
    If only "model_name" is provided, the provider is inferred from the model name.

    Args:
        model_spec: The model specification in the format "provider:model_name" or "model_name".
        retriever: Optional retriever to provide to the model for direct access.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A model instance.

    Raises:
        ConfigurationError: If the provider is not supported or the model cannot be created.
    """
    # Parse the model specification
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        # Infer provider from model name
        model_name = model_spec
        if model_name.startswith("gpt-"):
            provider = "openai"
        elif model_name.startswith("claude-"):
            provider = "anthropic"
        else:
            provider = "mock"  # Default to mock for testing

    # Create the model based on the provider
    try:
        if provider == "openai":
            # Import the OpenAI model implementation
            from sifaka.models.openai import create_openai_model

            return create_openai_model(model_name=model_name, retriever=retriever, **kwargs)
        elif provider == "anthropic":
            # Import the Anthropic model implementation
            from sifaka.models.anthropic import create_anthropic_model

            return create_anthropic_model(model_name=model_name, retriever=retriever, **kwargs)
        elif provider == "mock":
            # Create a mock model for testing
            return MockModel(model_name=model_name, retriever=retriever, **kwargs)
        else:
            raise ConfigurationError(
                f"Unsupported model provider: {provider}",
                suggestions=[
                    "Use 'openai' for OpenAI models",
                    "Use 'anthropic' for Anthropic models",
                    "Use 'mock' for mock models",
                ],
            )
    except ImportError as e:
        # Handle import errors gracefully
        raise ConfigurationError(
            f"Failed to import model implementation for provider '{provider}': {str(e)}",
            suggestions=[
                f"Install the required package for {provider} models",
                "Check that the package is installed correctly",
            ],
        ) from e
    except Exception as e:
        # Handle other errors
        raise ConfigurationError(
            f"Failed to create model for provider '{provider}': {str(e)}",
            suggestions=[
                "Check that the model name is correct",
                "Check that the API key is correct",
                "Check that the provider is supported",
            ],
        ) from e


class MockModel:
    """Mock model implementation for testing.

    This class provides a simple mock implementation of the Model protocol
    for testing purposes. It returns predefined responses for generate and
    count_tokens methods.
    """

    def __init__(self, model_name: str, retriever: Optional[Retriever] = None, **kwargs: Any):
        """Initialize the mock model.

        Args:
            model_name: The name of the model.
            retriever: Optional retriever for direct access.
            **kwargs: Additional keyword arguments.
        """
        self.model_name = model_name
        self.retriever = retriever
        self.kwargs = kwargs
        logger.debug(f"Created mock model with name: {model_name}")

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            A mock response.
        """
        logger.debug(f"Generating text with mock model: {self.model_name}")
        return f"Mock response from {self.model_name} for: {prompt[:50]}..."

    def generate_with_thought(self, thought: Thought, **options: Any) -> str:
        """Generate text using a Thought container.

        The model no longer handles retrieval - the Chain orchestrates all retrieval.
        The model just uses whatever context is already in the Thought container.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A mock response.
        """
        logger.debug(f"Generating text with mock model using Thought: {self.model_name}")

        # Extract information from the thought (Chain has already handled retrieval)
        prompt = thought.prompt
        system_prompt = thought.system_prompt or ""

        # Process pre-generation context if available (provided by Chain)
        context = ""
        if thought.pre_generation_context:
            context = "Context:\n" + "\n".join(doc.text for doc in thought.pre_generation_context)
            logger.debug(
                f"Using {len(thought.pre_generation_context)} context documents provided by Chain"
            )

        # Combine all information
        full_prompt = f"{system_prompt}\n\n{context}\n\n{prompt}".strip()

        return f"Mock response from {self.model_name} for: {full_prompt[:50]}..."

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            A mock token count.
        """
        # Simple approximation: count words
        return len(text.split())
