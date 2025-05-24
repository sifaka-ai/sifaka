"""Ollama model implementation for Sifaka.

This module provides an implementation of the Model protocol for Ollama models,
supporting local LLM inference through the Ollama REST API.

Ollama provides a simple REST API for running large language models locally.
This implementation supports text generation, token counting with model-specific
strategies, and health checking.

Example:
    ```python
    from sifaka.models.ollama import OllamaModel, create_ollama_model

    # Create a model directly
    model1 = OllamaModel(model_name="llama2", base_url="http://localhost:11434")

    # Create a model using the factory function
    model2 = create_ollama_model(model_name="mistral", base_url="http://localhost:11434")

    # Generate text
    response = model1.generate(
        "Write a short story about a robot.",
        temperature=0.7,
        max_tokens=500
    )
    print(response)

    # Count tokens
    token_count = model1.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

import json
import logging
from typing import Any, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ConfigurationError, ModelError, model_context
from sifaka.utils.mixins import ContextAwareMixin

logger = logging.getLogger(__name__)

# We use the REST API directly, so no need for the ollama package
# This implementation is self-contained using only requests


class OllamaTokenCounter:
    """Token counting strategies for Ollama models.

    This class provides model-specific token counting strategies for different
    Ollama models, with fallback to approximation for unknown models.
    """

    def __init__(self, model_name: str):
        """Initialize the token counter with model-specific strategy.

        Args:
            model_name: The name of the Ollama model.
        """
        self.model_name = model_name.lower()
        self.strategy = self._select_strategy()

    def _select_strategy(self) -> str:
        """Select the appropriate token counting strategy based on model name.

        Returns:
            The strategy name to use for token counting.
        """
        if "llama" in self.model_name:
            return "llama_tokenizer"
        elif "mistral" in self.model_name:
            return "mistral_tokenizer"
        elif "codellama" in self.model_name:
            return "llama_tokenizer"  # CodeLlama uses Llama tokenizer
        elif "phi" in self.model_name:
            return "approximate"
        else:
            return "approximate"  # Default fallback

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the selected strategy.

        Args:
            text: The text to count tokens in.

        Returns:
            The estimated number of tokens.
        """
        if not text:
            return 0

        if self.strategy == "llama_tokenizer":
            # Llama models: ~1 token per 4 characters, but more accurate for common patterns
            # Account for common tokens and subword patterns
            return max(1, int(len(text) * 0.3))
        elif self.strategy == "mistral_tokenizer":
            # Mistral models: similar to Llama but slightly different tokenization
            return max(1, int(len(text) * 0.28))
        else:
            # Approximate: 1 token ≈ 4 characters (conservative estimate)
            return max(1, len(text) // 4)


class OllamaConnection:
    """Connection manager for Ollama REST API.

    This class handles HTTP communication with the Ollama server,
    including health checking, model listing, and generation requests.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        """Initialize the Ollama connection.

        Args:
            base_url: The base URL for the Ollama server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def health_check(self) -> bool:
        """Check if the Ollama server is healthy and accessible.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models on the Ollama server.

        Returns:
            A list of available model names.

        Raises:
            ModelError: If the request fails.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise ModelError(f"Failed to list Ollama models: {e}")

    def generate(self, prompt: str, model_name: str, **options: Any) -> str:
        """Generate text using the Ollama API.

        Args:
            prompt: The prompt to generate text from.
            model_name: The name of the model to use.
            **options: Additional generation options.

        Returns:
            The generated text.

        Raises:
            ModelError: If the generation request fails.
        """
        # Build the request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # We want the complete response
        }

        # Add supported options
        if "temperature" in options:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = options["temperature"]
        if "max_tokens" in options:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = options["max_tokens"]
        if "stop" in options:
            payload["options"] = payload.get("options", {})
            payload["options"]["stop"] = options["stop"]

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except requests.exceptions.RequestException as e:
            raise ModelError(f"Ollama API request failed: {e}")
        except json.JSONDecodeError as e:
            raise ModelError(f"Failed to parse Ollama response: {e}")
        except Exception as e:
            raise ModelError(f"Unexpected error during Ollama generation: {e}")


class OllamaModel(ContextAwareMixin):
    """Ollama model implementation for local LLM inference.

    This class implements the Model protocol for Ollama models, providing
    local LLM inference through the Ollama REST API. It supports text generation,
    model-specific token counting, and health checking.

    The class automatically handles connection management, retries, and provides
    intelligent token counting based on the model type.

    Example:
        ```python
        from sifaka.models.ollama import OllamaModel

        # Create a model
        model = OllamaModel(
            model_name="llama2",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        # Generate text
        response = model.generate(
            "Write a short story about a robot.",
            max_tokens=500
        )
        print(response)
        ```
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        retriever: Optional[Retriever] = None,
        timeout: int = 60,
        **options: Any,
    ):
        """Initialize the Ollama model with the specified parameters.

        Args:
            model_name: The name of the Ollama model to use.
            base_url: The base URL for the Ollama server.
            retriever: Optional retriever for direct access.
            timeout: Request timeout in seconds.
            **options: Additional options for the model.

        Raises:
            ConfigurationError: If Ollama is not accessible.
            ModelError: If the model is not available.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.retriever = retriever
        self.timeout = timeout
        self.options = options

        # Initialize connection and token counter
        self.connection = OllamaConnection(base_url, timeout)
        self.token_counter = OllamaTokenCounter(model_name)

        # Verify connection and model availability
        self._verify_setup()

        logger.debug(f"Created Ollama model: {model_name} at {base_url}")

    def _verify_setup(self) -> None:
        """Verify that Ollama is accessible and the model is available.

        Raises:
            ConfigurationError: If Ollama is not accessible.
            ModelError: If the model is not available.
        """
        # Check if Ollama server is accessible
        if not self.connection.health_check():
            raise ConfigurationError(
                f"Cannot connect to Ollama server at {self.base_url}",
                suggestions=[
                    "Ensure Ollama is running (try 'ollama serve')",
                    "Check that the base_url is correct",
                    "Verify network connectivity to the Ollama server",
                ],
            )

        # Check if the model is available
        try:
            available_models = self.connection.list_models()
            if self.model_name not in available_models:
                raise ModelError(
                    f"Model '{self.model_name}' not found on Ollama server",
                    suggestions=[
                        f"Pull the model with 'ollama pull {self.model_name}'",
                        f"Available models: {', '.join(available_models)}",
                        "Check the model name spelling",
                    ],
                )
        except ModelError:
            raise  # Re-raise model errors
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
            # Continue anyway - the model might still work

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.
        """
        # Merge instance options with call-time options
        merged_options = {**self.options, **options}

        try:
            with model_context(
                model_name=self.model_name,
                operation="generation",
                message_prefix="Failed to generate text with Ollama model",
                suggestions=[
                    "Check that Ollama is running and accessible",
                    "Verify that the model is available",
                    "Check network connectivity to Ollama server",
                ],
                metadata={
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "prompt_length": len(prompt),
                    "temperature": merged_options.get("temperature"),
                    "max_tokens": merged_options.get("max_tokens"),
                },
            ):
                return self.connection.generate(prompt, self.model_name, **merged_options)

        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Ollama generation failed: {e}")
            raise

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container.

        The model uses whatever context is already in the Thought container,
        as the Chain orchestrates all retrieval operations.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        logger.debug(f"Generating text with Ollama model using Thought: {self.model_name}")

        # Use mixin to build contextualized prompt
        full_prompt = self._build_contextualized_prompt(thought, max_docs=5)

        # Add system prompt if available
        if thought.system_prompt:
            full_prompt = f"{thought.system_prompt}\n\n{full_prompt}"

        # Log context usage
        if self._has_context(thought):
            context_summary = self._get_context_summary(thought)
            logger.debug(f"OllamaModel using context: {context_summary}")

        # Generate text using the contextualized prompt
        generated_text = self.generate(full_prompt, **options)
        return generated_text, full_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model-specific strategies.

        Args:
            text: The text to count tokens in.

        Returns:
            The estimated number of tokens.
        """
        return self.token_counter.count_tokens(text)


def create_ollama_model(
    model_name: str,
    base_url: str = "http://localhost:11434",
    retriever: Optional[Retriever] = None,
    **kwargs: Any,
) -> OllamaModel:
    """Create an Ollama model instance.

    This factory function creates an Ollama model instance with the specified
    configuration. It provides a convenient way to create Ollama models with
    default settings.

    Args:
        model_name: The name of the Ollama model to use.
        base_url: The base URL for the Ollama server.
        retriever: Optional retriever for direct access.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        An OllamaModel instance.

    Raises:
        ConfigurationError: If Ollama is not accessible.
        ModelError: If the model is not available.

    Example:
        ```python
        from sifaka.models.ollama import create_ollama_model

        # Create a model using the factory function
        model = create_ollama_model(
            model_name="llama2",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        # Generate text
        response = model.generate("Write a short story about a robot.")
        print(response)
        ```
    """
    import logging

    from sifaka.utils.error_handling import log_error

    logger = logging.getLogger(__name__)

    # Log model creation attempt
    logger.debug(f"Creating Ollama model with name '{model_name}'")

    try:
        # Create the model instance
        model = OllamaModel(model_name=model_name, base_url=base_url, retriever=retriever, **kwargs)

        logger.debug(f"Successfully created Ollama model: {model_name}")
        return model

    except Exception as e:
        # Log the error with context
        log_error(
            logger,
            f"Failed to create Ollama model '{model_name}': {str(e)}",
            error=e,
            context={
                "model_name": model_name,
                "base_url": base_url,
                "kwargs": kwargs,
            },
        )
        raise
