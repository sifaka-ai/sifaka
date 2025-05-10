"""
OpenAI model provider for Sifaka.

This module provides integration with OpenAI's language models,
enabling text generation and completion capabilities.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import time

from pydantic import BaseModel, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.models.base import BaseModelProvider
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class OpenAIProvider(BaseModelProvider[OutputType], BaseComponent):
    """
    Provider for OpenAI language models.

    This class provides integration with OpenAI's language models,
    enabling text generation and completion capabilities.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        model_name: str,
        api_key: str,
        name: str = "openai_provider",
        description: str = "Provider for OpenAI language models",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            model_name: Name of the OpenAI model to use
            api_key: OpenAI API key
            name: Name of the provider
            description: Description of the provider
            config: Additional configuration
        """
        super().__init__()

        self._state.update("model_name", model_name)
        self._state.update("api_key", api_key)
        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "model_provider")
        self._state.set_metadata("creation_time", time.time())

    def invoke(self, prompt: str, **kwargs) -> OutputType:
        """
        Generate text using the OpenAI model.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters

        Returns:
            Generated text output

        Raises:
            ModelError: If generation fails
            APIError: If API call fails
            AuthenticationError: If API key is invalid
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state.get("result_cache", {})
        if prompt in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[prompt]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get model configuration
            model_name = self._state.get("model_name")
            api_key = self._state.get("api_key")

            # Import OpenAI here to avoid dependency issues
            import openai

            openai.api_key = api_key

            # Prepare generation parameters
            params = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "stop": kwargs.get("stop", None),
            }

            # Generate text
            response = openai.Completion.create(**params)
            output = response.choices[0].text.strip()

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[prompt] = output
            self._state.update("result_cache", cache)

            return output

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Generation error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state.get("execution_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "model_name": self._state.get("model_name", "unknown"),
        }

    def clear_cache(self) -> None:
        """Clear the provider result cache."""
        self._state.update("result_cache", {})
        logger.debug("Provider cache cleared")
