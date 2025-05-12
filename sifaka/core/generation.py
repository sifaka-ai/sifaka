"""
Text generation module for Sifaka.

This module provides functionality for generating text using various models,
with support for caching, statistics tracking, and error handling.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import time

from pydantic import BaseModel, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.models.core.provider import ModelProviderCore as BaseModelProvider
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class Generator(BaseComponent):
    """
    Text generator using model providers.

    This class provides functionality for generating text using various models,
    with support for caching, statistics tracking, and error handling.
    """

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        model: BaseModelProvider[OutputType],
        name: str = "generator",
        description: str = "Text generator using model providers",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the generator.

        Args:
            model: The model provider to use
            name: Name of the generator
            description: Description of the generator
            config: Additional configuration
        """
        super().__init__()

        self._state_manager.update("model", model)
        self._state_manager.update("name", name)
        self._state_manager.update("description", description)
        self._state_manager.update("config", config or {})
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "generator")
        self._state_manager.set_metadata("creation_time", time.time())

    def generate(self, prompt: str, **kwargs) -> OutputType:
        """
        Generate text using the model provider.

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
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state_manager.get("result_cache", {})
        if prompt in cache:
            self._state_manager.set_metadata("cache_hit", True)
            return cache[prompt]

        # Mark as cache miss
        self._state_manager.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get model from state
            model = self._state_manager.get("model")

            # Generate text
            output = model.invoke(prompt, **kwargs)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            count = self._state_manager.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state_manager.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[prompt] = output
            self._state_manager.update("result_cache", cache)

            return output

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Generation error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generator usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state_manager.get("execution_count", 0),
            "cache_size": len(self._state_manager.get("result_cache", {})),
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "model_name": self._state_manager.get("model").name,
        }

    def clear_cache(self) -> None:
        """Clear the generator result cache."""
        self._state_manager.update("result_cache", {})
        logger.debug("Generator cache cleared")
