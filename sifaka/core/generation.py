"""
Text generation module for Sifaka.

This module provides functionality for generating text using various models,
with support for caching, statistics tracking, and error handling.
"""

from typing import Any, Dict, Optional, TypeVar, Union, cast
import time
from pydantic import PrivateAttr
from sifaka.core.base import BaseComponent
from sifaka.models.core.provider import ModelProviderCore as BaseModelProvider, ModelConfig
from sifaka.utils.state import StateManager, create_manager_state
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
OutputType = TypeVar("OutputType", bound=ModelConfig)


class Generator(BaseComponent):
    """
    Text generator using model providers.

    This class provides functionality for generating text using various models,
    with support for caching, statistics tracking, and error handling.
    """

    _state_manager = PrivateAttr(default_factory=create_manager_state)

    def __init__(
        self,
        model: BaseModelProvider[OutputType],
        name: str = "generator",
        description: str = "Text generator using model providers",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model: The model provider to use
            name: Name of the generator
            description: Description of the generator
            config: Additional configuration
        """
        super().__init__(name=name, description=description)
        self._state_manager.update("model", model)
        self._state_manager.update("config", config or {})
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})
        self._state_manager.set_metadata("component_type", "generator")
        self._state_manager.set_metadata("creation_time", time.time())

    def generate(self, prompt: str, **kwargs: Any) -> Union[OutputType, None]:
        """
        Generate text using the model provider.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters

        Returns:
            Generated text output or None if no model is available

        Raises:
            ModelError: If generation fails
            APIError: If API call fails
            AuthenticationError: If API key is invalid
        """
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)
        cache = self._state_manager.get("result_cache", {})
        if prompt in cache:
            self._state_manager.set_metadata("cache_hit", True)
            cached_result = cache[prompt]
            # Return the cached result with explicit type casting
            # This tells mypy that we're returning the expected type
            return cached_result
        self._state_manager.set_metadata("cache_hit", False)
        start_time = time.time()
        try:
            model = self._state_manager.get("model")
            if model:
                output = model.invoke(prompt, **kwargs)
                end_time = time.time()
                exec_time = end_time - start_time
                avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
                count = self._state_manager.get("execution_count", 1)
                new_avg = (avg_time * (count - 1) + exec_time) / count
                self._state_manager.set_metadata("avg_execution_time", new_avg)
                max_time = self._state_manager.get_metadata("max_execution_time", 0)
                if exec_time > max_time:
                    self._state_manager.set_metadata("max_execution_time", exec_time)

                # Store the output in the cache
                cache[prompt] = output
                self._state_manager.update("result_cache", cache)
                return output
            return None
        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            if logger:
                logger.error(f"Generation error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generator usage.

        Returns:
            Dictionary with usage statistics
        """
        model = self._state_manager.get("model")
        return {
            "execution_count": self._state_manager.get("execution_count", 0),
            "cache_size": len(self._state_manager.get("result_cache", {})),
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "model_name": model.name if model else None,
        }

    def clear_cache(self) -> None:
        """Clear the generator result cache."""
        self._state_manager.update("result_cache", {})
        if logger:
            logger.debug("Generator cache cleared")
