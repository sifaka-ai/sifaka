"""
Text generation module for Sifaka.

This module provides the Generator class which is responsible for
generating text using model providers.
"""

from typing import Generic, TypeVar, Any, Dict
import time
from pydantic import PrivateAttr

from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class Generator(Generic[OutputType]):
    """
    Handles text generation using model providers.

    This class is responsible for generating text using model providers.
    It provides a consistent interface for text generation across different
    model providers.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(self, model: ModelProvider):
        """
        Initialize a Generator instance.

        Args:
            model: The model provider to use for text generation
        """
        self._model = model

        # Initialize state
        self._state.update("model", model)
        self._state.update("initialized", True)
        self._state.update("generation_count", 0)
        self._state.update("cache", {})

        # Initialize metadata
        self._state.set_metadata("component_type", "generator")
        self._state.set_metadata("model_type", model.__class__.__name__)
        self._state.set_metadata("creation_time", time.time())
        self._state.set_metadata("error_count", 0)
        self._state.set_metadata("total_generation_time", 0)
        self._state.set_metadata("prompt_token_count", 0)
        self._state.set_metadata("completion_token_count", 0)

    def generate(self, prompt: str) -> OutputType:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to generate from

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If generation fails
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Check if we have a cached result
        cache = self._state.get("cache", {})
        if prompt in cache:
            self._state.set_metadata("cache_hit", True)
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return cache[prompt]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Track generation count
        generation_count = self._state.get("generation_count", 0)
        self._state.update("generation_count", generation_count + 1)

        # Start timing
        start_time = time.time()

        try:
            # Use the model provider to generate text
            result = self._model.generate(prompt)  # type: ignore

            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update total generation time
            total_time = self._state.get_metadata("total_generation_time", 0)
            self._state.set_metadata("total_generation_time", total_time + execution_time)

            # Update average generation time
            count = self._state.get("generation_count", 1)
            avg_time = total_time / count
            self._state.set_metadata("avg_generation_time", avg_time)

            # Update max generation time if needed
            max_time = self._state.get_metadata("max_generation_time", 0)
            if execution_time > max_time:
                self._state.set_metadata("max_generation_time", execution_time)

            # Update token usage if method available
            if hasattr(self._model, "count_tokens"):
                try:
                    prompt_tokens = self._model.count_tokens(prompt)
                    completion_tokens = 0
                    if isinstance(result, str):
                        completion_tokens = self._model.count_tokens(result)

                    prompt_token_count = self._state.get_metadata("prompt_token_count", 0)
                    completion_token_count = self._state.get_metadata("completion_token_count", 0)

                    self._state.set_metadata(
                        "prompt_token_count", prompt_token_count + prompt_tokens
                    )
                    self._state.set_metadata(
                        "completion_token_count", completion_token_count + completion_tokens
                    )
                except Exception as e:
                    logger.debug(f"Failed to count tokens: {e}")

            # Cache the result (limit cache size)
            cache_size = 100  # Could make this configurable
            if len(cache) >= cache_size:
                # Simple strategy: just clear the cache when it gets full
                cache = {}

            cache[prompt] = result
            self._state.update("cache", cache)

            return result
        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)

            # Track error details
            errors = self._state.get_metadata("errors", {})
            error_type = type(e).__name__
            errors[error_type] = errors.get(error_type, 0) + 1
            self._state.set_metadata("errors", errors)

            error_msg = str(e)
            logger.error(f"Error generating text: {error_msg}")
            raise RuntimeError(f"Error generating text: {error_msg}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about text generation.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "generation_count": self._state.get("generation_count", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "avg_generation_time": self._state.get_metadata("avg_generation_time", 0),
            "max_generation_time": self._state.get_metadata("max_generation_time", 0),
            "total_generation_time": self._state.get_metadata("total_generation_time", 0),
            "prompt_token_count": self._state.get_metadata("prompt_token_count", 0),
            "completion_token_count": self._state.get_metadata("completion_token_count", 0),
            "cache_size": len(self._state.get("cache", {})),
            "model_type": self._state.get_metadata("model_type", "unknown"),
            "errors": self._state.get_metadata("errors", {}),
        }

    def clear_cache(self) -> None:
        """Clear the generation result cache."""
        self._state.update("cache", {})
