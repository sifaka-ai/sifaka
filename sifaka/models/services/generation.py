"""
Generation service for model providers.

This module provides the GenerationService class which is responsible for
generating text using model providers.
"""

from datetime import datetime
from typing import Generic, TypeVar, Any
from sifaka.utils.config.models import ModelConfig
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
C = TypeVar("C", bound=ClientManager)
T = TypeVar("T", bound=TokenCounterManager)


class GenerationService(Generic[C, T]):
    """Handles text generation for model providers."""

    def __init__(
        self,
        model_name: str,
        client_manager: C,
        token_counter_manager: T,
        tracing_manager: TracingManager,
    ):
        """Initialize a GenerationService instance."""
        self._model_name = model_name
        self._client_manager = client_manager
        self._token_counter_manager = token_counter_manager
        self._tracing_manager = tracing_manager

    def generate(self, prompt: str, config: ModelConfig) -> Any:
        """Generate text using the model."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip() if prompt else "":
            raise ValueError("prompt cannot be empty")
        prompt_tokens = (
            self._token_counter_manager.count_tokens(prompt) if self._token_counter_manager else 0
        )
        if prompt_tokens > config.max_tokens:
            logger.warning(
                f'Prompt tokens ({prompt_tokens}) if logger else "" exceed max_tokens ({config.max_tokens})'
            )
        start_time = datetime.now() if datetime else ""
        client = self._client_manager.get_client() if self._client_manager else None
        try:
            response = client.send_prompt(prompt, config) if client else ""
            end_time = datetime.now() if datetime else ""
            duration_ms = (end_time - start_time).total_seconds() * 1000
            if self._tracing_manager:
                self._tracing_manager.trace_event(
                    "generate",
                    {
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": (
                            self._token_counter_manager.count_tokens(response)
                            if self._token_counter_manager
                            else 0
                        ),
                        "duration_ms": duration_ms,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "success": True,
                    },
                )
            logger.debug(
                f'Generated response in {duration_ms:.2f}ms (prompt: {prompt_tokens} tokens) if logger else ""'
            )
            return response
        except Exception as e:
            error_msg = str(e)
            (
                logger.error(f"Error generating text with {self._model_name}: {error_msg}")
                if logger
                else ""
            )
            (
                self._tracing_manager.trace_event(
                    "error",
                    {
                        "error": error_msg,
                        "prompt_tokens": prompt_tokens,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                    },
                )
                if self._tracing_manager
                else None
            )
            raise RuntimeError(f"Error generating text with {self._model_name}: {error_msg}") from e
