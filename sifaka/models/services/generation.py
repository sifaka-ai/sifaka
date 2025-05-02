"""
Generation service for model providers.

This module provides the GenerationService class which is responsible for
generating text using model providers.
"""

from datetime import datetime
from typing import Dict, Any, Generic, TypeVar

from sifaka.models.base import ModelConfig
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for manager types
C = TypeVar("C", bound=ClientManager)
T = TypeVar("T", bound=TokenCounterManager)


class GenerationService(Generic[C, T]):
    """
    Handles text generation for model providers.

    This class is responsible for generating text using API clients,
    handling errors, and recording metrics.

    Type Parameters:
        C: The client manager type
        T: The token counter manager type

    Lifecycle:
    1. Initialization: Set up the service with necessary managers
    2. Usage: Generate text using the configured managers
    3. Cleanup: Release any resources when no longer needed

    Examples:
        ```python
        # Create a generation service with managers
        service = GenerationService(
            model_name="claude-3-opus",
            client_manager=client_manager,
            token_counter_manager=token_counter_manager,
            tracing_manager=tracing_manager
        )

        # Generate text
        response = service.generate("Explain quantum computing", config)
        ```
    """

    def __init__(
        self,
        model_name: str,
        client_manager: C,
        token_counter_manager: T,
        tracing_manager: TracingManager,
    ):
        """
        Initialize a GenerationService instance.

        Args:
            model_name: The name of the model to use
            client_manager: The client manager to use
            token_counter_manager: The token counter manager to use
            tracing_manager: The tracing manager to use
        """
        self._model_name = model_name
        self._client_manager = client_manager
        self._token_counter_manager = token_counter_manager
        self._tracing_manager = tracing_manager

    def generate(self, prompt: str, config: ModelConfig) -> str:
        """
        Generate text using the model.

        This method coordinates the generation process by:
        1. Validating the prompt
        2. Counting tokens in the prompt
        3. Getting a client from the client manager
        4. Sending the prompt to the client
        5. Recording metrics and trace events

        Args:
            prompt: The prompt to generate from
            config: The model configuration to use

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If an error occurs during generation
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Count tokens before generation
        prompt_tokens = self._token_counter_manager.count_tokens(prompt)
        if prompt_tokens > config.max_tokens:
            logger.warning(
                f"Prompt tokens ({prompt_tokens}) exceed max_tokens ({config.max_tokens})"
            )

        start_time = datetime.now()
        client = self._client_manager.get_client()

        try:
            response = client.send_prompt(prompt, config)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._tracing_manager.trace_event(
                "generate",
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": self._token_counter_manager.count_tokens(response),
                    "duration_ms": duration_ms,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "success": True,
                },
            )

            logger.debug(
                f"Generated response in {duration_ms:.2f}ms "
                f"(prompt: {prompt_tokens} tokens)"
            )

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with {self._model_name}: {error_msg}")

            self._tracing_manager.trace_event(
                "error",
                {
                    "error": error_msg,
                    "prompt_tokens": prompt_tokens,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
            )

            raise RuntimeError(f"Error generating text with {self._model_name}: {error_msg}") from e
