"""
OpenAI token counter manager for model providers.

This module provides the OpenAITokenCounterManager class which is responsible for
managing OpenAI token counters for model providers.
"""

import tiktoken
from typing import Optional

from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.utils.errors.safe_execution import safely_execute_component_operation
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-4") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """

        # Define the initialization operation
        def init_operation():
            # Get the appropriate encoding for the model
            if "gpt-4" in model:
                encoding_name = "cl100k_base"
            elif "gpt-3.5" in model:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "p50k_base"  # Default for older models

            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug(
                f"Initialized token counter for model {model} with encoding {encoding_name}"
            )
            return self.encoding

        # Use the standardized safely_execute_component_operation function
        safely_execute_component_operation(
            operation=init_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            additional_metadata={"model_name": model},
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """

        # Define the token counting operation
        def count_operation():
            return len(self.encoding.encode(text))

        # Use the standardized safely_execute_component_operation function
        return safely_execute_component_operation(
            operation=count_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            additional_metadata={"model_name": "tiktoken"},
        )


class OpenAITokenCounterManager(TokenCounterManager[OpenAITokenCounter]):
    """
    Manages OpenAI token counters for model providers.

    This class extends the TokenCounterManager to provide OpenAI-specific
    token counter management functionality.
    """

    def _create_default_token_counter(self) -> OpenAITokenCounter:
        """
        Create a default OpenAI token counter if none was provided.

        Returns:
            A default OpenAI token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        logger.debug(f"Creating default OpenAI token counter for {self._model_name}")
        return OpenAITokenCounter(model=self._model_name)
