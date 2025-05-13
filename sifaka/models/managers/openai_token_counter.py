"""
OpenAI token counter manager for model providers.

This module provides the OpenAITokenCounterManager class which is responsible for
managing OpenAI token counters for model providers.
"""

import tiktoken
from typing import Any
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-4") -> Any:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """

        def init_operation() -> Any:
            if "gpt-4" in model:
                encoding_name = "cl100k_base"
            elif "gpt-3.5" in model:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "p50k_base"
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug(
                f"Initialized token counter for model {model} with encoding {encoding_name}"
            )
            return self.encoding

        from sifaka.utils.errors.safe_execution import safely_execute_component
        from sifaka.utils.errors.component import ModelError

        safely_execute_component(
            operation=init_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            error_class=ModelError,
            additional_metadata={"model_name": model},
        )

    def count_tokens(self, text: str) -> Any:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """

        def count_operation() -> Any:
            return len(self.encoding.encode(text))

        from sifaka.utils.errors.safe_execution import safely_execute_component
        from sifaka.utils.errors.component import ModelError

        return safely_execute_component(
            operation=count_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            error_class=ModelError,
            additional_metadata={"model_name": "tiktoken"},
        )


class OpenAITokenCounterManager(TokenCounterManager[OpenAITokenCounter]):
    """
    Manages OpenAI token counters for model providers.

    This class extends the TokenCounterManager to provide OpenAI-specific
    token counter management functionality.
    """

    def _create_default_token_counter(self) -> Any:
        """
        Create a default OpenAI token counter if none was provided.

        Returns:
            A default OpenAI token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        if logger:
            logger.debug(f"Creating default OpenAI token counter for {self._model_name}")
        return OpenAITokenCounter(model=self._model_name)
