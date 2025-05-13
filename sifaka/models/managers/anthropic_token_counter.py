"""
Anthropic token counter manager for model providers.

This module provides the AnthropicTokenCounterManager class which is responsible for
managing Anthropic token counters for model providers.
"""

import time
from typing import Any, Optional
import tiktoken
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicTokenCounter(TokenCounter):
    """
    Token counter using tiktoken for Anthropic models.

    This class provides token counting functionality for Anthropic Claude models
    using the tiktoken library. It uses the cl100k_base encoding which is
    compatible with Claude models.
    """

    def __init__(self, model: str = "claude-3-opus-20240229") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model name to use for token counting
        """
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = model
            if logger:
                logger.debug(f"Initialized token counter for model {model}")
            self._count_calls = 0
            self._total_tokens_counted = 0
            self._error_count = 0
            self._last_count_time: Optional[float] = None
        except Exception as e:
            if logger:
                logger.error(f"Error initializing token counter: {str(e)}")
            raise ValueError(f"Failed to initialize token counter: {str(e)}")

    def count_tokens(self, text: str) -> Any:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If text is not a string
            RuntimeError: If token counting fails
        """
        start_time = time.time()
        self._last_count_time = start_time  # type: ignore
        try:
            if not isinstance(text, str):
                raise ValueError("Text must be a string")
            token_count = len(self.encoding.encode(text))
            self._count_calls += 1
            self._total_tokens_counted += token_count
            return token_count
        except Exception as e:
            self._error_count += 1
            if logger:
                logger.error(f"Error counting tokens: {str(e)}")
            raise RuntimeError(f"Error counting tokens: {str(e)}")
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            if logger:
                logger.debug(f"Token counting completed in {duration_ms:.2f}ms")

    def get_statistics(self) -> Any:
        """
        Get token counter usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "model": self.model,
            "count_calls": self._count_calls,
            "total_tokens_counted": self._total_tokens_counted,
            "error_count": self._error_count,
            "last_count_time": self._last_count_time,
            "average_tokens_per_call": (
                self._total_tokens_counted / self._count_calls if self._count_calls > 0 else 0
            ),
        }


class AnthropicTokenCounterManager(TokenCounterManager[AnthropicTokenCounter]):
    """
    Manages Anthropic token counters for model providers.

    This class extends the TokenCounterManager to provide Anthropic-specific
    token counter management functionality.
    """

    def _create_default_token_counter(self) -> Any:
        """
        Create a default Anthropic token counter if none was provided.

        Returns:
            A default Anthropic token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        if logger:
            logger.debug(f"Creating default Anthropic token counter for {self._model_name}")
        return AnthropicTokenCounter(model=self._model_name)
