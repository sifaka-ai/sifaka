"""
Token counter manager for model providers.

This module provides the TokenCounterManager class which is responsible for
managing token counting functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional

from sifaka.models.base import TokenCounter
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class TokenCounterManager:
    """
    Manages token counting functionality for model providers.
    
    This class is responsible for creating and managing token counters,
    and providing a consistent interface for token counting.
    """
    
    def __init__(self, model_name: str, token_counter: Optional[TokenCounter] = None):
        """
        Initialize a TokenCounterManager instance.
        
        Args:
            model_name: The name of the model to count tokens for
            token_counter: Optional token counter to use
        """
        self._model_name = model_name
        self._token_counter = token_counter
        
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
            
        Raises:
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
            
        counter = self._ensure_token_counter()
        return counter.count_tokens(text)
        
    def _ensure_token_counter(self) -> TokenCounter:
        """
        Ensure a token counter is available, creating a default one if needed.
        
        Returns:
            The token counter to use
        """
        if self._token_counter is None:
            logger.debug(f"Creating default token counter for {self._model_name}")
            self._token_counter = self._create_default_token_counter()
        return self._token_counter
        
    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.
        
        Returns:
            A default token counter for the model
        """
        ...
