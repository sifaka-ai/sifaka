"""
Compatibility module for Sifaka.

This module provides backward compatibility with the old Chain class.
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..models.base import ModelProvider
from ..rules import Rule
from .core import ChainCore
from .factories import create_simple_chain
from .result import ChainResult
from ..utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class Chain(Generic[OutputType]):
    """
    Backward compatibility class for the old Chain class.
    
    This class provides backward compatibility with the old Chain class
    by delegating to the new ChainCore class.
    """
    
    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a Chain instance.
        
        Args:
            model: The model provider to use for generation
            rules: The rules to validate against
            critic: Optional critic for improving outputs
            max_attempts: Maximum number of attempts
        """
        # Create a ChainCore instance
        self._chain = create_simple_chain(
            model=model,
            rules=rules,
            critic=critic,
            max_attempts=max_attempts,
        )
        
        # Store components for backward compatibility
        self.model = model
        self.rules = rules
        self.critic = critic
        self.max_attempts = max_attempts
        
    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the prompt through the chain.
        
        Args:
            prompt: The input prompt to process
            
        Returns:
            ChainResult containing the output and validation details
            
        Raises:
            ValueError: If validation fails after max attempts
        """
        return self._chain.run(prompt)
