"""
Factory functions for creating chains.

This module provides factory functions for creating different types of chains.
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..models.base import ModelProvider
from ..rules import Rule
from .core import ChainCore
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .strategies.retry import BackoffRetryStrategy, SimpleRetryStrategy
from ..utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


def create_simple_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
) -> ChainCore[OutputType]:
    """
    Create a simple chain with the given parameters.
    
    Args:
        model: The model provider to use
        rules: The rules to validate against
        critic: Optional critic to use
        max_attempts: Maximum number of attempts
        
    Returns:
        A configured chain
    """
    validation_manager = ValidationManager[OutputType](rules)
    prompt_manager = PromptManager()
    retry_strategy = SimpleRetryStrategy[OutputType](max_attempts=max_attempts)
    result_formatter = ResultFormatter[OutputType]()
    
    return ChainCore[OutputType](
        model=model,
        validation_manager=validation_manager,
        prompt_manager=prompt_manager,
        retry_strategy=retry_strategy,
        result_formatter=result_formatter,
        critic=critic,
    )


def create_backoff_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
) -> ChainCore[OutputType]:
    """
    Create a chain with exponential backoff retry strategy.
    
    Args:
        model: The model provider to use
        rules: The rules to validate against
        critic: Optional critic to use
        max_attempts: Maximum number of attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff by each attempt
        max_backoff: Maximum backoff time in seconds
        
    Returns:
        A configured chain
    """
    validation_manager = ValidationManager[OutputType](rules)
    prompt_manager = PromptManager()
    retry_strategy = BackoffRetryStrategy[OutputType](
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
    )
    result_formatter = ResultFormatter[OutputType]()
    
    return ChainCore[OutputType](
        model=model,
        validation_manager=validation_manager,
        prompt_manager=prompt_manager,
        retry_strategy=retry_strategy,
        result_formatter=result_formatter,
        critic=critic,
    )
