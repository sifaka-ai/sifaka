"""
Factory functions for creating chains.

This module provides factory functions for creating different types of chains.
"""

from typing import List, Optional, TypeVar

from ..critics import CriticCore
from ..models.base import ModelProvider
from ..rules import Rule
from .config import ChainConfig
from .implementation import Chain
from .implementations import SimpleChainImplementation, BackoffChainImplementation
from ..utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


def create_simple_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "A simple chain with a fixed number of retries",
) -> Chain[OutputType]:
    """
    Create a simple chain with the given parameters.

    Args:
        model: The model provider to use
        rules: The rules to validate against
        critic: Optional critic to use
        max_attempts: Maximum number of attempts
        name: The name of the chain
        description: The description of the chain

    Returns:
        A configured chain
    """
    # Create config
    config = ChainConfig(
        max_attempts=max_attempts,
        params={
            "use_critic": critic is not None,
        },
    )

    # Create implementation
    implementation = SimpleChainImplementation[OutputType](
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=max_attempts,
    )

    # Create and return chain
    return Chain[OutputType](
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_backoff_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    name: str = "backoff_chain",
    description: str = "A chain with exponential backoff retry strategy",
) -> Chain[OutputType]:
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
        name: The name of the chain
        description: The description of the chain

    Returns:
        A configured chain
    """
    # Create config
    config = ChainConfig(
        max_attempts=max_attempts,
        params={
            "use_critic": critic is not None,
            "initial_backoff": initial_backoff,
            "backoff_factor": backoff_factor,
            "max_backoff": max_backoff,
        },
    )

    # Create implementation
    implementation = BackoffChainImplementation[OutputType](
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
    )

    # Create and return chain
    return Chain[OutputType](
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
