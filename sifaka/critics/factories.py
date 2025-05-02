"""
Factory functions for creating critics.

This module provides factory functions for creating different types of critics.
"""

from typing import Any

from .models import CriticConfig
from .core import CriticCore
from .managers.memory import MemoryManager
from .managers.prompt_factories import PromptCriticPromptManager, ReflexionCriticPromptManager
from .managers.response import ResponseParser
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_prompt_critic(
    llm_provider: Any,
    config: CriticConfig = None,
) -> CriticCore:
    """
    Create a prompt critic with the given parameters.

    Args:
        llm_provider: Language model provider to use
        config: Optional critic configuration

    Returns:
        A configured prompt critic
    """
    # Create default configuration if none provided
    if config is None:
        config = CriticConfig(
            name="prompt_critic",
            description="Evaluates and improves text using language models",
            min_confidence=0.7,
            max_attempts=3,
        )

    # Create managers
    prompt_manager = PromptCriticPromptManager(config)
    response_parser = ResponseParser()

    # Create critic
    return CriticCore(
        config=config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
    )


def create_reflexion_critic(
    llm_provider: Any,
    config: CriticConfig = None,
    memory_buffer_size: int = 5,
) -> CriticCore:
    """
    Create a reflexion critic with the given parameters.

    Args:
        llm_provider: Language model provider to use
        config: Optional critic configuration
        memory_buffer_size: Maximum number of reflections to store

    Returns:
        A configured reflexion critic
    """
    # Create default configuration if none provided
    if config is None:
        config = CriticConfig(
            name="reflexion_critic",
            description="Improves text using reflections on past feedback",
            min_confidence=0.7,
            max_attempts=3,
        )

    # Create managers
    prompt_manager = ReflexionCriticPromptManager(config)
    response_parser = ResponseParser()
    memory_manager = MemoryManager(buffer_size=memory_buffer_size)

    # Create critic
    return CriticCore(
        config=config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
        memory_manager=memory_manager,
    )
