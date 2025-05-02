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
    name: str = "prompt_critic",
    description: str = "Evaluates and improves text using language models",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert editor that improves text.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    config: CriticConfig = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a prompt critic with the given parameters.

    This factory function creates a configured prompt critic instance
    that uses a language model to evaluate and improve text.

    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured prompt critic
    """
    # Use provided config or create one from parameters
    if config is None:
        from .models import PromptCriticConfig

        config = PromptCriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Create managers
    prompt_manager = PromptCriticPromptManager(config)
    response_parser = ResponseParser()

    # Create critic - filter out any kwargs not accepted by CriticCore
    core_kwargs = {
        'config': config,
        'llm_provider': llm_provider,
        'prompt_manager': prompt_manager,
        'response_parser': response_parser,
    }

    # Create critic
    return CriticCore(**core_kwargs)


def create_reflexion_critic(
    llm_provider: Any,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert editor that learns from past feedback.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
    config: CriticConfig = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a reflexion critic with the given parameters.

    This factory function creates a configured reflexion critic instance
    that uses a language model to evaluate and improve text, while maintaining
    a memory of past improvements to guide future improvements.

    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        memory_buffer_size: Maximum number of reflections to store
        reflection_depth: How many levels of reflection to perform
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured reflexion critic
    """
    # Use provided config or create one from parameters
    if config is None:
        from .models import ReflexionCriticConfig

        config = ReflexionCriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            memory_buffer_size=memory_buffer_size,
            reflection_depth=reflection_depth,
        )

    # Create managers
    prompt_manager = ReflexionCriticPromptManager(config)
    response_parser = ResponseParser()

    # Use the buffer size from the config (which could be from the provided config parameter)
    buffer_size = getattr(config, 'memory_buffer_size', memory_buffer_size)
    memory_manager = MemoryManager(buffer_size=buffer_size)

    # Create critic - filter out any kwargs not accepted by CriticCore
    core_kwargs = {
        'config': config,
        'llm_provider': llm_provider,
        'prompt_manager': prompt_manager,
        'response_parser': response_parser,
        'memory_manager': memory_manager,
    }

    # Create critic
    return CriticCore(**core_kwargs)
