"""Factory for creating critics using the registry."""

from typing import List, Optional, Type, Union, cast

from ...core.config import Config
from ...core.interfaces import Critic
from ...core.llm_client import Provider
from .base import BaseCritic
from .registry import CriticRegistry


def create_critic(
    name: str,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None,
    config: Optional[Config] = None,
    provider: Optional[Union[str, Provider]] = None,
    api_key: Optional[str] = None,
) -> Critic:
    """Create a critic instance by name.

    Args:
        name: Name of the critic (e.g., "reflexion", "constitutional")
        model: LLM model to use
        temperature: Temperature for generation
        config: Critic configuration
        provider: LLM provider
        api_key: API key for the provider

    Returns:
        Initialized critic instance

    Raises:
        ValueError: If critic name is unknown
    """
    critic_class = CriticRegistry.get(name)
    if not critic_class:
        available = ", ".join(CriticRegistry.list())
        raise ValueError(f"Unknown critic: '{name}'. Available: {available}")

    # Cast to proper type for instantiation
    critic_type = cast(Type[BaseCritic], critic_class)
    return critic_type(
        model=model,
        temperature=temperature,
        config=config,
        provider=provider,
        api_key=api_key,
    )


def create_critics(
    names: List[str],
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None,
    config: Optional[Config] = None,
    provider: Optional[Union[str, Provider]] = None,
    api_key: Optional[str] = None,
) -> List[Critic]:
    """Create multiple critic instances.

    Args:
        names: List of critic names
        model: LLM model to use
        temperature: Temperature for generation
        config: Shared critic configuration
        provider: LLM provider
        api_key: API key for the provider

    Returns:
        List of initialized critic instances
    """
    return [
        create_critic(name, model, temperature, config, provider, api_key)
        for name in names
    ]


def list_available_critics() -> List[str]:
    """Get list of available critic names."""
    return CriticRegistry.list()


def register_critic(
    name: str, critic_class: type, aliases: Optional[List[str]] = None
) -> None:
    """Register a custom critic.

    Args:
        name: Primary name for the critic
        critic_class: The critic class to register
        aliases: Optional alternative names
    """
    CriticRegistry.register(name, critic_class, aliases)
