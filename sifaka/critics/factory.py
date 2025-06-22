"""Simple factory for creating critics."""

from typing import Dict, Type, Optional, List, Union
from ..core.llm_client import Provider
from .base import CriticConfig, BaseCritic
from .reflexion import ReflexionCritic
from .constitutional import ConstitutionalCritic
from .self_refine import SelfRefineCritic
from .n_critics import NCriticsCritic
from .self_rag import SelfRAGCritic
from .meta_rewarding import MetaRewardingCritic
from .self_consistency import SelfConsistencyCritic
from .prompt import PromptCritic

# Registry of available critics
CRITIC_REGISTRY: Dict[str, Type[BaseCritic]] = {
    "reflexion": ReflexionCritic,
    "constitutional": ConstitutionalCritic,
    "self_refine": SelfRefineCritic,
    "n_critics": NCriticsCritic,
    "self_rag": SelfRAGCritic,
    "meta_rewarding": MetaRewardingCritic,
    "self_consistency": SelfConsistencyCritic,
    "prompt": PromptCritic,
}


def create_critic(
    name: str,
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None,
    config: Optional[CriticConfig] = None,
    provider: Optional[Union[str, Provider]] = None,
    api_key: Optional[str] = None,
) -> BaseCritic:
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
    if name not in CRITIC_REGISTRY:
        available = ", ".join(sorted(CRITIC_REGISTRY.keys()))
        raise ValueError(f"Unknown critic: '{name}'. Available: {available}")
    
    critic_class = CRITIC_REGISTRY[name]
    return critic_class(
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
    config: Optional[CriticConfig] = None,
    provider: Optional[Union[str, Provider]] = None,
    api_key: Optional[str] = None,
) -> List[BaseCritic]:
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
    return sorted(CRITIC_REGISTRY.keys())


# For backwards compatibility
class CriticFactory:
    """Factory class for backwards compatibility."""
    
    @classmethod
    def create(cls, critic_name: str, **kwargs) -> BaseCritic:
        return create_critic(critic_name, **kwargs)
    
    @classmethod
    def create_multiple(cls, critic_names: List[str], **kwargs) -> List[BaseCritic]:
        return create_critics(critic_names, **kwargs)
    
    @classmethod
    def list_critics(cls) -> List[str]:
        return list_available_critics()