"""Research-backed critics for text improvement."""

from .core.base import BaseCritic, CriticResponse
from .reflexion import ReflexionCritic
from .constitutional import ConstitutionalCritic
from .self_refine import SelfRefineCritic
from .n_critics import NCriticsCritic
from .self_rag import SelfRAGCritic
from .meta_rewarding import MetaRewardingCritic
from .self_consistency import SelfConsistencyCritic
from .style import StyleCritic, style_critic_from_file
from .prompt import PromptCritic, create_academic_critic
from .core.factory import (
    CriticFactory,
    create_critic,
    create_critics,
    list_available_critics,
    register_critic,
)
from .core.registry import CriticRegistry

__all__ = [
    # Base classes
    "BaseCritic",
    "CriticResponse",
    # Critics
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRefineCritic",
    "NCriticsCritic",
    "SelfRAGCritic",
    "MetaRewardingCritic",
    "SelfConsistencyCritic",
    "StyleCritic",
    "PromptCritic",
    # Factory
    "CriticFactory",
    "create_critic",
    "create_critics",
    "list_available_critics",
    "register_critic",
    # Registry
    "CriticRegistry",
    # Factory functions
    "create_academic_critic",
    "style_critic_from_file",
]
