"""Research-backed critics for text improvement."""

from .base import BaseCritic, CriticConfig, CriticResponse
from .reflexion import ReflexionCritic
from .constitutional import ConstitutionalCritic
from .self_refine import SelfRefineCritic
from .n_critics import NCriticsCritic
from .self_rag import SelfRAGCritic
from .meta_rewarding import MetaRewardingCritic
from .self_consistency import SelfConsistencyCritic
from .prompt import PromptCritic, create_academic_critic
from .factory import CriticFactory, create_critic, create_critics, list_available_critics

__all__ = [
    # Base classes
    "BaseCritic",
    "CriticConfig",
    "CriticResponse",
    # Critics
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRefineCritic",
    "NCriticsCritic",
    "SelfRAGCritic",
    "MetaRewardingCritic",
    "SelfConsistencyCritic",
    "PromptCritic",
    # Factory
    "CriticFactory",
    "create_critic",
    "create_critics",
    "list_available_critics",
    # Factory functions
    "create_academic_critic",
]
