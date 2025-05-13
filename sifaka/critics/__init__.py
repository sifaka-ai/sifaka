"""
Critics functionality for Sifaka.

This package provides components for critiquing, validating, and improving text:
- CriticCore: Core implementation with component-based architecture
- Factory functions for creating critics (create_prompt_critic, create_reflexion_critic, create_constitutional_critic, create_self_refine_critic, create_self_rag_critic)
- Specialized managers and services for different aspects of critics
"""

from typing import Any, List
from .base import CriticResultEnum as CriticResult
from .core import CriticCore
from sifaka.utils.config.critics import (
    CriticConfig,
    CriticMetadata,
    PromptCriticConfig,
    ReflexionCriticConfig,
    ConstitutionalCriticConfig,
    SelfRefineCriticConfig,
    SelfRAGCriticConfig,
    FeedbackCriticConfig,
    ValueCriticConfig,
    LACCriticConfig,
    DEFAULT_PROMPT_CRITIC_CONFIG,
    DEFAULT_REFLEXION_CRITIC_CONFIG,
    DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG,
    DEFAULT_SELF_REFINE_CRITIC_CONFIG,
    DEFAULT_SELF_RAG_CRITIC_CONFIG,
    DEFAULT_FEEDBACK_CRITIC_CONFIG,
    DEFAULT_VALUE_CRITIC_CONFIG,
    DEFAULT_LAC_CRITIC_CONFIG,
)
from .implementations import (
    PromptCritic,
    ReflexionCritic,
    ConstitutionalCritic,
    FeedbackCritic,
    ValueCritic,
    LACCritic,
    SelfRefineCritic,
    SelfRAGCritic,
    create_prompt_critic,
    create_reflexion_critic,
    create_constitutional_critic,
    create_feedback_critic,
    create_value_critic,
    create_lac_critic,
    create_self_refine_critic,
    create_self_rag_critic,
)
from sifaka.interfaces import (
    TextValidator,
    TextImprover,
    TextCritic,
    LLMProvider,
    PromptFactory,
    CritiqueResult,
)
from .strategies import ImprovementStrategy, DefaultImprovementStrategy
from .utils import (
    CriticMetadata as CriticMetadataUtil,
    create_critic_metadata,
    create_error_metadata,
    try_critique,
)
from sifaka.core.managers.prompt import (
    DefaultPromptManager,
    PromptCriticPromptManager,
    CriticPromptManager as PromptManager,
    ReflexionCriticPromptManager,
)
from .managers import ResponseParser
from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
from .services import CritiqueService

__all__: List[Any] = [
    "CriticCore",
    "CriticResult",
    "CriticConfig",
    "CriticMetadata",
    "PromptCriticConfig",
    "ReflexionCriticConfig",
    "ConstitutionalCriticConfig",
    "SelfRefineCriticConfig",
    "SelfRAGCriticConfig",
    "FeedbackCriticConfig",
    "ValueCriticConfig",
    "LACCriticConfig",
    "CriticMetadataUtil",
    "create_critic_metadata",
    "create_error_metadata",
    "try_critique",
    "PromptCritic",
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRefineCritic",
    "SelfRAGCritic",
    "FeedbackCritic",
    "ValueCritic",
    "LACCritic",
    "create_prompt_critic",
    "create_reflexion_critic",
    "create_constitutional_critic",
    "create_self_refine_critic",
    "create_self_rag_critic",
    "create_feedback_critic",
    "create_value_critic",
    "create_lac_critic",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    "CritiqueResult",
    "ImprovementStrategy",
    "DefaultImprovementStrategy",
    "DefaultPromptManager",
    "MemoryManager",
    "PromptManager",
    "PromptCriticPromptManager",
    "ReflexionCriticPromptManager",
    "ResponseParser",
    "CritiqueService",
    "DEFAULT_PROMPT_CRITIC_CONFIG",
    "DEFAULT_REFLEXION_CRITIC_CONFIG",
    "DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG",
    "DEFAULT_SELF_REFINE_CRITIC_CONFIG",
    "DEFAULT_SELF_RAG_CRITIC_CONFIG",
    "DEFAULT_FEEDBACK_CRITIC_CONFIG",
    "DEFAULT_VALUE_CRITIC_CONFIG",
    "DEFAULT_LAC_CRITIC_CONFIG",
]
