"""
Critics functionality for Sifaka.

This package provides components for critiquing, validating, and improving text:
- CriticCore: Core implementation with component-based architecture
- Factory functions for creating critics (create_prompt_critic, create_reflexion_critic, create_constitutional_critic, create_self_refine_critic, create_self_rag_critic)
- Specialized managers and services for different aspects of critics
"""

from .base import CriticResultEnum as CriticResult
from .core import CriticCore

# Import configuration
from sifaka.utils.config import (
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
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_REFLEXION_CONFIG,
    DEFAULT_CONSTITUTIONAL_CONFIG,
    DEFAULT_SELF_REFINE_CONFIG,
    DEFAULT_SELF_RAG_CONFIG,
    DEFAULT_FEEDBACK_CONFIG,
    DEFAULT_VALUE_CONFIG,
    DEFAULT_LAC_CONFIG,
)

# Import implementations
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

# Import interfaces from main interfaces directory
from sifaka.interfaces import (
    TextValidator,
    TextImprover,
    TextCritic,
    LLMProvider,
    PromptFactory,
    CritiqueResult,
)

# Import strategies
from .strategies import (
    ImprovementStrategy,
    DefaultImprovementStrategy,
)

# Import utility functions
from .utils import (
    CriticMetadata as CriticMetadataUtil,
    create_critic_metadata,
    create_error_metadata,
    try_critique,
)

# Import managers and services for advanced usage
from sifaka.core.managers.prompt import (
    DefaultPromptManager,
    PromptCriticPromptManager,
    CriticPromptManager as PromptManager,
    ReflexionCriticPromptManager,
)
from .managers import ResponseParser
from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
from .services import CritiqueService

__all__ = [
    # Core components
    "CriticCore",
    "CriticResult",
    # Models
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
    # Utility functions
    "CriticMetadataUtil",
    "create_critic_metadata",
    "create_error_metadata",
    "try_critique",
    # Critics
    "PromptCritic",
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRefineCritic",
    "SelfRAGCritic",
    "FeedbackCritic",
    "ValueCritic",
    "LACCritic",
    # Factory functions
    "create_prompt_critic",
    "create_reflexion_critic",
    "create_constitutional_critic",
    "create_self_refine_critic",
    "create_self_rag_critic",
    "create_feedback_critic",
    "create_value_critic",
    "create_lac_critic",
    # Interfaces
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    "CritiqueResult",
    # Strategies
    "ImprovementStrategy",
    "DefaultImprovementStrategy",
    # Managers
    "DefaultPromptManager",
    "MemoryManager",
    "PromptManager",
    "PromptCriticPromptManager",
    "ReflexionCriticPromptManager",
    "ResponseParser",
    # Services
    "CritiqueService",
    # Default configurations
    "DEFAULT_PROMPT_CONFIG",
    "DEFAULT_REFLEXION_CONFIG",
    "DEFAULT_CONSTITUTIONAL_CONFIG",
    "DEFAULT_SELF_REFINE_CONFIG",
    "DEFAULT_SELF_RAG_CONFIG",
    "DEFAULT_FEEDBACK_CONFIG",
    "DEFAULT_VALUE_CONFIG",
    "DEFAULT_LAC_CONFIG",
]
