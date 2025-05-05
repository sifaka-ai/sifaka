"""
Critics functionality for Sifaka.

This package provides components for critiquing, validating, and improving text:
- CriticCore: Core implementation with component-based architecture
- Factory functions for creating critics (create_prompt_critic, create_reflexion_critic, create_constitutional_critic, create_self_refine_critic, create_self_rag_critic)
- Specialized managers and services for different aspects of critics
"""

from .base import CriticResult
from .core import CriticCore
from .constitutional import (
    ConstitutionalCritic,
    ConstitutionalCriticConfig,
    create_constitutional_critic,
)
from .factories import create_prompt_critic, create_reflexion_critic
from .lac import (
    FeedbackCritic,
    ValueCritic,
    LACCritic,
    FeedbackCriticConfig,
    ValueCriticConfig,
    LACCriticConfig,
    create_feedback_critic,
    create_value_critic,
    create_lac_critic,
)
from .models import CriticConfig, CriticMetadata, PromptCriticConfig, ReflexionCriticConfig
from .self_refine import (
    SelfRefineCritic,
    SelfRefineCriticConfig,
    create_self_refine_critic,
)
from .self_rag import (
    SelfRAGCritic,
    SelfRAGCriticConfig,
    create_self_rag_critic,
)

# Import managers and services for advanced usage
from .managers import (
    DefaultPromptManager,
    MemoryManager,
    PromptCriticPromptManager,
    PromptManager,
    ReflexionCriticPromptManager,
    ResponseParser,
)
from .services import CritiqueService

# Default configuration for prompt critics
DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="Default Prompt Critic",
    description="Default text evaluation using language models",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert editor that improves text.",
    temperature=0.7,
    max_tokens=1000,
)

# Default configuration for reflexion critics
DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="Default Reflexion Critic",
    description="Evaluates and improves text using reflections on past feedback",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert editor that improves text through reflection.",
    temperature=0.7,
    max_tokens=1000,
    memory_buffer_size=5,
    reflection_depth=1,
)

# Default configuration for constitutional critics
DEFAULT_CONSTITUTIONAL_CONFIG = ConstitutionalCriticConfig(
    name="Default Constitutional Critic",
    description="Evaluates responses against principles",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at evaluating content against principles.",
    temperature=0.7,
    max_tokens=1000,
    principles=[
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
    ],
)

# Default configuration for self-refine critics
DEFAULT_SELF_REFINE_CONFIG = SelfRefineCriticConfig(
    name="Default Self-Refine Critic",
    description="Improves text through iterative self-critique and revision",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at critiquing and revising content.",
    temperature=0.7,
    max_tokens=1000,
    max_iterations=3,
)

# Default configuration for self-rag critics
DEFAULT_SELF_RAG_CONFIG = SelfRAGCriticConfig(
    name="Default Self-RAG Critic",
    description="Improves text through self-reflective retrieval-augmented generation",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at deciding when to retrieve information and reflecting on its relevance.",
    temperature=0.7,
    max_tokens=1000,
    retrieval_threshold=0.5,
)

# Default configuration for feedback critics
DEFAULT_FEEDBACK_CONFIG = FeedbackCriticConfig(
    name="Default Feedback Critic",
    description="Provides natural language feedback for text",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at providing constructive feedback.",
    temperature=0.7,
    max_tokens=1000,
)

# Default configuration for value critics
DEFAULT_VALUE_CONFIG = ValueCriticConfig(
    name="Default Value Critic",
    description="Estimates numeric values for text",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at estimating the quality of responses.",
    temperature=0.3,
    max_tokens=100,
)

# Default configuration for LAC critics
DEFAULT_LAC_CONFIG = LACCriticConfig(
    name="Default LAC Critic",
    description="Combines language feedback and value scoring",
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    system_prompt="You are an expert at evaluating and improving text.",
    temperature=0.7,
    max_tokens=1000,
)

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
    # Critics
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
