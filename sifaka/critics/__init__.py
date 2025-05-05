"""
Critics functionality for Sifaka.

This package provides components for critiquing, validating, and improving text:
- CriticCore: Core implementation with component-based architecture
- Factory functions for creating critics (create_prompt_critic, create_reflexion_critic, create_constitutional_critic)
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
from .models import CriticConfig, CriticMetadata, PromptCriticConfig, ReflexionCriticConfig

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
    # Critics
    "ConstitutionalCritic",
    # Factory functions
    "create_prompt_critic",
    "create_reflexion_critic",
    "create_constitutional_critic",
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
]
