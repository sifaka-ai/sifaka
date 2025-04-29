"""Critics functionality for Sifaka."""

from .base import Critic, CriticResult
from .prompt import PromptCritic, PromptCriticConfig
from .reflexion import ReflexionCritic, ReflexionCriticConfig, create_reflexion_critic

# Default configuration for prompt critics
DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="Default Prompt Critic",
    description="Default text evaluation using language models",
    system_prompt="You are an expert editor that improves text quality.",
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
)

# Default configuration for reflexion critics
DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="Default Reflexion Critic",
    description="Evaluates and improves text using reflections on past feedback",
    system_prompt="You are an expert editor that improves text through reflection.",
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    priority=1,
    cost=1.0,
    memory_buffer_size=5,
    reflection_depth=1,
)

__all__ = [
    "Critic",
    "CriticResult",
    "PromptCritic",
    "PromptCriticConfig",
    "ReflexionCritic",
    "ReflexionCriticConfig",
    "create_reflexion_critic",
    "DEFAULT_PROMPT_CONFIG",
    "DEFAULT_REFLEXION_CONFIG",
]
