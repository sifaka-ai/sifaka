"""Critics functionality for Sifaka."""

from .base import Critic, CriticResult
from .prompt import PromptCritic, PromptCriticConfig

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

__all__ = [
    "Critic",
    "CriticResult",
    "PromptCritic",
    "PromptCriticConfig",
    "DEFAULT_PROMPT_CONFIG",
]
