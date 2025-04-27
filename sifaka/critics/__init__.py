"""Critics functionality for Sifaka."""

from .base import Critic
from .prompt import PromptCritic

__all__ = [
    "Critic",
    "PromptCritic",
]
