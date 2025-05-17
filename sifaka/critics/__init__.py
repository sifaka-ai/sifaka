"""
Critics for validating and improving text quality.

This module provides critics for validating and improving various aspects of text quality.
Critics are specialized components that use language models to evaluate and enhance text.
"""

from sifaka.critics.base import Critic
from sifaka.critics.constitutional import ConstitutionalCritic, create_constitutional_critic
from sifaka.critics.n_critics import NCriticsCritic, create_n_critics_critic
from sifaka.critics.prompt import PromptCritic, create_prompt_critic
from sifaka.critics.reflexion import ReflexionCritic, create_reflexion_critic
from sifaka.critics.self_rag import SelfRAGCritic, create_self_rag_critic
from sifaka.critics.self_refine import SelfRefineCritic, create_self_refine_critic

__all__ = [
    # Base critic
    "Critic",
    # Specific critics
    "ConstitutionalCritic",
    "NCriticsCritic",
    "PromptCritic",
    "ReflexionCritic",
    "SelfRAGCritic",
    "SelfRefineCritic",
    # Factory functions
    "create_constitutional_critic",
    "create_n_critics_critic",
    "create_prompt_critic",
    "create_reflexion_critic",
    "create_self_rag_critic",
    "create_self_refine_critic",
]
