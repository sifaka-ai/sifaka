"""
Critics for improving LLM outputs.

This module provides critics that use LLMs to critique and improve text.
"""

from sifaka.critics.base import Critic
from sifaka.critics.constitutional import ConstitutionalCritic, create_constitutional_critic
from sifaka.critics.reflexion import ReflexionCritic, create_reflexion_critic
from sifaka.critics.prompt import PromptCritic, create_prompt_critic
from sifaka.critics.lac import LACCritic, create_lac_critic
from sifaka.critics.self_rag import SelfRAGCritic, create_self_rag_critic
from sifaka.critics.self_refine import SelfRefineCritic, create_self_refine_critic

__all__ = [
    "Critic",
    "ConstitutionalCritic",
    "create_constitutional_critic",
    "ReflexionCritic",
    "create_reflexion_critic",
    "PromptCritic",
    "create_prompt_critic",
    "LACCritic",
    "create_lac_critic",
    "SelfRAGCritic",
    "create_self_rag_critic",
    "SelfRefineCritic",
    "create_self_refine_critic",
]
