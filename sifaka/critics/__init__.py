"""Critics for iterative improvement in Sifaka.

This module provides research-based critics implemented as PydanticAI agents
with structured output and optional retrieval augmentation support.

Available Critics:
- ReflexionCritic: Implements Shinn et al. 2023 verbal reinforcement learning
- ConstitutionalCritic: Implements Bai et al. 2022 Constitutional AI principles
- SelfRefineCritic: Implements Madaan et al. 2023 iterative self-refinement
- NCriticsCritic: Implements ensemble critique approach from Tian et al. 2023
- SelfRAGCritic: Implements Asai et al. 2023 retrieval-augmented critique
- MetaEvaluationCritic: Inspired by Wu et al. 2024 meta-judging approach
- SelfConsistencyCritic: Implements Wang et al. 2022 self-consistency sampling
- PromptCritic: Configurable prompt-based critic for custom evaluation criteria
"""

from sifaka.critics.base import BaseCritic, CritiqueFeedback
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.meta_rewarding import MetaEvaluationCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic

__all__ = [
    # Base classes
    "BaseCritic",
    "CritiqueFeedback",
    # Research-based critics
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRefineCritic",
    "NCriticsCritic",
    "SelfRAGCritic",
    "MetaEvaluationCritic",
    "SelfConsistencyCritic",
    # Configurable critics
    "PromptCritic",
]
