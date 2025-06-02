"""Critics module for Sifaka v0.3.0+

This module provides various critics for evaluating and improving generated text.
All critics use PydanticAI agents with structured output and return CriticResult objects.

Key improvements in v0.3.0:
- PydanticAI agents with structured output
- CriticResult objects with rich metadata
- Pure async implementation
- No backward compatibility code
"""

# New PydanticAI-based critics
from sifaka.critics.base_pydantic import PydanticAICritic
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.critics.self_rag import SelfRAGCritic

# Legacy critics (deprecated - will be removed in future versions)
from sifaka.critics.base import BaseCritic
from sifaka.critics.feedback_summarizer import FeedbackSummarizer
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.critics.prompt import PromptCritic
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.critics.self_refine import SelfRefineCritic

__all__ = [
    # New PydanticAI-based critics (recommended)
    "PydanticAICritic",
    "ConstitutionalCritic",
    "MetaRewardingCritic",
    "ReflexionCritic",
    "SelfRefineCritic",
    "SelfRAGCritic",
    # Legacy critics (deprecated)
    "BaseCritic",
    "SelfConsistencyCritic",
    "NCriticsCritic",
    "PromptCritic",
    "FeedbackSummarizer",
]
