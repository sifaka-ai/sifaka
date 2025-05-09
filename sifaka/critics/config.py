"""
Configuration for critics.

This module provides configuration classes and utilities for critics in the Sifaka framework.
It centralizes configuration management and provides default configurations for different
critic types.

## Configuration Overview

The module provides several key configuration classes:

1. **CriticConfig**: Base configuration for all critics
2. **PromptCriticConfig**: Configuration for prompt-based critics
3. **ReflexionCriticConfig**: Configuration for reflexion critics
4. **ConstitutionalCriticConfig**: Configuration for constitutional critics
5. **SelfRefineCriticConfig**: Configuration for self-refine critics
6. **SelfRAGCriticConfig**: Configuration for self-RAG critics
7. **LACCriticConfig**: Configuration for LAC critics

## Default Configurations

The module also provides default configurations for different critic types:

1. **DEFAULT_PROMPT_CONFIG**: Default configuration for prompt-based critics
2. **DEFAULT_REFLEXION_CONFIG**: Default configuration for reflexion critics
3. **DEFAULT_CONSTITUTIONAL_CONFIG**: Default configuration for constitutional critics
4. **DEFAULT_SELF_REFINE_CONFIG**: Default configuration for self-refine critics
5. **DEFAULT_SELF_RAG_CONFIG**: Default configuration for self-RAG critics
6. **DEFAULT_LAC_CONFIG**: Default configuration for LAC critics
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

# Import models from models.py for backward compatibility
from .models import (
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
)

# Default system prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text."""

DEFAULT_REFLEXION_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text, using reflections on past feedback to guide your improvements."""

DEFAULT_CONSTITUTIONAL_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text, ensuring that the text adheres to a set of principles."""

DEFAULT_SELF_REFINE_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text, using self-refinement to iteratively improve your responses."""

DEFAULT_SELF_RAG_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text, using retrieval-augmented generation to enhance your responses with relevant information."""

DEFAULT_LAC_SYSTEM_PROMPT = """You are a helpful assistant that provides high-quality feedback and improvements for text, using learned alignment from human feedback to guide your improvements."""

# Default configurations
DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="prompt_critic",
    description="A critic that uses prompts to improve text",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
)

DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="reflexion_critic",
    description="A critic that uses reflections to improve text",
    system_prompt=DEFAULT_REFLEXION_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    memory_buffer_size=5,
    reflection_depth=1,
)

DEFAULT_CONSTITUTIONAL_CONFIG = ConstitutionalCriticConfig(
    name="constitutional_critic",
    description="A critic that uses principles to improve text",
    system_prompt=DEFAULT_CONSTITUTIONAL_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    principles=[
        "The text should be clear and concise.",
        "The text should be grammatically correct.",
        "The text should be well-structured.",
        "The text should be factually accurate.",
        "The text should be appropriate for the intended audience.",
    ],
)

DEFAULT_SELF_REFINE_CONFIG = SelfRefineCriticConfig(
    name="self_refine_critic",
    description="A critic that uses self-refinement to improve text",
    system_prompt=DEFAULT_SELF_REFINE_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    max_iterations=3,
    improvement_threshold=0.1,
)

DEFAULT_SELF_RAG_CONFIG = SelfRAGCriticConfig(
    name="self_rag_critic",
    description="A critic that uses retrieval-augmented generation to improve text",
    system_prompt=DEFAULT_SELF_RAG_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    retrieval_top_k=3,
    reflection_enabled=True,
)

DEFAULT_FEEDBACK_CONFIG = FeedbackCriticConfig(
    name="feedback_critic",
    description="A critic that provides natural language feedback",
    system_prompt=DEFAULT_LAC_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    feedback_dimensions=["clarity", "coherence", "relevance", "accuracy"],
)

DEFAULT_VALUE_CONFIG = ValueCriticConfig(
    name="value_critic",
    description="A critic that provides numeric value scoring",
    system_prompt=DEFAULT_LAC_SYSTEM_PROMPT,
    temperature=0.3,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    value_dimensions=["clarity", "coherence", "relevance", "accuracy"],
    min_score=0.0,
    max_score=1.0,
)

DEFAULT_LAC_CONFIG = LACCriticConfig(
    name="lac_critic",
    description="A critic that uses learned alignment from human feedback",
    system_prompt=DEFAULT_LAC_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    cache_size=100,
    feedback_dimensions=["clarity", "coherence", "relevance", "accuracy"],
    value_dimensions=["clarity", "coherence", "relevance", "accuracy"],
    min_score=0.0,
    max_score=1.0,
)

# Export public configurations
__all__ = [
    # Configuration classes
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
