"""
Configuration for critics.

This module imports standardized configuration classes from utils/config.py and
extends them with critic-specific functionality. It provides default configurations
for different critic types.

## Overview
The module provides:
- Default system prompts for each critic type
- Default configurations with sensible defaults
- Configuration validation and type checking

## Components
1. **Configuration Classes** (imported from utils/config.py and models.py)
   - CriticConfig: Base configuration for all critics
   - PromptCriticConfig: Configuration for prompt-based critics
   - ReflexionCriticConfig: Configuration for reflexion critics
   - ConstitutionalCriticConfig: Configuration for constitutional critics
   - SelfRefineCriticConfig: Configuration for self-refine critics
   - SelfRAGCriticConfig: Configuration for self-RAG critics
   - FeedbackCriticConfig: Configuration for feedback critics
   - ValueCriticConfig: Configuration for value critics
   - LACCriticConfig: Configuration for LAC critics

2. **Default Configurations**
   - DEFAULT_PROMPT_CONFIG: Default configuration for prompt-based critics
   - DEFAULT_REFLEXION_CONFIG: Default configuration for reflexion critics
   - DEFAULT_CONSTITUTIONAL_CONFIG: Default configuration for constitutional critics
   - DEFAULT_SELF_REFINE_CONFIG: Default configuration for self-refine critics
   - DEFAULT_SELF_RAG_CONFIG: Default configuration for self-RAG critics
   - DEFAULT_FEEDBACK_CONFIG: Default configuration for feedback critics
   - DEFAULT_VALUE_CONFIG: Default configuration for value critics
   - DEFAULT_LAC_CONFIG: Default configuration for LAC critics

## Usage Examples
```python
from sifaka.critics.config import (
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_REFLEXION_CONFIG
)
from sifaka.critics.models import PromptCriticConfig
from sifaka.utils.config import standardize_critic_config

# Use default configurations
critic = PromptCriticConfig(**DEFAULT_PROMPT_CONFIG.model_dump())

# Create custom configuration
custom_config = PromptCriticConfig(
    name="custom_critic",
    description="A custom prompt-based critic",
    system_prompt="You are a specialized critic",
    temperature=0.5,
    max_tokens=2000
)

# Combine with default configuration
combined_config = PromptCriticConfig(
    **DEFAULT_PROMPT_CONFIG.model_dump(),
    name="combined_critic",
    temperature=0.5
)

# Standardize configuration
std_config = standardize_critic_config(
    config_class=PromptCriticConfig,
    system_prompt="You are a specialized critic",
    temperature=0.5,
    max_tokens=2000
)
```

## Error Handling
The module implements:
- Configuration validation
- Type checking for all fields
- Default value handling
- Required field validation
- Range validation for numeric fields
"""

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
DEFAULT_SYSTEM_PROMPT = (
    """You are a helpful assistant that provides high-quality feedback and improvements for text."""
)

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
