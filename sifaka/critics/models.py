"""
Model classes for critics in Sifaka.

This module provides model classes for critics in the Sifaka framework,
including configuration models and result models.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class CriticMetadata(BaseModel):
    """
    Metadata for critic results.

    This class provides a standardized structure for critic metadata,
    including scores, feedback, issues, and suggestions.

    Attributes:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        issues: List of identified issues
        suggestions: List of improvement suggestions
        metadata: Additional metadata
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for the critique (0.0 to 1.0)",
    )
    feedback: str = Field(
        description="Human-readable feedback",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of identified issues",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of improvement suggestions",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class CriticConfig(BaseModel):
    """
    Base configuration for critics.

    This class provides the base configuration for all critics,
    including common parameters like name, description, and cache size.

    Attributes:
        name: Name of the critic
        description: Description of the critic
        cache_size: Size of the cache for critic results
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        description="Name of the critic",
    )
    description: str = Field(
        description="Description of the critic",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of the cache for critic results",
    )


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt-based critics.

    This class extends the base critic configuration with parameters
    specific to prompt-based critics.

    Attributes:
        system_prompt: System prompt for the critic
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens for text generation
        min_confidence: Minimum confidence score for critique
        max_attempts: Maximum number of attempts for critique
    """

    system_prompt: str = Field(
        description="System prompt for the critic",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens for text generation",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for critique",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of attempts for critique",
    )


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.

    This class extends the prompt critic configuration with parameters
    specific to reflexion critics.

    Attributes:
        memory_buffer_size: Size of the memory buffer for reflections
        reflection_depth: Depth of reflections
    """

    memory_buffer_size: int = Field(
        default=5,
        ge=1,
        description="Size of the memory buffer for reflections",
    )
    reflection_depth: int = Field(
        default=1,
        ge=1,
        description="Depth of reflections",
    )


class ConstitutionalCriticConfig(PromptCriticConfig):
    """
    Configuration for constitutional critics.

    This class extends the prompt critic configuration with parameters
    specific to constitutional critics.

    Attributes:
        principles: List of principles to follow
    """

    principles: List[str] = Field(
        default_factory=list,
        description="List of principles to follow",
    )


class SelfRefineCriticConfig(PromptCriticConfig):
    """
    Configuration for self-refine critics.

    This class extends the prompt critic configuration with parameters
    specific to self-refine critics.

    Attributes:
        max_iterations: Maximum number of refinement iterations
        improvement_threshold: Threshold for improvement
    """

    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of refinement iterations",
    )
    improvement_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for improvement",
    )


class SelfRAGCriticConfig(PromptCriticConfig):
    """
    Configuration for self-RAG critics.

    This class extends the prompt critic configuration with parameters
    specific to self-RAG critics.

    Attributes:
        retrieval_top_k: Number of top documents to retrieve
        reflection_enabled: Whether to enable reflection
    """

    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        description="Number of top documents to retrieve",
    )
    reflection_enabled: bool = Field(
        default=True,
        description="Whether to enable reflection",
    )


class FeedbackCriticConfig(PromptCriticConfig):
    """
    Configuration for feedback critics.

    This class extends the prompt critic configuration with parameters
    specific to feedback critics.

    Attributes:
        feedback_dimensions: List of feedback dimensions
    """

    feedback_dimensions: List[str] = Field(
        default_factory=list,
        description="List of feedback dimensions",
    )


class ValueCriticConfig(PromptCriticConfig):
    """
    Configuration for value critics.

    This class extends the prompt critic configuration with parameters
    specific to value critics.

    Attributes:
        value_dimensions: List of value dimensions
        min_score: Minimum score for values
        max_score: Maximum score for values
    """

    value_dimensions: List[str] = Field(
        default_factory=list,
        description="List of value dimensions",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum score for values",
    )
    max_score: float = Field(
        default=1.0,
        description="Maximum score for values",
    )


class LACCriticConfig(PromptCriticConfig):
    """
    Configuration for LAC critics.

    This class extends the prompt critic configuration with parameters
    specific to LAC critics.

    Attributes:
        feedback_dimensions: List of feedback dimensions
        value_dimensions: List of value dimensions
        min_score: Minimum score for values
        max_score: Maximum score for values
    """

    feedback_dimensions: List[str] = Field(
        default_factory=list,
        description="List of feedback dimensions",
    )
    value_dimensions: List[str] = Field(
        default_factory=list,
        description="List of value dimensions",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum score for values",
    )
    max_score: float = Field(
        default=1.0,
        description="Maximum score for values",
    )
