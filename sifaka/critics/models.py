"""
Pydantic models for critics.

This module provides Pydantic models for critic configurations and metadata.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CriticConfig(BaseModel):
    """
    Configuration for critics.

    This model defines the configuration options for critics, including
    name, description, confidence thresholds, and other settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Name of the critic", min_length=1)
    description: str = Field(description="Description of the critic", min_length=1)
    min_confidence: float = Field(
        default=0.7, description="Minimum confidence threshold", ge=0.0, le=1.0
    )
    max_attempts: int = Field(default=3, description="Maximum number of improvement attempts", gt=0)
    cache_size: int = Field(default=100, description="Size of the cache", ge=0)
    priority: int = Field(default=1, description="Priority of the critic", ge=0)
    cost: float = Field(default=1.0, description="Cost of using the critic", ge=0.0)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v.strip():
            raise ValueError("name cannot be empty or whitespace")
        return v

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        """Validate that description is not empty."""
        if not v.strip():
            raise ValueError("description cannot be empty or whitespace")
        return v


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt critics.

    This model extends CriticConfig with prompt-specific settings.
    """

    system_prompt: str = Field(
        default="You are an expert editor that improves text.",
        description="System prompt for the model",
        min_length=1,
    )
    temperature: float = Field(
        default=0.7, description="Temperature for model generation", ge=0.0, le=1.0
    )
    max_tokens: int = Field(default=1000, description="Maximum tokens for model generation", gt=0)

    @field_validator("system_prompt")
    @classmethod
    def system_prompt_must_not_be_empty(cls, v: str) -> str:
        """Validate that system_prompt is not empty."""
        if not v.strip():
            raise ValueError("system_prompt cannot be empty or whitespace")
        return v


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.

    This model extends PromptCriticConfig with reflexion-specific settings.
    """

    memory_buffer_size: int = Field(
        default=5, description="Maximum number of reflections to store", gt=0
    )
    reflection_depth: int = Field(
        default=1, description="How many levels of reflection to perform", gt=0
    )


class CriticMetadata(BaseModel):
    """
    Metadata for critic results.

    This model defines the metadata for critic results, including
    score, feedback, issues, and suggestions.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(description="Score between 0 and 1", ge=0.0, le=1.0)
    feedback: str = Field(description="General feedback")
    issues: List[str] = Field(default_factory=list, description="List of issues")
    suggestions: List[str] = Field(default_factory=list, description="List of suggestions")
    attempt_number: int = Field(default=1, description="Attempt number", gt=0)
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds", ge=0.0
    )
