"""
Pydantic models for critics.

This module provides Pydantic models for critic configurations and metadata.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CriticConfig(BaseModel):
    """
    Configuration for critics.
    
    This model defines the configuration options for critics, including
    name, description, confidence thresholds, and other settings.
    """
    
    name: str = Field(description="Name of the critic")
    description: str = Field(description="Description of the critic")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")
    max_attempts: int = Field(default=3, description="Maximum number of improvement attempts")
    cache_size: int = Field(default=100, description="Size of the cache")
    priority: int = Field(default=1, description="Priority of the critic")
    cost: float = Field(default=1.0, description="Cost of using the critic")
    
    @field_validator("name")
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty or whitespace")
        return v
        
    @field_validator("description")
    def description_must_not_be_empty(cls, v: str) -> str:
        """Validate that description is not empty."""
        if not v or not v.strip():
            raise ValueError("description cannot be empty or whitespace")
        return v
        
    @field_validator("min_confidence")
    def min_confidence_must_be_valid(cls, v: float) -> float:
        """Validate that min_confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        return v
        
    @field_validator("max_attempts")
    def max_attempts_must_be_positive(cls, v: int) -> int:
        """Validate that max_attempts is positive."""
        if v < 1:
            raise ValueError("max_attempts must be positive")
        return v
        
    @field_validator("cache_size")
    def cache_size_must_be_non_negative(cls, v: int) -> int:
        """Validate that cache_size is non-negative."""
        if v < 0:
            raise ValueError("cache_size must be non-negative")
        return v
        
    @field_validator("priority")
    def priority_must_be_non_negative(cls, v: int) -> int:
        """Validate that priority is non-negative."""
        if v < 0:
            raise ValueError("priority must be non-negative")
        return v
        
    @field_validator("cost")
    def cost_must_be_non_negative(cls, v: float) -> float:
        """Validate that cost is non-negative."""
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt critics.
    
    This model extends CriticConfig with prompt-specific settings.
    """
    
    system_prompt: str = Field(
        default="You are an expert editor that improves text.",
        description="System prompt for the model",
    )
    temperature: float = Field(default=0.7, description="Temperature for model generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for model generation")
    
    @field_validator("system_prompt")
    def system_prompt_must_not_be_empty(cls, v: str) -> str:
        """Validate that system_prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("system_prompt cannot be empty or whitespace")
        return v
        
    @field_validator("temperature")
    def temperature_must_be_valid(cls, v: float) -> float:
        """Validate that temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("temperature must be between 0 and 1")
        return v
        
    @field_validator("max_tokens")
    def max_tokens_must_be_positive(cls, v: int) -> int:
        """Validate that max_tokens is positive."""
        if v < 1:
            raise ValueError("max_tokens must be positive")
        return v


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.
    
    This model extends PromptCriticConfig with reflexion-specific settings.
    """
    
    memory_buffer_size: int = Field(default=5, description="Maximum number of reflections to store")
    reflection_depth: int = Field(default=1, description="How many levels of reflection to perform")
    
    @field_validator("memory_buffer_size")
    def memory_buffer_size_must_be_positive(cls, v: int) -> int:
        """Validate that memory_buffer_size is positive."""
        if v < 1:
            raise ValueError("memory_buffer_size must be positive")
        return v
        
    @field_validator("reflection_depth")
    def reflection_depth_must_be_positive(cls, v: int) -> int:
        """Validate that reflection_depth is positive."""
        if v < 1:
            raise ValueError("reflection_depth must be positive")
        return v


class CriticMetadata(BaseModel):
    """
    Metadata for critic results.
    
    This model defines the metadata for critic results, including
    score, feedback, issues, and suggestions.
    """
    
    score: float = Field(description="Score between 0 and 1")
    feedback: str = Field(description="General feedback")
    issues: List[str] = Field(default_factory=list, description="List of issues")
    suggestions: List[str] = Field(default_factory=list, description="List of suggestions")
    attempt_number: int = Field(default=1, description="Attempt number")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    
    @field_validator("score")
    def score_must_be_valid(cls, v: float) -> float:
        """Validate that score is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("score must be between 0 and 1")
        return v
        
    @field_validator("attempt_number")
    def attempt_number_must_be_positive(cls, v: int) -> int:
        """Validate that attempt_number is positive."""
        if v < 1:
            raise ValueError("attempt_number must be positive")
        return v
        
    @field_validator("processing_time_ms")
    def processing_time_ms_must_be_non_negative(cls, v: float) -> float:
        """Validate that processing_time_ms is non-negative."""
        if v < 0:
            raise ValueError("processing_time_ms must be non-negative")
        return v
