"""Unified configuration for Sifaka."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class Config(BaseModel):
    """Unified configuration for Sifaka with sensible defaults."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Model settings
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for generation"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature for generation"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for generation"
    )
    
    # Critic model settings (optional, defaults to main model)
    critic_model: Optional[str] = Field(
        default=None,
        description="Model for critics (default: same as model)"
    )
    
    critic_temperature: Optional[float] = Field(
        default=None,
        description="Temperature for critics (default: same as temperature)"
    )
    
    # Iteration settings
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum improvement iterations"
    )
    
    force_improvements: bool = Field(
        default=False,
        description="Always run critics even if validation passes"
    )
    
    show_improvement_prompt: bool = Field(
        default=False,
        description="Print improvement prompts for debugging"
    )
    
    # Critics
    critics: List[str] = Field(
        default_factory=lambda: ["reflexion"],
        description="List of critics to use"
    )
    
    # Timeouts
    timeout_seconds: float = Field(
        default=300.0,
        ge=0.001,
        le=3600,
        description="Maximum processing time in seconds"
    )
    
    # Retry configuration (simplified)
    retry_enabled: bool = Field(
        default=True,
        description="Enable retry logic"
    )
    
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    
    retry_initial_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial retry delay in seconds"
    )
    
    retry_exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=10.0,
        description="Exponential backoff base"
    )