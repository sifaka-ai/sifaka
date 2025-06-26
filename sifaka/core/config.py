"""Unified configuration for Sifaka."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Config(BaseModel):
    """Unified configuration for Sifaka with sensible defaults."""

    model_config = ConfigDict(extra="forbid")

    # Model settings
    model: str = Field(
        default="gpt-4o-mini", description="LLM model to use for generation"
    )

    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature for generation"
    )

    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens for generation"
    )

    # Critic model settings (optional, defaults to main model)
    critic_model: Optional[str] = Field(
        default=None, description="Model for critics (default: same as model)"
    )

    critic_temperature: Optional[float] = Field(
        default=None,
        description="Temperature for critics (default: same as temperature)",
    )

    # Critic-specific settings
    critic_base_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Base confidence level for critic assessments",
    )

    critic_context_window: int = Field(
        default=3, ge=1, le=10, description="Number of previous critiques to consider"
    )

    # Logfire monitoring
    logfire_token: Optional[str] = Field(
        default=None, description="Logfire token for monitoring (optional)"
    )

    # Iteration settings
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum improvement iterations"
    )

    force_improvements: bool = Field(
        default=False, description="Always run critics even if validation passes"
    )

    show_improvement_prompt: bool = Field(
        default=False, description="Print improvement prompts for debugging"
    )

    # Critics
    critics: List[str] = Field(
        default_factory=lambda: ["reflexion"], description="List of critics to use"
    )

    # Timeouts
    timeout_seconds: float = Field(
        default=300.0,
        ge=0.001,
        le=3600,
        description="Maximum processing time in seconds",
    )

    critic_timeout_seconds: float = Field(
        default=30.0,
        ge=10.0,
        le=300.0,
        description="Timeout for individual critic operations",
    )

    # Retry configuration (simplified)
    retry_enabled: bool = Field(default=True, description="Enable retry logic")

    retry_max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts"
    )

    retry_initial_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Initial retry delay in seconds"
    )

    retry_exponential_base: float = Field(
        default=2.0, ge=1.1, le=10.0, description="Exponential backoff base"
    )

    # Special critic configurations
    self_consistency_num_samples: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of evaluation samples for Self-Consistency critic",
    )

    constitutional_principles: Optional[List[str]] = Field(
        default=None, description="Custom principles for Constitutional AI critic"
    )

    # Tool settings
    enable_tools: bool = Field(
        default=False, description="Enable tool usage for critics"
    )

    tool_timeout: float = Field(default=5.0, description="Maximum time for tool calls")

    tool_cache_ttl: int = Field(
        default=3600, description="Tool result cache TTL in seconds"
    )

    # Per-critic tool settings
    critic_tool_settings: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "self_rag": {"enable_tools": True},  # On by default
            "constitutional": {"enable_tools": False},
            "meta_rewarding": {"enable_tools": False},
            "reflexion": {"enable_tools": False},
            "self_consistency": {"enable_tools": False},
            "self_refine": {"enable_tools": False},
        },
        description="Per-critic tool configuration",
    )
