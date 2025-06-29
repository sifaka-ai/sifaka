"""Configuration system for Sifaka text improvement.

This module provides a centralized configuration class that controls all
aspects of Sifaka's behavior. The Config class uses Pydantic for validation
and provides sensible defaults while allowing full customization.

Configuration can be provided in multiple ways:
1. Direct instantiation: Config(model="gpt-4", temperature=0.9)
2. From environment variables (with SIFAKA_ prefix)
3. From configuration files (JSON/YAML)
4. Through the API functions"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Config(BaseModel):
    """Central configuration for all Sifaka operations.
    
    This configuration object controls every aspect of the text improvement
    process, from model selection to critic behavior to retry policies.
    All fields have sensible defaults, so a Config() with no arguments
    will work out of the box.
    
    The configuration is organized into logical sections:
    - Model settings: LLM model selection and parameters
    - Critic settings: How critics behave and which models they use
    - Iteration control: How many rounds of improvement to attempt
    - Timeouts and retries: Reliability and performance settings
    - Critic-specific options: Custom settings for individual critics
    
    Example:
        >>> # Basic configuration
        >>> config = Config(model="gpt-4", temperature=0.8)
        >>> 
        >>> # Advanced configuration
        >>> config = Config(
        ...     model="claude-3-opus",
        ...     temperature=0.7,
        ...     critic_model="claude-3-sonnet",  # Cheaper model for critics
        ...     max_iterations=5,
        ...     style_description="Academic writing style"
        ... )
    
    Attributes are grouped below by category for clarity.
    """

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
        default="gpt-3.5-turbo",
        description="Model for critics (default: gpt-3.5-turbo)",
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

    # Style critic settings
    style_reference_text: Optional[str] = Field(
        default=None, description="Reference text for Style critic"
    )

    style_description: Optional[str] = Field(
        default=None, description="Description of target style for Style critic"
    )

    style_examples: Optional[List[str]] = Field(
        default=None, description="Example phrases in target style for Style critic"
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
