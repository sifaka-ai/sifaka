"""Configuration system for Sifaka text improvement.

This module provides a centralized configuration class that controls all
aspects of Sifaka's behavior. The Config class uses Pydantic for validation
and provides sensible defaults while allowing full customization.

Configuration can be provided in multiple ways:
1. Direct instantiation: Config(model="gpt-4", temperature=0.9)
2. From environment variables (with SIFAKA_ prefix)
3. From configuration files (JSON/YAML)
4. Through the API functions"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
        default=True, description="Enable tool usage for critics that support it"
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

    @classmethod
    def fast(cls) -> "Config":
        """Fast configuration for quick iterations.

        Uses cheaper, faster models with lower quality thresholds.
        Good for drafts, testing, or when speed matters more than quality.

        Returns:
            Config optimized for speed

        Example:
            >>> result = await improve(text, config=Config.fast())
        """
        return cls(
            model="gpt-3.5-turbo",
            critic_model="gpt-3.5-turbo",
            temperature=0.7,
            max_iterations=2,
            timeout_seconds=60,
            critic_timeout_seconds=15,
            force_improvements=False,
        )

    @classmethod
    def quality(cls) -> "Config":
        """High-quality configuration for best results.

        Uses more powerful models with thorough critique.
        Good for final drafts, important content, or when quality matters most.

        Returns:
            Config optimized for quality

        Example:
            >>> result = await improve(text, config=Config.quality())
        """
        return cls(
            model="gpt-4o",
            critic_model="gpt-4o-mini",
            temperature=0.6,
            max_iterations=5,
            timeout_seconds=600,
            critic_timeout_seconds=60,
            force_improvements=True,
            critic_base_confidence=0.8,
        )

    @classmethod
    def creative(cls) -> "Config":
        """Creative configuration for imaginative content.

        Higher temperature and creative-focused critics.
        Good for stories, marketing copy, or creative writing.

        Returns:
            Config optimized for creativity

        Example:
            >>> result = await improve(
            ...     text,
            ...     critics=["reflexion", "style"],
            ...     config=Config.creative()
            ... )
        """
        return cls(
            model="gpt-4o-mini",
            critic_model="gpt-4o-mini",
            temperature=0.9,
            critic_temperature=0.7,
            max_iterations=3,
            force_improvements=True,
        )

    @classmethod
    def research(cls) -> "Config":
        """Research configuration with fact-checking tools.

        Enables tools for fact verification and source checking.
        Good for academic writing, journalism, or fact-heavy content.

        Returns:
            Config optimized for research and accuracy

        Example:
            >>> result = await improve(
            ...     text,
            ...     critics=["self_rag", "constitutional"],
            ...     config=Config.research()
            ... )
        """
        return cls(
            model="gpt-4o",
            critic_model="gpt-4o",
            temperature=0.5,
            max_iterations=4,
            enable_tools=True,
            critic_tool_settings={
                "self_rag": {"enable_tools": True},
                "constitutional": {"enable_tools": True},
                "reflexion": {"enable_tools": True},
            },
            force_improvements=True,
        )

    @classmethod
    def style_transfer(
        cls,
        style_guide: str,
        examples: Optional[List[str]] = None,
        reference_text: Optional[str] = None,
    ) -> "Config":
        """Configuration for style transformation.

        Args:
            style_guide: Description of target style
            examples: Optional example phrases in target style
            reference_text: Optional full text exemplifying the style

        Returns:
            Config optimized for style transformation

        Example:
            >>> config = Config.style_transfer(
            ...     "Professional business email",
            ...     examples=["Per our discussion", "Moving forward"]
            ... )
            >>> result = await improve(text, critics=["style"], config=config)
        """
        return cls(
            model="gpt-4o-mini",
            critic_model="gpt-3.5-turbo",
            temperature=0.7,
            max_iterations=3,
            style_description=style_guide,
            style_examples=examples,
            style_reference_text=reference_text,
            force_improvements=True,
        )
