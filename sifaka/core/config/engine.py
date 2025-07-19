"""Engine-specific configuration."""

from typing import Optional

from pydantic import Field

from .base import BaseConfig


class EngineConfig(BaseConfig):
    """Configuration for the Sifaka engine behavior.

    Controls iteration limits, quality thresholds, debugging options,
    and other engine-level settings.

    Example:
        >>> engine_config = EngineConfig(
        ...     max_iterations=5,
        ...     min_quality_score=0.8,
        ...     parallel_critics=True
        ... )
    """

    # Iteration control
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum number of improvement iterations"
    )

    min_iterations: int = Field(
        default=1, ge=1, le=10, description="Minimum number of iterations to run"
    )

    # Quality thresholds
    min_quality_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to stop iterating",
    )

    force_improvements: bool = Field(
        default=False, description="Always run critics even if validation passes"
    )

    # Performance settings
    parallel_critics: bool = Field(
        default=True, description="Run critics in parallel for better performance"
    )

    max_parallel_critics: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of critics to run in parallel",
    )

    critic_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Timeout for individual critic execution",
    )

    # Timeout settings
    total_timeout_seconds: float = Field(
        default=300.0,
        ge=0.001,
        le=3600,
        description="Maximum total processing time in seconds",
    )

    # Retry configuration
    retry_enabled: bool = Field(
        default=True, description="Enable automatic retry on transient failures"
    )

    retry_max_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts for failures"
    )

    retry_initial_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Initial retry delay in seconds"
    )

    retry_exponential_base: float = Field(
        default=2.0, ge=1.1, le=10.0, description="Exponential backoff multiplier"
    )

    # Debugging options
    show_improvement_prompt: bool = Field(
        default=False, description="Print improvement prompts for debugging"
    )

    show_critic_prompts: bool = Field(
        default=False, description="Print critic prompts for debugging"
    )

    track_token_usage: bool = Field(
        default=True, description="Track token usage for cost monitoring"
    )

    # Middleware settings
    enable_middleware: bool = Field(
        default=True, description="Enable middleware pipeline"
    )

    enable_caching: bool = Field(default=True, description="Enable result caching")

    cache_ttl_seconds: int = Field(
        default=3600, ge=0, description="Cache time-to-live in seconds (0 = no expiry)"
    )

    # Monitoring
    logfire_token: Optional[str] = Field(
        default=None,
        description="Logfire token for monitoring (uses env var if not set)",
    )

    enable_metrics: bool = Field(
        default=True, description="Enable performance metrics collection"
    )

    def should_continue_iterating(
        self, iteration: int, quality_score: Optional[float] = None
    ) -> bool:
        """Determine if the engine should continue iterating.

        Args:
            iteration: Current iteration number (1-based)
            quality_score: Current quality score if available

        Returns:
            True if should continue, False otherwise
        """
        # Always run minimum iterations
        if iteration < self.min_iterations:
            return True

        # Stop at max iterations
        if iteration >= self.max_iterations:
            return False

        # Check quality threshold if available
        if quality_score is not None and quality_score >= self.min_quality_score:
            return False

        return True
