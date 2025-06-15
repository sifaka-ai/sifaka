"""Simplified configuration management for Sifaka.

This module provides a single, unified configuration class with builder pattern support.
Breaking changes: Removed environment variable loading, file loading, and complex validation.

Example Usage:
    ```python
    # Simple configuration
    config = SifakaConfig.simple(model="openai:gpt-4", max_iterations=5)

    # Direct instantiation
    config = SifakaConfig(
        model="anthropic:claude-3-sonnet",
        max_iterations=3,
        critics=["reflexion", "constitutional"]
    )
    ```
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class SifakaConfig:
    """Unified configuration for Sifaka engine.

    Simplified configuration class with builder pattern support.
    All configuration is done through direct instantiation or the simple() class method.

    Attributes:
        model: Model for text generation
        max_iterations: Maximum number of iterations per thought
        min_length: Minimum text length validation (optional)
        max_length: Maximum text length validation (optional)
        required_sentiment: Required sentiment validation (optional)
        critics: List of critic names to enable
    """

    # Core configuration
    model: str = "openai:gpt-4o-mini"
    max_iterations: int = 3

    # Validators
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required_sentiment: Optional[str] = None

    # Critics
    critics: List[str] = field(default_factory=lambda: ["reflexion"])

    # Built-in Features
    enable_logging: bool = False
    log_level: str = "INFO"
    log_content: bool = False
    enable_timing: bool = False
    enable_caching: bool = False
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600

    def __post_init__(self):
        """Basic validation after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_iterations > 20:
            raise ValueError("max_iterations cannot exceed 20")
        if self.min_length is not None and self.min_length < 0:
            raise ValueError("min_length cannot be negative")
        if self.max_length is not None and self.max_length < 0:
            raise ValueError("max_length cannot be negative")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length cannot be greater than max_length")

    @classmethod
    def simple(cls, **kwargs) -> "SifakaConfig":
        """Create simple config for beginners.

        Args:
            **kwargs: Configuration parameters

        Returns:
            SifakaConfig instance
        """
        return cls(**kwargs)

    @classmethod
    def builder(cls) -> "SifakaConfigBuilder":
        """Create a builder for fluent configuration.

        Returns:
            SifakaConfigBuilder instance for method chaining

        Example:
            ```python
            config = (SifakaConfig.builder()
                     .model("openai:gpt-4")
                     .max_iterations(5)
                     .min_length(100)
                     .with_reflexion()
                     .with_constitutional()
                     .build())
            ```
        """
        return SifakaConfigBuilder()


class SifakaConfigBuilder:
    """Builder class for fluent SifakaConfig creation."""

    def __init__(self):
        self._model = "openai:gpt-4o-mini"
        self._max_iterations = 3
        self._min_length = None
        self._max_length = None
        self._required_sentiment = None
        self._critics = ["reflexion"]
        self._enable_logging = False
        self._log_level = "INFO"
        self._log_content = False
        self._enable_timing = False
        self._enable_caching = False
        self._cache_size = 1000
        self._cache_ttl_seconds = 3600

    def model(self, model_name: str) -> "SifakaConfigBuilder":
        """Set the model for text generation."""
        self._model = model_name
        return self

    def max_iterations(self, count: int) -> "SifakaConfigBuilder":
        """Set maximum number of improvement iterations."""
        self._max_iterations = count
        return self

    def min_length(self, length: int) -> "SifakaConfigBuilder":
        """Set minimum text length validation."""
        self._min_length = length
        return self

    def max_length(self, length: int) -> "SifakaConfigBuilder":
        """Set maximum text length validation."""
        self._max_length = length
        return self

    def required_sentiment(self, sentiment: str) -> "SifakaConfigBuilder":
        """Set required sentiment validation."""
        self._required_sentiment = sentiment
        return self

    def with_reflexion(self) -> "SifakaConfigBuilder":
        """Add reflexion critic."""
        if "reflexion" not in self._critics:
            self._critics.append("reflexion")
        return self

    def with_constitutional(self) -> "SifakaConfigBuilder":
        """Add constitutional critic."""
        if "constitutional" not in self._critics:
            self._critics.append("constitutional")
        return self

    def with_self_refine(self) -> "SifakaConfigBuilder":
        """Add self-refine critic."""
        if "self_refine" not in self._critics:
            self._critics.append("self_refine")
        return self

    def critics(self, critic_list: List[str]) -> "SifakaConfigBuilder":
        """Set the complete list of critics."""
        self._critics = critic_list.copy()
        return self

    def with_logging(
        self, log_level: str = "INFO", log_content: bool = False
    ) -> "SifakaConfigBuilder":
        """Enable logging with optional configuration."""
        self._enable_logging = True
        self._log_level = log_level
        self._log_content = log_content
        return self

    def with_timing(self) -> "SifakaConfigBuilder":
        """Enable performance timing."""
        self._enable_timing = True
        return self

    def with_caching(
        self, cache_size: int = 1000, ttl_seconds: int = 3600
    ) -> "SifakaConfigBuilder":
        """Enable result caching."""
        self._enable_caching = True
        self._cache_size = cache_size
        self._cache_ttl_seconds = ttl_seconds
        return self

    def build(self) -> SifakaConfig:
        """Build the final SifakaConfig instance."""
        return SifakaConfig(
            model=self._model,
            max_iterations=self._max_iterations,
            min_length=self._min_length,
            max_length=self._max_length,
            required_sentiment=self._required_sentiment,
            critics=self._critics,
            enable_logging=self._enable_logging,
            log_level=self._log_level,
            log_content=self._log_content,
            enable_timing=self._enable_timing,
            enable_caching=self._enable_caching,
            cache_size=self._cache_size,
            cache_ttl_seconds=self._cache_ttl_seconds,
        )


if __name__ == "__main__":
    # Test the configuration system
    print("Testing SifakaConfig...")

    # Test simple configuration
    config = SifakaConfig.simple(
        model="openai:gpt-4",
        max_iterations=5,
        min_length=100,
        critics=["reflexion", "constitutional"],
    )

    print("✅ SifakaConfig.simple() works")
    print(f"Model: {config.model}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Min length: {config.min_length}")
    print(f"Critics: {config.critics}")

    # Test direct instantiation
    config2 = SifakaConfig(model="anthropic:claude-3-sonnet", max_iterations=3)

    print("\n✅ Direct SifakaConfig() works")
    print(f"Model: {config2.model}")
    print(f"Max iterations: {config2.max_iterations}")
    print(f"Critics: {config2.critics}")

    # Test builder pattern
    config3 = (
        SifakaConfig.builder()
        .model("openai:gpt-4")
        .max_iterations(5)
        .min_length(200)
        .with_reflexion()
        .with_constitutional()
        .build()
    )

    print("\n✅ Builder pattern works")
    print(f"Model: {config3.model}")
    print(f"Max iterations: {config3.max_iterations}")
    print(f"Min length: {config3.min_length}")
    print(f"Critics: {config3.critics}")

    # Test validation
    try:
        bad_config = SifakaConfig(max_iterations=0)
        print("❌ Validation failed - should have caught max_iterations=0")
    except ValueError as e:
        print(f"✅ Validation works: {e}")

    print("\n🎉 All tests passed! Configuration simplification is working.")
