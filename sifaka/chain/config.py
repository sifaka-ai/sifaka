"""
Configuration classes for chains.

This module provides standardized configuration classes for chains,
following the same pattern as RuleConfig and ClassifierConfig.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChainConfig(BaseModel):
    """
    Immutable configuration for chains.

    This class provides a consistent way to configure chains across the Sifaka framework.
    It handles common configuration options like max_attempts, while
    allowing chain-specific options through the params dictionary.

    Lifecycle:
        1. Creation: Instantiated with configuration options
        2. Usage: Accessed by chains during setup and execution
        3. Modification: New instances created with updated options (immutable pattern)
        4. Extension: Specialized config classes can extend this base class

    Examples:
        ```python
        from sifaka.chain.config import ChainConfig

        # Create a basic chain configuration
        config = ChainConfig(
            max_attempts=3,
            params={
                "system_prompt": "You are a helpful assistant.",
                "use_critic": True,
            }
        )

        # Use the configuration with a chain
        chain = ChainOrchestrator(
            model=model,
            rules=rules,
            config=config
        )

        # Create a new configuration with updated options
        updated_config = config.with_options(max_attempts=5)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(system_prompt="You are an expert coder.")

        # Access configuration values
        print(f"Max attempts: {config.max_attempts}")
        print(f"System prompt: {config.params.get('system_prompt')}")
        ```

    Attributes:
        max_attempts: Maximum number of generation attempts
        trace_enabled: Whether to enable tracing
        params: Dictionary of chain-specific configuration parameters
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of generation attempts",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chain-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "ChainConfig":
        """
        Create a new config with updated options.

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return ChainConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "ChainConfig":
        """
        Create a new config with updated params.

        This method is useful for updating or adding chain-specific
        parameters without modifying other configuration options.

        Args:
            **kwargs: Params to update

        Returns:
            New config with updated params
        """
        return ChainConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class RetryConfig(BaseModel):
    """
    Immutable configuration for retry strategies.

    This class provides a consistent way to configure retry strategies across the Sifaka framework.
    It handles common configuration options like max_attempts, while
    allowing strategy-specific options through the params dictionary.

    Lifecycle:
        1. Creation: Instantiated with configuration options
        2. Usage: Accessed by retry strategies during setup and execution
        3. Modification: New instances created with updated options (immutable pattern)
        4. Extension: Specialized config classes can extend this base class

    Examples:
        ```python
        from sifaka.chain.config import RetryConfig

        # Create a basic retry configuration
        config = RetryConfig(
            max_attempts=3,
            params={
                "use_backoff": True,
            }
        )

        # Use the configuration with a retry strategy
        strategy = SimpleRetryStrategy(config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(max_attempts=5)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(use_backoff=False)

        # Access configuration values
        print(f"Max attempts: {config.max_attempts}")
        print(f"Use backoff: {config.params.get('use_backoff')}")
        ```

    Attributes:
        max_attempts: Maximum number of retry attempts
        params: Dictionary of strategy-specific configuration parameters
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "RetryConfig":
        """
        Create a new config with updated options.

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return RetryConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "RetryConfig":
        """
        Create a new config with updated params.

        This method is useful for updating or adding strategy-specific
        parameters without modifying other configuration options.

        Args:
            **kwargs: Params to update

        Returns:
            New config with updated params
        """
        return RetryConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class BackoffRetryConfig(RetryConfig):
    """
    Configuration for backoff retry strategies.

    This class extends RetryConfig with backoff-specific configuration options.

    Examples:
        ```python
        from sifaka.chain.config import BackoffRetryConfig

        # Create a backoff retry configuration
        config = BackoffRetryConfig(
            max_attempts=5,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0,
            params={
                "jitter": True,
            }
        )

        # Use the configuration with a backoff retry strategy
        strategy = BackoffRetryStrategy(config=config)
        ```

    Attributes:
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff by each attempt
        max_backoff: Maximum backoff time in seconds
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_backoff: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial backoff time in seconds",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Factor to multiply backoff by each attempt",
    )
    max_backoff: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum backoff time in seconds",
    )

    def with_options(self, **kwargs: Any) -> "BackoffRetryConfig":
        """Create a new config with updated options."""
        return BackoffRetryConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "BackoffRetryConfig":
        """Create a new config with updated params."""
        return BackoffRetryConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class ValidationConfig(BaseModel):
    """
    Immutable configuration for validation managers.

    This class provides a consistent way to configure validation managers across the Sifaka framework.
    It handles common configuration options like prioritize_by_cost, while
    allowing manager-specific options through the params dictionary.

    Lifecycle:
        1. Creation: Instantiated with configuration options
        2. Usage: Accessed by validation managers during setup and execution
        3. Modification: New instances created with updated options (immutable pattern)
        4. Extension: Specialized config classes can extend this base class

    Examples:
        ```python
        from sifaka.chain.config import ValidationConfig

        # Create a basic validation configuration
        config = ValidationConfig(
            prioritize_by_cost=True,
            params={
                "fail_fast": True,
            }
        )

        # Use the configuration with a validation manager
        manager = ValidationManager(rules=rules, config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(prioritize_by_cost=False)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(fail_fast=False)

        # Access configuration values
        print(f"Prioritize by cost: {config.prioritize_by_cost}")
        print(f"Fail fast: {config.params.get('fail_fast')}")
        ```

    Attributes:
        prioritize_by_cost: Whether to prioritize rules by cost
        params: Dictionary of manager-specific configuration parameters
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize rules by cost",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Manager-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "ValidationConfig":
        """
        Create a new config with updated options.

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return ValidationConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "ValidationConfig":
        """
        Create a new config with updated params.

        This method is useful for updating or adding manager-specific
        parameters without modifying other configuration options.

        Args:
            **kwargs: Params to update

        Returns:
            New config with updated params
        """
        return ValidationConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )
