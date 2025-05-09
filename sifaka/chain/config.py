"""
Chain Configuration Module

Standardized configuration classes for Sifaka's chain system.

## Overview
This module provides standardized configuration classes for chains,
following the same pattern as RuleConfig and ClassifierConfig. These
classes provide a consistent way to configure different aspects of
the chain system, including chain behavior, retry strategies, and
validation management.

## Components
1. **ChainConfig**: Base configuration for chains
2. **RetryConfig**: Configuration for retry strategies
3. **BackoffRetryConfig**: Configuration for backoff retry strategies
4. **ValidationConfig**: Configuration for validation managers

## Usage Examples
```python
from sifaka.chain.config import ChainConfig, RetryConfig, BackoffRetryConfig, ValidationConfig

# Create chain configuration
chain_config = ChainConfig(
    max_attempts=3,
    trace_enabled=True,
    params={
        "system_prompt": "You are a helpful assistant.",
        "use_critic": True,
    }
)

# Create retry configuration
retry_config = RetryConfig(
    max_attempts=3,
    params={
        "use_backoff": True,
    }
)

# Create backoff retry configuration
backoff_config = BackoffRetryConfig(
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0,
    params={
        "jitter": True,
    }
)

# Create validation configuration
validation_config = ValidationConfig(
    prioritize_by_cost=True,
    params={
        "fail_fast": True,
    }
)

# Use configurations
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    config=chain_config
)

strategy = BackoffRetryStrategy(config=backoff_config)
manager = ValidationManager(rules=rules, config=validation_config)
```

## Error Handling
- ValueError: Raised when configuration values are invalid
- ValidationError: Raised when configuration validation fails

## Configuration
- ChainConfig:
  - max_attempts: Maximum number of generation attempts
  - trace_enabled: Whether to enable tracing
  - params: Dictionary of chain-specific parameters

- RetryConfig:
  - max_attempts: Maximum number of retry attempts
  - params: Dictionary of strategy-specific parameters

- BackoffRetryConfig:
  - max_attempts: Maximum number of retry attempts
  - initial_backoff: Initial backoff time in seconds
  - backoff_factor: Factor to multiply backoff by each attempt
  - max_backoff: Maximum backoff time in seconds
  - params: Dictionary of strategy-specific parameters

- ValidationConfig:
  - prioritize_by_cost: Whether to prioritize rules by cost
  - params: Dictionary of manager-specific parameters
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChainConfig(BaseModel):
    """
    Immutable configuration for chains.

    Detailed description of what the class does, including:
    - Provides a consistent way to configure chains across the Sifaka framework
    - Handles common configuration options like max_attempts
    - Allows chain-specific options through the params dictionary

    Attributes:
        max_attempts (int): Maximum number of generation attempts
        trace_enabled (bool): Whether to enable tracing
        params (Dict[str, Any]): Dictionary of chain-specific configuration parameters

    Example:
        ```python
        from sifaka.chain.config import ChainConfig

        # Create a basic chain configuration
        config = ChainConfig(
            max_attempts=3,
            trace_enabled=True,
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

        Detailed description of what the method does, including:
        - Creates a new configuration with updated top-level options
        - Preserves existing params dictionary
        - Maintains immutability

        Args:
            **kwargs: Options to update

        Returns:
            ChainConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=5)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "ChainConfig":
        """
        Create a new config with updated parameters.

        Detailed description of what the method does, including:
        - Creates a new configuration with updated params
        - Preserves existing top-level options
        - Maintains immutability

        Args:
            **kwargs: Parameters to update

        Returns:
            ChainConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(system_prompt="You are an expert coder.")
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class RetryConfig(BaseModel):
    """
    Immutable configuration for retry strategies.

    Detailed description of what the class does, including:
    - Provides a consistent way to configure retry strategies across the Sifaka framework
    - Handles common configuration options like max_attempts
    - Allows strategy-specific options through the params dictionary

    Attributes:
        max_attempts (int): Maximum number of retry attempts
        params (Dict[str, Any]): Dictionary of strategy-specific configuration parameters

    Example:
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

        Detailed description of what the method does, including:
        - Creates a new configuration with updated top-level options
        - Preserves existing params dictionary
        - Maintains immutability

        Args:
            **kwargs: Options to update

        Returns:
            RetryConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=5)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "RetryConfig":
        """
        Create a new config with updated parameters.

        Detailed description of what the method does, including:
        - Creates a new configuration with updated params
        - Preserves existing top-level options
        - Maintains immutability

        Args:
            **kwargs: Parameters to update

        Returns:
            RetryConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(use_backoff=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class BackoffRetryConfig(RetryConfig):
    """
    Configuration for backoff retry strategies.

    Detailed description of what the class does, including:
    - Extends RetryConfig with backoff-specific configuration options
    - Provides a consistent way to configure exponential backoff retry strategies
    - Allows backoff-specific options through the params dictionary

    Attributes:
        initial_backoff (float): Initial backoff time in seconds
        backoff_factor (float): Factor to multiply backoff by each attempt
        max_backoff (float): Maximum backoff time in seconds

    Example:
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

        # Create a new configuration with updated options
        updated_config = config.with_options(max_attempts=7)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(jitter=False)

        # Access configuration values
        print(f"Max attempts: {config.max_attempts}")
        print(f"Initial backoff: {config.initial_backoff}")
        print(f"Backoff factor: {config.backoff_factor}")
        print(f"Max backoff: {config.max_backoff}")
        print(f"Use jitter: {config.params.get('jitter')}")
        ```
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
        """
        Create a new config with updated options.

        Detailed description of what the method does, including:
        - Creates a new configuration with updated top-level options
        - Preserves existing params dictionary
        - Maintains immutability

        Args:
            **kwargs: Options to update

        Returns:
            BackoffRetryConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=7)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "BackoffRetryConfig":
        """
        Create a new config with updated parameters.

        Detailed description of what the method does, including:
        - Creates a new configuration with updated params
        - Preserves existing top-level options
        - Maintains immutability

        Args:
            **kwargs: Parameters to update

        Returns:
            BackoffRetryConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(jitter=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class ValidationConfig(BaseModel):
    """
    Immutable configuration for validation managers.

    Detailed description of what the class does, including:
    - Provides a consistent way to configure validation managers across the Sifaka framework
    - Handles common configuration options like prioritize_by_cost
    - Allows manager-specific options through the params dictionary

    Attributes:
        prioritize_by_cost (bool): Whether to prioritize rules by cost
        params (Dict[str, Any]): Dictionary of manager-specific configuration parameters

    Example:
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

        Detailed description of what the method does, including:
        - Creates a new configuration with updated top-level options
        - Preserves existing params dictionary
        - Maintains immutability

        Args:
            **kwargs: Options to update

        Returns:
            ValidationConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(prioritize_by_cost=False)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "ValidationConfig":
        """
        Create a new config with updated parameters.

        Detailed description of what the method does, including:
        - Creates a new configuration with updated params
        - Preserves existing top-level options
        - Maintains immutability

        Args:
            **kwargs: Parameters to update

        Returns:
            ValidationConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Example:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(fail_fast=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})
