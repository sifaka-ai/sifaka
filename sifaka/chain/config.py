"""
Chain Configuration Module

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

    ## Overview
    This class provides a consistent way to configure chains across the Sifaka framework.
    It handles common configuration options like max_attempts and allows chain-specific
    options through the params dictionary.

    ## Architecture
    ChainConfig follows a configuration pattern:
    1. **Base Options**: Common configuration options
       - Maximum attempts
       - Tracing settings
    2. **Custom Parameters**: Chain-specific options
       - System prompts
       - Critic settings
       - Other parameters

    ## Lifecycle
    1. **Creation**: Create configuration
       - Set base options
       - Set custom parameters
    2. **Modification**: Update configuration
       - Update options
       - Update parameters
    3. **Usage**: Apply configuration
       - Use with chain
       - Access values

    ## Error Handling
    - ValueError: Raised when configuration values are invalid
    - ValidationError: Raised when configuration validation fails

    ## Examples
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

    Attributes:
        max_attempts (int): Maximum number of generation attempts
        trace_enabled (bool): Whether to enable tracing
        params (Dict[str, Any]): Dictionary of chain-specific configuration parameters
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

        ## Overview
        This method creates a new configuration with updated top-level options,
        preserving the existing params dictionary and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate options
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update options
           - Preserve params

        Args:
            **kwargs: Options to update

        Returns:
            ChainConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=5)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "ChainConfig":
        """
        Create a new config with updated parameters.

        ## Overview
        This method creates a new configuration with updated params,
        preserving existing top-level options and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate parameters
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update params
           - Preserve options

        Args:
            **kwargs: Parameters to update

        Returns:
            ChainConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(system_prompt="You are an expert coder.")
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class RetryConfig(BaseModel):
    """
    Immutable configuration for retry strategies.

    ## Overview
    This class provides a consistent way to configure retry strategies across
    the Sifaka framework. It handles common configuration options like max_attempts
    and allows strategy-specific options through the params dictionary.

    ## Architecture
    RetryConfig follows a configuration pattern:
    1. **Base Options**: Common configuration options
       - Maximum attempts
       - Basic settings
    2. **Custom Parameters**: Strategy-specific options
       - Retry behavior
       - Other parameters

    ## Lifecycle
    1. **Creation**: Create configuration
       - Set base options
       - Set custom parameters
    2. **Modification**: Update configuration
       - Update options
       - Update parameters
    3. **Usage**: Apply configuration
       - Use with strategy
       - Access values

    ## Error Handling
    - ValueError: Raised when configuration values are invalid
    - ValidationError: Raised when configuration validation fails

    ## Examples
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
        max_attempts (int): Maximum number of retry attempts
        params (Dict[str, Any]): Dictionary of strategy-specific configuration parameters
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

        ## Overview
        This method creates a new configuration with updated top-level options,
        preserving the existing params dictionary and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate options
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update options
           - Preserve params

        Args:
            **kwargs: Options to update

        Returns:
            RetryConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=5)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "RetryConfig":
        """
        Create a new config with updated parameters.

        ## Overview
        This method creates a new configuration with updated params,
        preserving existing top-level options and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate parameters
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update params
           - Preserve options

        Args:
            **kwargs: Parameters to update

        Returns:
            RetryConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(use_backoff=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class BackoffRetryConfig(RetryConfig):
    """
    Configuration for backoff retry strategies.

    ## Overview
    This class extends RetryConfig with backoff-specific configuration options,
    providing a consistent way to configure exponential backoff retry strategies
    and allowing backoff-specific options through the params dictionary.

    ## Architecture
    BackoffRetryConfig follows a configuration pattern:
    1. **Base Options**: Common configuration options
       - Maximum attempts
       - Basic settings
    2. **Backoff Options**: Backoff-specific options
       - Initial backoff
       - Backoff factor
       - Maximum backoff
    3. **Custom Parameters**: Strategy-specific options
       - Jitter settings
       - Other parameters

    ## Lifecycle
    1. **Creation**: Create configuration
       - Set base options
       - Set backoff options
       - Set custom parameters
    2. **Modification**: Update configuration
       - Update options
       - Update parameters
    3. **Usage**: Apply configuration
       - Use with strategy
       - Access values

    ## Error Handling
    - ValueError: Raised when configuration values are invalid
    - ValidationError: Raised when configuration validation fails

    ## Examples
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

    Attributes:
        initial_backoff (float): Initial backoff time in seconds
        backoff_factor (float): Factor to multiply backoff by each attempt
        max_backoff (float): Maximum backoff time in seconds
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

        ## Overview
        This method creates a new configuration with updated top-level options,
        preserving the existing params dictionary and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate options
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update options
           - Preserve params

        Args:
            **kwargs: Options to update

        Returns:
            BackoffRetryConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(max_attempts=7)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "BackoffRetryConfig":
        """
        Create a new config with updated parameters.

        ## Overview
        This method creates a new configuration with updated params,
        preserving existing top-level options and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate parameters
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update params
           - Preserve options

        Args:
            **kwargs: Parameters to update

        Returns:
            BackoffRetryConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(jitter=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})


class ValidationConfig(BaseModel):
    """
    Immutable configuration for validation managers.

    ## Overview
    This class provides a consistent way to configure validation managers across
    the Sifaka framework. It handles common configuration options like prioritize_by_cost
    and allows manager-specific options through the params dictionary.

    ## Architecture
    ValidationConfig follows a configuration pattern:
    1. **Base Options**: Common configuration options
       - Rule prioritization
       - Basic settings
    2. **Custom Parameters**: Manager-specific options
       - Validation behavior
       - Other parameters

    ## Lifecycle
    1. **Creation**: Create configuration
       - Set base options
       - Set custom parameters
    2. **Modification**: Update configuration
       - Update options
       - Update parameters
    3. **Usage**: Apply configuration
       - Use with manager
       - Access values

    ## Error Handling
    - ValueError: Raised when configuration values are invalid
    - ValidationError: Raised when configuration validation fails

    ## Examples
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
        prioritize_by_cost (bool): Whether to prioritize rules by cost
        params (Dict[str, Any]): Dictionary of manager-specific configuration parameters
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

        ## Overview
        This method creates a new configuration with updated top-level options,
        preserving the existing params dictionary and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate options
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update options
           - Preserve params

        Args:
            **kwargs: Options to update

        Returns:
            ValidationConfig: New configuration with updated options

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated options
            updated_config = config.with_options(prioritize_by_cost=False)
            ```
        """
        return self.model_copy(update=kwargs)

    def with_params(self, **kwargs: Any) -> "ValidationConfig":
        """
        Create a new config with updated parameters.

        ## Overview
        This method creates a new configuration with updated params,
        preserving existing top-level options and maintaining immutability.

        ## Lifecycle
        1. **Input Processing**: Process inputs
           - Validate parameters
           - Check constraints
        2. **Configuration Creation**: Create new config
           - Copy existing config
           - Update params
           - Preserve options

        Args:
            **kwargs: Parameters to update

        Returns:
            ValidationConfig: New configuration with updated parameters

        Raises:
            ValueError: When configuration values are invalid
            ValidationError: When configuration validation fails

        Examples:
            ```python
            # Create a new configuration with updated params
            parameterized_config = config.with_params(fail_fast=False)
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})
