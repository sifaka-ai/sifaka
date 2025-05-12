"""
Chain Configuration Module

This module provides configuration classes and standardization functions for chains.

## Overview
The chain configuration module defines configuration classes for chains in the Sifaka framework.
It provides a consistent approach to configuring chains with standardized parameter handling,
validation, and serialization.

## Components
- **ChainConfig**: Configuration for chains
- **EngineConfig**: Configuration for execution engines
- **ValidatorConfig**: Configuration for validators
- **ImproverConfig**: Configuration for improvers
- **FormatterConfig**: Configuration for formatters
- **standardize_chain_config**: Standardization function for chain configurations

## Usage Examples
```python
from sifaka.utils.config.chain import (
    ChainConfig, EngineConfig, standardize_chain_config
)

# Create a chain configuration
config = ChainConfig(
    name="my_chain",
    description="A custom chain",
    max_attempts=3,
    timeout_seconds=30
)

# Create an engine configuration
engine_config = EngineConfig(
    max_attempts=3,
    timeout_seconds=30,
    fail_fast=True
)

# Use standardization function
config = standardize_chain_config(
    max_attempts=3,
    timeout_seconds=30,
    params={
        "fail_fast": True
    }
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import Field

from .base import BaseConfig

# Type variables for generic configuration handling
T = TypeVar("T", bound="ChainConfig")


class ChainConfig(BaseConfig):
    """
    Configuration for chains.

    This class provides a consistent way to configure chains across the Sifaka framework.
    It handles common configuration options like max_attempts and timeout, while
    allowing chain-specific options through the params dictionary.

    ## Architecture
    ChainConfig extends BaseConfig with chain-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during chain initialization and
    remain immutable throughout the chain's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.chain import ChainConfig

    # Create a chain configuration
    config = ChainConfig(
        name="my_chain",
        description="A custom chain",
        max_attempts=3,
        timeout=30,
        trace_enabled=True,
        params={
            "fail_fast": True
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Max attempts: {config.max_attempts}")
    print(f"Timeout: {config.timeout}")
    print(f"Fail fast: {config.params.get('fail_fast')}")

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(fail_fast=False)
    ```

    Attributes:
        max_attempts: Maximum number of attempts
        timeout: Timeout in seconds (alias for timeout_seconds)
        timeout_seconds: Timeout in seconds
        trace_enabled: Whether to enable tracing
        retry_delay: Delay between retry attempts in seconds
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of attempts",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout in seconds",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retry attempts in seconds",
    )

    @property
    def timeout(self) -> float:
        """
        Get the timeout in seconds.

        This is an alias for timeout_seconds to maintain backward compatibility.

        Returns:
            float: Timeout in seconds
        """
        return self.timeout_seconds


class EngineConfig(BaseConfig):
    """
    Configuration for execution engines.

    This class provides a consistent way to configure execution engines across the Sifaka framework.
    It handles common configuration options like max_attempts and timeout_seconds, while
    allowing engine-specific options through the params dictionary.

    ## Architecture
    EngineConfig extends BaseConfig with engine-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during engine initialization and
    remain immutable throughout the engine's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.chain import EngineConfig

    # Create an engine configuration
    config = EngineConfig(
        name="my_engine",
        description="A custom engine",
        max_attempts=3,
        timeout_seconds=30,
        fail_fast=True,
        params={
            "retry_delay_seconds": 1.0
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Max attempts: {config.max_attempts}")
    print(f"Fail fast: {config.fail_fast}")

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(retry_delay_seconds=2.0)
    ```

    Attributes:
        max_attempts: Maximum number of attempts
        timeout_seconds: Timeout in seconds
        fail_fast: Whether to fail fast on validation errors
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of attempts",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout in seconds",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to fail fast on validation errors",
    )


class ValidatorConfig(BaseConfig):
    """
    Configuration for validators.

    This class provides a consistent way to configure validators across the Sifaka framework.
    It handles common configuration options like prioritize_by_cost and parallel_validation, while
    allowing validator-specific options through the params dictionary.

    ## Architecture
    ValidatorConfig extends BaseConfig with validator-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during validator initialization and
    remain immutable throughout the validator's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.chain import ValidatorConfig

    # Create a validator configuration
    config = ValidatorConfig(
        name="my_validator",
        description="A custom validator",
        prioritize_by_cost=True,
        parallel_validation=False,
        params={
            "fail_fast": True
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Prioritize by cost: {config.prioritize_by_cost}")
    print(f"Parallel validation: {config.parallel_validation}")

    # Create a new configuration with updated options
    updated_config = config.with_options(parallel_validation=True)

    # Create a new configuration with updated params
    updated_config = config.with_params(fail_fast=False)
    ```

    Attributes:
        prioritize_by_cost: Whether to prioritize rules by cost
        parallel_validation: Whether to run validation in parallel
    """

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize rules by cost",
    )
    parallel_validation: bool = Field(
        default=False,
        description="Whether to run validation in parallel",
    )


class ImproverConfig(BaseConfig):
    """
    Configuration for improvers.

    This class provides a consistent way to configure improvers across the Sifaka framework.
    It handles common configuration options like max_attempts and timeout_seconds, while
    allowing improver-specific options through the params dictionary.

    ## Architecture
    ImproverConfig extends BaseConfig with improver-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during improver initialization and
    remain immutable throughout the improver's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.chain import ImproverConfig

    # Create an improver configuration
    config = ImproverConfig(
        name="my_improver",
        description="A custom improver",
        max_attempts=3,
        timeout_seconds=30,
        params={
            "improvement_strategy": "iterative"
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Max attempts: {config.max_attempts}")
    print(f"Improvement strategy: {config.params.get('improvement_strategy')}")

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(improvement_strategy="single_pass")
    ```

    Attributes:
        max_attempts: Maximum number of improvement attempts
        timeout_seconds: Timeout in seconds
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of improvement attempts",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout in seconds",
    )


class FormatterConfig(BaseConfig):
    """
    Configuration for formatters.

    This class provides a consistent way to configure formatters across the Sifaka framework.
    It handles common configuration options like include_metadata and pretty_print, while
    allowing formatter-specific options through the params dictionary.

    ## Architecture
    FormatterConfig extends BaseConfig with formatter-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during formatter initialization and
    remain immutable throughout the formatter's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.chain import FormatterConfig

    # Create a formatter configuration
    config = FormatterConfig(
        name="my_formatter",
        description="A custom formatter",
        include_metadata=True,
        pretty_print=True,
        params={
            "format": "json"
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Include metadata: {config.include_metadata}")
    print(f"Pretty print: {config.pretty_print}")

    # Create a new configuration with updated options
    updated_config = config.with_options(pretty_print=False)

    # Create a new configuration with updated params
    updated_config = config.with_params(format="yaml")
    ```

    Attributes:
        include_metadata: Whether to include metadata in the formatted output
        pretty_print: Whether to pretty-print the formatted output
    """

    include_metadata: bool = Field(
        default=False,
        description="Whether to include metadata in the formatted output",
    )
    pretty_print: bool = Field(
        default=False,
        description="Whether to pretty-print the formatted output",
    )


def standardize_chain_config(
    config: Optional[Union[Dict[str, Any], ChainConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = ChainConfig,
    **kwargs: Any,
) -> T:
    """
    Standardize chain configuration.

    This utility function ensures that chain configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ChainConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or ChainConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: ChainConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ChainConfig object or subclass

    Examples:
        ```python
        from sifaka.utils.config.chain import standardize_chain_config, EngineConfig

        # Create from parameters
        config = standardize_chain_config(
            max_attempts=3,
            timeout=30,  # Can use either timeout or timeout_seconds
            params={
                "fail_fast": True
            }
        )

        # Create from existing config
        from sifaka.utils.config.chain import ChainConfig
        existing = ChainConfig(max_attempts=3)
        updated = standardize_chain_config(
            config=existing,
            params={
                "fail_fast": True
            }
        )

        # Create from dictionary
        dict_config = {
            "max_attempts": 3,
            "timeout": 30,  # Can use either timeout or timeout_seconds
            "params": {
                "fail_fast": True
            }
        }
        config = standardize_chain_config(config=dict_config)

        # Create specialized config
        engine_config = standardize_chain_config(
            config_class=EngineConfig,
            max_attempts=3,
            timeout=30,  # Can use either timeout or timeout_seconds
            fail_fast=True
        )
        ```
    """
    # Start with empty params dictionary
    final_params: Dict[str, Any] = {}

    # If params is provided, use it as the base
    if params:
        final_params.update(params)

    # Handle timeout/timeout_seconds compatibility
    if "timeout" in kwargs and "timeout_seconds" not in kwargs:
        kwargs["timeout_seconds"] = kwargs.pop("timeout")

    # If config is a dictionary
    if isinstance(config, dict):
        # Extract params from the dictionary
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)

        # Handle timeout/timeout_seconds compatibility in config dict
        if config and "timeout" in config and "timeout_seconds" not in config:
            config["timeout_seconds"] = config.pop("timeout")

        # Create config with the remaining options and the merged params
        return cast(
            T, config_class(**({} if config is None else config), params=final_params, **kwargs)
        )

    # If config is a ChainConfig
    elif isinstance(config, ChainConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return cast(T, config_class(params=final_params, **kwargs))
