"""
Chain Configuration Module

A module providing configuration classes for the Sifaka chain system.

## Overview
This module provides configuration classes for the Sifaka chain system,
importing standardized configuration classes from utils/config.py and
extending them with chain-specific functionality. These configuration
classes ensure consistent configuration across the chain system and
provide type safety and validation through Pydantic.

## Components
- ChainConfig: Configuration for chains
- EngineConfig: Configuration for the execution engine
- ValidatorConfig: Configuration for validators
- ImproverConfig: Configuration for improvers
- FormatterConfig: Configuration for formatters

## Usage Examples
```python
from sifaka.chain.config import ChainConfig, EngineConfig
from sifaka.utils.config import standardize_chain_config

# Create chain configuration
config = ChainConfig(
    max_attempts=3,
    cache_enabled=True,
    trace_enabled=True,
    async_enabled=False,
    timeout=60.0,
    params={
        "system_prompt": "You are a helpful assistant.",
        "use_improver": True,
    }
)

# Create engine configuration
engine_config = EngineConfig(
    max_attempts=3,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_retry_delay=30.0,
    jitter=True
)

# Create chain with configuration
chain = Chain(
    model=model,
    validators=validators,
    improver=improver,
    config=config
)

# Update configuration
updated_config = config.with_options(max_attempts=5)
chain.update_config(updated_config)

# Standardize configuration
std_config = standardize_chain_config(
    max_attempts=3,
    params={"system_prompt": "You are a helpful assistant."}
)
```

## Configuration Principles
Sifaka follows these principles for configuration:
- Consistency: All components use a similar configuration pattern
- Immutability: Configuration objects are immutable to prevent accidental changes
- Extensibility: Base configuration classes can be extended for specialized needs
- Standardization: All component-specific parameters are stored in a `params` dictionary
- Type Safety: All configurations use Pydantic models for type safety and validation
"""

from pydantic import Field

from sifaka.utils.config import BaseConfig, ValidationConfig


class ChainConfig(BaseConfig):
    """
    Configuration for chains.

    This class provides a standardized way to configure chains in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig from utils/config.py to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ChainConfig(
        max_attempts=3,
        cache_enabled=True,
        trace_enabled=False
    )

    # Create a configuration with params
    config = ChainConfig(
        max_attempts=3,
        params={
            "system_prompt": "You are a helpful assistant.",
            "use_improver": True
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)

    # Create a new configuration with updated params
    updated_params_config = config.with_params(
        system_prompt="You are an expert editor."
    )
    ```

    Attributes:
        max_attempts (int): Maximum number of generation attempts
        cache_enabled (bool): Whether to enable result caching
        trace_enabled (bool): Whether to enable execution tracing
        async_enabled (bool): Whether to enable async execution
        timeout (float): Timeout for chain operations in seconds
    """

    max_attempts: int = Field(default=3, ge=1, description="Maximum number of generation attempts")
    cache_enabled: bool = Field(default=True, description="Whether to enable result caching")
    trace_enabled: bool = Field(default=False, description="Whether to enable execution tracing")
    async_enabled: bool = Field(default=False, description="Whether to enable async execution")
    timeout: float = Field(
        default=60.0, ge=0.0, description="Timeout for chain operations in seconds"
    )


class EngineConfig(BaseConfig):
    """
    Configuration for the execution engine.

    This class provides a standardized way to configure the execution engine
    in the Sifaka framework, with immutable configuration values to ensure
    consistency. It includes settings for retry behavior and backoff strategies.

    ## Architecture
    Extends BaseConfig from utils/config.py to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = EngineConfig(
        max_attempts=3,
        retry_delay=1.0,
        backoff_factor=2.0
    )

    # Create a configuration with jitter
    config = EngineConfig(
        max_attempts=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_retry_delay=30.0,
        jitter=True
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)
    ```

    Attributes:
        max_attempts (int): Maximum number of generation attempts
        retry_delay (float): Delay between retry attempts in seconds
        backoff_factor (float): Factor to multiply retry delay by each attempt
        max_retry_delay (float): Maximum retry delay in seconds
        jitter (bool): Whether to add random jitter to retry delays
    """

    max_attempts: int = Field(default=3, ge=1, description="Maximum number of generation attempts")
    retry_delay: float = Field(
        default=0.0, ge=0.0, description="Delay between retry attempts in seconds"
    )
    backoff_factor: float = Field(
        default=1.0, ge=1.0, description="Factor to multiply retry delay by each attempt"
    )
    max_retry_delay: float = Field(
        default=60.0, ge=0.0, description="Maximum retry delay in seconds"
    )
    jitter: bool = Field(default=False, description="Whether to add random jitter to retry delays")


class ValidatorConfig(ValidationConfig):
    """
    Configuration for validators.

    This class provides a standardized way to configure validators in the Sifaka
    framework, with immutable configuration values to ensure consistency.
    It extends ValidationConfig with validator-specific settings.

    ## Architecture
    Extends ValidationConfig from utils/config.py to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ValidatorConfig(
        timeout=10.0
    )

    # Create a configuration with params
    config = ValidatorConfig(
        timeout=10.0,
        params={
            "min_length": 10,
            "max_length": 1000
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(timeout=20.0)
    ```

    Attributes:
        timeout (float): Timeout for validation operations in seconds
    """

    timeout: float = Field(
        default=10.0, ge=0.0, description="Timeout for validation operations in seconds"
    )


class ImproverConfig(BaseConfig):
    """
    Configuration for improvers.

    This class provides a standardized way to configure improvers in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig from utils/config.py to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ImproverConfig(
        timeout=30.0,
        max_improvement_attempts=3
    )

    # Create a configuration with params
    config = ImproverConfig(
        timeout=30.0,
        max_improvement_attempts=3,
        params={
            "system_prompt": "You are an expert editor.",
            "improvement_strategy": "iterative"
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_improvement_attempts=5)
    ```

    Attributes:
        timeout (float): Timeout for improvement operations in seconds
        max_improvement_attempts (int): Maximum number of improvement attempts
    """

    timeout: float = Field(
        default=30.0, ge=0.0, description="Timeout for improvement operations in seconds"
    )
    max_improvement_attempts: int = Field(
        default=3, ge=1, description="Maximum number of improvement attempts"
    )


class FormatterConfig(BaseConfig):
    """
    Configuration for formatters.

    This class provides a standardized way to configure formatters in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig from utils/config.py to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = FormatterConfig(
        include_metadata=True,
        include_validation_results=True
    )

    # Create a configuration with params
    config = FormatterConfig(
        include_metadata=True,
        include_validation_results=True,
        params={
            "format": "json",
            "pretty_print": True
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(include_metadata=False)
    ```

    Attributes:
        include_metadata (bool): Whether to include metadata in results
        include_validation_results (bool): Whether to include validation results in results
    """

    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in results"
    )
    include_validation_results: bool = Field(
        default=True, description="Whether to include validation results in results"
    )
