"""
Chain Configuration Module

This module provides configuration classes for the Sifaka chain system.
It imports standardized configuration classes from utils/config.py and
extends them with chain-specific functionality.

## Components
1. **ChainConfig**: Configuration for chains
2. **EngineConfig**: Configuration for the execution engine
3. **ValidatorConfig**: Configuration for validators
4. **ImproverConfig**: Configuration for improvers
5. **FormatterConfig**: Configuration for formatters

## Usage Examples
```python
from sifaka.chain.config import ChainConfig
from sifaka.utils.config import standardize_chain_config

# Create chain configuration
config = ChainConfig(
    max_attempts=3,
    cache_enabled=True,
    trace_enabled=True,
    params={
        "system_prompt": "You are a helpful assistant.",
        "use_improver": True,
    }
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
"""

from pydantic import Field

from sifaka.utils.config import BaseConfig, ValidationConfig


class EngineConfig(BaseConfig):
    """Configuration for the execution engine."""

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
    """Configuration for validators."""

    timeout: float = Field(
        default=10.0, ge=0.0, description="Timeout for validation operations in seconds"
    )


class ImproverConfig(BaseConfig):
    """Configuration for improvers."""

    timeout: float = Field(
        default=30.0, ge=0.0, description="Timeout for improvement operations in seconds"
    )
    max_improvement_attempts: int = Field(
        default=3, ge=1, description="Maximum number of improvement attempts"
    )


class FormatterConfig(BaseConfig):
    """Configuration for formatters."""

    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in results"
    )
    include_validation_results: bool = Field(
        default=True, description="Whether to include validation results in results"
    )
