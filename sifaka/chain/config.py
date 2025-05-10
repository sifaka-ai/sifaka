"""
Chain Configuration Module

This module provides configuration classes for the Sifaka chain system.
It defines a unified configuration system with sensible defaults and validation.

## Components
1. **ChainConfig**: Configuration for chains
2. **EngineConfig**: Configuration for the execution engine
3. **ModelConfig**: Configuration for models
4. **ValidatorConfig**: Configuration for validators
5. **ImproverConfig**: Configuration for improvers
6. **FormatterConfig**: Configuration for formatters

## Usage Examples
```python
from sifaka.chain.v2.config import ChainConfig

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
updated_config = config.model_copy(update={"max_attempts": 5})
chain.update_config(updated_config)
```
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration for all components."""
    
    name: str = Field(default="", description="Component name")
    description: str = Field(default="", description="Component description")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    model_config = ConfigDict(frozen=True)
    
    def with_params(self, **kwargs: Any) -> "BaseConfig":
        """
        Create a new configuration with updated parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            New configuration with updated parameters
        """
        return self.model_copy(
            update={"params": {**self.params, **kwargs}}
        )


class ChainConfig(BaseConfig):
    """Configuration for chains."""
    
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of generation attempts"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable result caching"
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Maximum number of cached results"
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable execution tracing"
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop on first validation failure"
    )
    async_enabled: bool = Field(
        default=False,
        description="Whether to enable asynchronous execution"
    )


class EngineConfig(BaseConfig):
    """Configuration for the execution engine."""
    
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of generation attempts"
    )
    retry_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Delay between retry attempts in seconds"
    )
    backoff_factor: float = Field(
        default=1.0,
        ge=1.0,
        description="Factor to multiply retry delay by each attempt"
    )
    max_retry_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum retry delay in seconds"
    )
    jitter: bool = Field(
        default=False,
        description="Whether to add random jitter to retry delays"
    )


class ModelConfig(BaseConfig):
    """Configuration for models."""
    
    timeout: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout for model operations in seconds"
    )
    max_tokens: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )


class ValidatorConfig(BaseConfig):
    """Configuration for validators."""
    
    timeout: float = Field(
        default=10.0,
        ge=0.0,
        description="Timeout for validation operations in seconds"
    )
    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize validators by cost"
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop on first validation failure"
    )


class ImproverConfig(BaseConfig):
    """Configuration for improvers."""
    
    timeout: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout for improvement operations in seconds"
    )
    max_improvement_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of improvement attempts"
    )


class FormatterConfig(BaseConfig):
    """Configuration for formatters."""
    
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results"
    )
    include_validation_results: bool = Field(
        default=True,
        description="Whether to include validation results in results"
    )
