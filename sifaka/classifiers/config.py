"""
Classifier Configuration Module

This module provides configuration classes for the Sifaka classifiers system.
It defines a unified configuration system with sensible defaults and validation.

## Components
1. **ClassifierConfig**: Configuration for classifiers
2. **ImplementationConfig**: Configuration for classifier implementations

## Usage Examples
```python
from sifaka.classifiers.v2.config import ClassifierConfig

# Create classifier configuration
config = ClassifierConfig(
    cache_enabled=True,
    cache_size=100,
    min_confidence=0.7,
    params={
        "threshold": 0.8,
        "use_fallback": True,
    }
)

# Create classifier with configuration
classifier = Classifier(
    implementation=implementation,
    config=config
)

# Update configuration
updated_config = config.model_copy(update={"min_confidence": 0.5})
classifier.update_config(updated_config)
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


class ClassifierConfig(BaseConfig):
    """Configuration for classifiers."""
    
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable result caching"
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Maximum number of cached results"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    async_enabled: bool = Field(
        default=False,
        description="Whether to enable asynchronous execution"
    )
    labels: List[str] = Field(
        default_factory=list,
        description="List of valid labels"
    )


class ImplementationConfig(BaseConfig):
    """Configuration for classifier implementations."""
    
    timeout: float = Field(
        default=10.0,
        ge=0.0,
        description="Timeout for classification operations in seconds"
    )
    fallback_label: str = Field(
        default="unknown",
        description="Fallback label to use when classification fails"
    )
    fallback_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for fallback label"
    )
