"""
Classifier Configuration Module

This module provides configuration classes and standardization functions for classifiers.

## Overview
The classifier configuration module defines configuration classes for classifiers in the Sifaka framework.
It provides a consistent approach to configuring classifiers with standardized parameter handling,
validation, and serialization.

## Components
- **ClassifierConfig**: Configuration for classifiers
- **ImplementationConfig**: Configuration for classifier implementations
- **standardize_classifier_config**: Standardization function for classifier configurations
- **extract_classifier_config_params**: Utility function to extract classifier parameters

## Usage Examples
```python
from sifaka.utils.config.classifiers import (
    ClassifierConfig, ImplementationConfig, standardize_classifier_config
)

# Create a classifier configuration
config = ClassifierConfig(
    name="my_classifier",
    description="A custom classifier",
    threshold=0.7,
    cache_size=100
)

# Create an implementation configuration
impl_config = ImplementationConfig(
    name="my_implementation",
    description="A custom implementation",
    model="gpt-4",
    temperature=0.7
)

# Use standardization function
config = standardize_classifier_config(
    threshold=0.7,
    cache_size=100,
    params={
        "model": "gpt-4",
        "temperature": 0.7
    }
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, cast
from pydantic import Field
from .base import BaseConfig

T = TypeVar("T", bound="ClassifierConfig")
R = TypeVar("R")


class ClassifierConfig(BaseConfig, Generic[R]):
    """
    Configuration for classifiers.

    This class provides a consistent way to configure classifiers across the Sifaka framework.
    It handles common configuration options like threshold and cache_size, while
    allowing classifier-specific options through the params dictionary.

    ## Architecture
    ClassifierConfig extends BaseConfig with classifier-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during classifier initialization and
    remain immutable throughout the classifier's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.classifiers import ClassifierConfig

    # Create a classifier configuration
    config = ClassifierConfig(
        name="my_classifier",
        description="A custom classifier",
        threshold=0.7,
        cache_size=100,
        trace_enabled=True,
        params={
            "model": "gpt-4",
            "temperature": 0.7
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Threshold: {config.threshold}")
    print(f"Model: {config.params.get('model') if params else "")")

    # Create a new configuration with updated options
    updated_config = config.with_options(threshold=0.8) if config else ""

    # Create a new configuration with updated params
    updated_config = config.with_params(model="gpt-3.5-turbo") if config else ""
    ```

    Attributes:
        threshold: Classification threshold
        cache_size: Size of the result cache
        trace_enabled: Whether to enable tracing
    """

    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Classification threshold")
    cache_size: int = Field(default=100, ge=0, description="Size of the result cache")
    trace_enabled: bool = Field(default=False, description="Whether to enable tracing")


class ImplementationConfig(BaseConfig):
    """
    Configuration for classifier implementations.

    This class provides a consistent way to configure classifier implementations across the Sifaka framework.
    It handles common configuration options like model and temperature, while
    allowing implementation-specific options through the params dictionary.

    ## Architecture
    ImplementationConfig extends BaseConfig with implementation-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during implementation initialization and
    remain immutable throughout the implementation's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.classifiers import ImplementationConfig

    # Create an implementation configuration
    config = ImplementationConfig(
        name="my_implementation",
        description="A custom implementation",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        params={
            "top_p": 0.9
        }
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")

    # Create a new configuration with updated options
    updated_config = config.with_options(temperature=0.8) if config else ""

    # Create a new configuration with updated params
    updated_config = config.with_params(top_p=0.95) if config else ""
    ```

    Attributes:
        model: Model name to use
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
    """

    model: str = Field(default="", description="Model name to use")
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Temperature for text generation"
    )
    max_tokens: int = Field(default=1000, ge=1, description="Maximum number of tokens to generate")


def extract_classifier_config_params(config: Dict[str, Any]) -> Any:
    """
    Extract classifier configuration parameters from a dictionary.

    This utility function extracts classifier-specific parameters from a dictionary,
    separating them from implementation-specific parameters.

    Args:
        config: Dictionary containing configuration parameters

    Returns:
        Dictionary containing classifier-specific parameters

    Examples:
        ```python
        from sifaka.utils.config.classifiers import extract_classifier_config_params

        # Extract classifier parameters
        config = {
            "threshold": 0.7,
            "cache_size": 100,
            "model": "gpt-4",
            "temperature": 0.7
        }
        classifier_params = extract_classifier_config_params(config)
        # classifier_params = {"threshold": 0.7, "cache_size": 100}
        ```
    """
    classifier_fields = set(ClassifierConfig.__annotations__.keys())
    base_fields = set(BaseConfig.__annotations__.keys())
    all_fields = classifier_fields.union(base_fields)
    return {k: v for k, v in config.items() if k in all_fields}


def standardize_classifier_config(
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = ClassifierConfig,
    **kwargs: Any,
) -> Any:
    """
    Standardize classifier configuration.

    This utility function ensures that classifier configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ClassifierConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or ClassifierConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: ClassifierConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ClassifierConfig object or subclass

    Examples:
        ```python
        from sifaka.utils.config.classifiers import standardize_classifier_config

        # Create from parameters
        config = standardize_classifier_config(
            threshold=0.7,
            cache_size=100,
            params={
                "model": "gpt-4",
                "temperature": 0.7
            }
        )

        # Create from existing config
        from sifaka.utils.config.classifiers import ClassifierConfig
        existing = ClassifierConfig(threshold=0.7)
        updated = standardize_classifier_config(
            config=existing,
            params={
                "model": "gpt-4",
                "temperature": 0.7
            }
        )

        # Create from dictionary
        dict_config = {
            "threshold": 0.7,
            "cache_size": 100,
            "params": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }
        config = standardize_classifier_config(config=dict_config)
        ```
    """
    final_params: Dict[str, Any] = {}
    if params:
        final_params.update(params)
    if isinstance(config, dict):
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)
        return cast(
            T, config_class(**(config if config is not None else {}), params=final_params, **kwargs)
        )
    elif isinstance(config, ClassifierConfig):
        final_params.update(config.params)
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))
    else:
        return cast(T, config_class(params=final_params, **kwargs))
