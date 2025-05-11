"""
Classifier Configuration Module

This module provides configuration classes and utilities for the Sifaka classifiers system.
It extends the standardized configuration classes from utils/config.py with classifier-specific
functionality, ensuring consistent configuration handling across the framework.

## Overview
Configuration is a critical aspect of the classifiers system, enabling customization
of classifier behavior, performance characteristics, and integration options. This module
provides Pydantic-based configuration classes with validation, default values, and
utility functions for standardizing configuration across different components.

## Components
1. **ClassifierConfig**: Configuration for classifiers with type-safe label handling
2. **ImplementationConfig**: Configuration for classifier implementations
3. **standardize_classifier_config**: Utility function for standardizing configuration
4. **extract_classifier_config_params**: Utility function for extracting configuration parameters

## Architecture
The configuration system follows a layered approach:
1. **Base Configuration**: Inherited from utils/config.py BaseConfig
2. **Classifier Configuration**: Extends base with classifier-specific options
3. **Implementation Configuration**: Specialized for classifier implementations
4. **Utility Functions**: Standardize configuration handling across components

## Usage Examples
```python
from sifaka.classifiers.config import ClassifierConfig, standardize_classifier_config

# Create classifier configuration
config = ClassifierConfig(
    name="sentiment_classifier",
    description="Classifies text sentiment",
    cache_size=100,
    min_confidence=0.7,
    labels=["positive", "negative", "neutral"],
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
updated_config = config.with_options(min_confidence=0.5)
classifier.update_config(updated_config)

# Standardize configuration
std_config = standardize_classifier_config(
    min_confidence=0.7,
    labels=["positive", "negative", "neutral"],
    params={"threshold": 0.8}
)

# Extract configuration parameters
config_params = extract_classifier_config_params(
    labels=["positive", "negative", "neutral"],
    cache_size=100,
    min_confidence=0.7,
    cost=1.5,
    provided_params={"threshold": 0.8},
    default_params={"use_fallback": True}
)
```

## Error Handling
The configuration classes use Pydantic for validation, which provides:
- Type checking for all configuration fields
- Range validation for numeric fields
- Default values for optional fields
- Clear error messages for invalid configuration

## Configuration Options
Common configuration options include:
- name: Name of the classifier
- description: Description of the classifier
- min_confidence: Minimum confidence threshold
- labels: Valid classification labels
- cache_size: Size of the classification cache
- params: Additional parameters dictionary
- async_enabled: Whether async classification is enabled
"""

from typing import List, Dict, Any, TypeVar, Generic, Optional, Union, Type, cast
from pydantic import Field

from sifaka.utils.config import BaseConfig

# Type variable for label type
T = TypeVar("T")


class ClassifierConfig(BaseConfig, Generic[T]):
    """
    Configuration for classifiers.

    This class defines the configuration options for classifiers in the Sifaka framework.
    It is generic over the label type T, which allows for type-safe configuration of
    classifiers with different label types.

    ## Architecture
    ClassifierConfig uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Range validation for numeric fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Generic type parameter for label type

    ## Lifecycle
    1. **Creation**: Instantiate with desired configuration options
    2. **Validation**: Pydantic validates all fields during creation
    3. **Usage**: Pass to Classifier constructor or update_config method
    4. **Modification**: Create new instances with with_options() for immutability

    ## Examples
    ```python
    # Create basic configuration
    config = ClassifierConfig(
        name="toxicity_classifier",
        description="Detects toxic content in text",
        min_confidence=0.7
    )

    # Create configuration with labels and params
    config = ClassifierConfig(
        name="sentiment_classifier",
        labels=["positive", "negative", "neutral"],
        params={"threshold": 0.8}
    )

    # Create configuration with async enabled
    config = ClassifierConfig(
        name="async_classifier",
        async_enabled=True
    )
    ```
    """

    name: str = Field(default="classifier", description="Name of the classifier")
    description: str = Field(default="Text classifier", description="Description of the classifier")
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for classification"
    )
    labels: List[str] = Field(default_factory=list, description="Valid classification labels")
    cache_size: int = Field(default=100, ge=0, description="Size of the classification cache")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    async_enabled: bool = Field(
        default=False, description="Whether async classification is enabled"
    )


class ImplementationConfig(BaseConfig):
    """
    Configuration for classifier implementations.

    This class defines the configuration options specific to classifier implementations,
    focusing on operational parameters like timeouts and fallback behavior.

    ## Architecture
    ImplementationConfig uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Range validation for numeric fields
    - Default values for optional fields
    - Field descriptions for documentation

    ## Lifecycle
    1. **Creation**: Instantiate with desired implementation options
    2. **Validation**: Pydantic validates all fields during creation
    3. **Usage**: Pass to implementation constructors or configuration methods
    4. **Modification**: Create new instances with with_options() for immutability

    ## Examples
    ```python
    # Create basic implementation configuration
    config = ImplementationConfig(
        timeout=5.0,
        fallback_label="unknown"
    )

    # Create implementation with custom fallback
    config = ImplementationConfig(
        fallback_label="neutral",
        fallback_confidence=0.3
    )
    ```
    """

    timeout: float = Field(
        default=10.0, ge=0.0, description="Timeout for classification operations in seconds"
    )
    fallback_label: str = Field(
        default="unknown", description="Fallback label to use when classification fails"
    )
    fallback_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score for fallback label"
    )


def standardize_classifier_config(
    config: Optional[Union[Dict[str, Any], "ClassifierConfig"]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = None,
    **kwargs: Any,
) -> T:
    """
    Standardize classifier configuration.

    This utility function ensures that classifier configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ClassifierConfig object or a subclass.

    ## Architecture
    The function handles three main input scenarios:
    1. Dictionary configuration: Converts to ClassifierConfig
    2. ClassifierConfig instance: Updates with new parameters
    3. No configuration: Creates new ClassifierConfig with defaults

    In all cases, it merges parameters from multiple sources with consistent precedence.

    ## Examples
    ```python
    # From dictionary
    config = standardize_classifier_config(
        config={"name": "my_classifier", "min_confidence": 0.7},
        params={"threshold": 0.8}
    )

    # From existing ClassifierConfig
    existing_config = ClassifierConfig(name="base_classifier")
    updated_config = standardize_classifier_config(
        config=existing_config,
        min_confidence=0.8
    )

    # From scratch with parameters
    config = standardize_classifier_config(
        min_confidence=0.7,
        labels=["positive", "negative", "neutral"],
        params={"threshold": 0.8}
    )

    # With custom config class
    config = standardize_classifier_config(
        config_class=MyCustomConfig,
        name="custom_classifier"
    )
    ```

    Args:
        config: Optional configuration (either a dict or ClassifierConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: ClassifierConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ClassifierConfig object or subclass
    """
    # Use ClassifierConfig as the default config class
    if config_class is None:
        config_class = ClassifierConfig

    # Start with empty params dictionary
    final_params: Dict[str, Any] = {}

    # If params is provided, use it as the base
    if params:
        final_params.update(params)

    # If config is a dictionary
    if isinstance(config, dict):
        # Extract params from the dictionary
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)

        # Create config with the remaining options and the merged params
        return cast(
            T, config_class(**({} if config is None else config), params=final_params, **kwargs)
        )

    # If config is a ClassifierConfig
    elif isinstance(config, ClassifierConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return cast(T, config_class(params=final_params, **kwargs))


def extract_classifier_config_params(
    labels: Optional[List[str]] = None,
    cache_size: int = 0,
    min_confidence: float = 0.0,
    cost: Optional[float] = None,
    provided_params: Optional[Dict[str, Any]] = None,
    default_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Extract and merge configuration parameters for classifier factory methods.

    This utility function standardizes parameter extraction for classifier factory methods,
    ensuring consistent handling of configuration options. It merges parameters from various
    sources with the following precedence (highest to lowest):
    1. Explicitly provided kwargs
    2. Values in provided_params dictionary
    3. Default values in default_params dictionary

    ## Architecture
    The function follows a parameter merging pattern:
    1. Start with provided_params or empty dict
    2. Fill in missing values from default_params
    3. Create config dict with standard parameters
    4. Add optional parameters if provided
    5. Merge in any remaining kwargs

    ## Examples
    ```python
    # Basic usage
    config_params = extract_classifier_config_params(
        labels=["positive", "negative", "neutral"],
        cache_size=100,
        min_confidence=0.7
    )

    # With provided and default params
    config_params = extract_classifier_config_params(
        labels=["toxic", "non_toxic"],
        provided_params={"threshold": 0.8},
        default_params={"use_fallback": True}
    )

    # With computational cost
    config_params = extract_classifier_config_params(
        cache_size=100,
        cost=1.5,
        provided_params={"model_name": "toxicity-v2"}
    )

    # With additional kwargs
    config_params = extract_classifier_config_params(
        labels=["spam", "ham"],
        async_enabled=True,
        custom_option="value"
    )
    ```

    Args:
        labels: Optional list of classification labels
        cache_size: Size of the classification result cache
        min_confidence: Minimum confidence threshold
        cost: Optional computational cost metric
        provided_params: Dictionary of parameters provided by the caller
        default_params: Dictionary of default parameters
        **kwargs: Additional parameters to extract

    Returns:
        Dict containing merged configuration parameters and a params dictionary
    """
    # Extract params from kwargs if not explicitly provided
    params = kwargs.pop("params", {}) if provided_params is None else provided_params.copy()

    # Start with default params if provided
    if default_params:
        # Only use defaults for keys not in params
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

    # Create config dictionary
    config_dict = {"cache_size": cache_size, "min_confidence": min_confidence, "params": params}

    # Add cost if provided
    if cost is not None:
        config_dict["cost"] = cost

    # Add labels if provided
    if labels is not None:
        config_dict["labels"] = labels

    # Add any remaining kwargs
    config_dict.update(kwargs)

    return config_dict
