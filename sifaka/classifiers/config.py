"""
Classifier Configuration Module

This module imports standardized configuration classes from utils/config.py and
extends them with classifier-specific functionality.

## Components
1. **ClassifierConfig**: Configuration for classifiers
2. **ImplementationConfig**: Configuration for classifier implementations

## Usage Examples
```python
from sifaka.utils.config import ClassifierConfig, standardize_classifier_config

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
```
"""

from typing import List, Dict, Any, TypeVar, Generic, Optional, Union, Type, cast
from pydantic import Field

from sifaka.utils.config import BaseConfig

# Type variable for label type
T = TypeVar("T")


class ClassifierConfig(BaseConfig, Generic[T]):
    """
    Configuration for classifiers.

    This class is generic over the label type T, which allows for type-safe
    configuration of classifiers with different label types.
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
    """Configuration for classifier implementations."""

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
