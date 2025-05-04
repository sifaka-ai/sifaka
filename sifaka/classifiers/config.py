"""
Configuration utilities for classifiers.

This module provides utility functions for handling classifier configuration objects
consistently across the Sifaka framework.
"""

from typing import Any, Dict, List, Optional, Union

from sifaka.classifiers.base import ClassifierConfig


def standardize_classifier_config(
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    labels: Optional[list] = None,
    **kwargs: Any,
) -> ClassifierConfig:
    """
    Standardize classifier configuration.

    This utility function ensures that classifier configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ClassifierConfig object.

    Args:
        config: Optional configuration (either a dict or ClassifierConfig)
        params: Optional params dictionary to merge with config
        labels: Optional labels list (required if not in config)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ClassifierConfig object

    Examples:
        ```python
        # Create from parameters
        config = standardize_classifier_config(
            labels=["positive", "negative"],
            params={"threshold": 0.7}
        )

        # Create from existing config
        existing = ClassifierConfig(labels=["yes", "no"])
        updated = standardize_classifier_config(
            config=existing,
            params={"threshold": 0.8}
        )

        # Create from dictionary
        dict_config = {
            "labels": ["spam", "ham"],
            "params": {"threshold": 0.5}
        }
        config = standardize_classifier_config(config=dict_config)
        ```
    """
    # Start with empty params dictionary
    final_params: Dict[str, Any] = {}
    final_labels = labels or []

    # If params is provided, use it as the base
    if params:
        final_params.update(params)

    # If config is a dictionary
    if isinstance(config, dict):
        # Extract params and labels from the dictionary
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)

        # Extract labels if present
        if "labels" in config:
            final_labels = config.pop("labels")

        # Create ClassifierConfig with the remaining options and the merged params
        config_kwargs = {} if config is None else config
        return ClassifierConfig(labels=final_labels, params=final_params, **kwargs)

    # If config is a ClassifierConfig
    elif isinstance(config, ClassifierConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Use the existing labels if none provided
        if not final_labels:
            final_labels = config.labels

        # Create a new ClassifierConfig with the updated params
        return config.with_options(params=final_params, labels=final_labels, **kwargs)

    # If no config is provided
    else:
        # Create a new ClassifierConfig with the params and kwargs
        if not final_labels:
            raise ValueError("Labels must be provided when creating a new ClassifierConfig")

        return ClassifierConfig(labels=final_labels, params=final_params, **kwargs)
