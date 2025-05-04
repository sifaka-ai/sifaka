"""
Configuration utilities for Sifaka.

This module provides utility functions for handling configuration objects
consistently across the Sifaka framework.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from pydantic import BaseModel

from sifaka.rules.base import RuleConfig
from sifaka.classifiers.base import ClassifierConfig
from sifaka.critics.models import CriticConfig
from sifaka.models.config import ModelConfig
from sifaka.chain.config import ChainConfig, RetryConfig, ValidationConfig

T = TypeVar("T", bound=BaseModel)


def standardize_rule_config(
    config: Optional[Union[Dict[str, Any], RuleConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> RuleConfig:
    """
    Standardize rule configuration.

    This utility function ensures that rule configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized RuleConfig object.

    Args:
        config: Optional configuration (either a dict or RuleConfig)
        params: Optional params dictionary to merge with config
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized RuleConfig object

    Examples:
        ```python
        # Create from parameters
        config = standardize_rule_config(
            priority="HIGH",
            params={"min_length": 10, "max_length": 100}
        )

        # Create from existing config
        existing = RuleConfig(priority="MEDIUM")
        updated = standardize_rule_config(
            config=existing,
            params={"min_length": 20}
        )

        # Create from dictionary
        dict_config = {"priority": "LOW", "params": {"min_length": 5}}
        config = standardize_rule_config(config=dict_config)
        ```
    """
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

        # Create RuleConfig with the remaining options and the merged params
        return RuleConfig(**({} if config is None else config), params=final_params, **kwargs)

    # If config is a RuleConfig
    elif isinstance(config, RuleConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new RuleConfig with the updated params
        return config.with_options(params=final_params, **kwargs)

    # If no config is provided
    else:
        # Create a new RuleConfig with the params and kwargs
        return RuleConfig(params=final_params, **kwargs)


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
        return ClassifierConfig(**config_kwargs, labels=final_labels, params=final_params, **kwargs)

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


def standardize_critic_config(
    config: Optional[Union[Dict[str, Any], CriticConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = CriticConfig,
    **kwargs: Any,
) -> T:
    """
    Standardize critic configuration.

    This utility function ensures that critic configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized CriticConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or CriticConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: CriticConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized CriticConfig object or subclass

    Examples:
        ```python
        # Create from parameters
        config = standardize_critic_config(
            min_confidence=0.8,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from existing config
        existing = CriticConfig(min_confidence=0.7)
        updated = standardize_critic_config(
            config=existing,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from dictionary
        dict_config = {
            "min_confidence": 0.9,
            "params": {"system_prompt": "You are an expert editor."}
        }
        config = standardize_critic_config(config=dict_config)

        # Create specialized config
        from sifaka.critics.models import PromptCriticConfig
        prompt_config = standardize_critic_config(
            config_class=PromptCriticConfig,
            system_prompt="You are an expert editor.",
            temperature=0.7
        )
        ```
    """
    # Get the model fields
    model_fields = set(config_class.model_fields.keys())

    # Separate kwargs into model fields and extra params
    model_kwargs = {k: v for k, v in kwargs.items() if k in model_fields}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in model_fields}

    # If config is a dictionary
    if isinstance(config, dict):
        # Extract params from the dictionary
        dict_params = config.pop("params", {}) if config else {}
        extra_kwargs.update(dict_params)

        # Create config with the remaining options
        config_dict = (
            {} if config is None else {k: v for k, v in config.items() if k in model_fields}
        )
        config_dict.update(model_kwargs)
        return cast(T, config_class(**config_dict))

    # If config is a CriticConfig
    elif isinstance(config, CriticConfig):
        # Create a new config with the updated params
        config_dict = {k: v for k, v in config.model_dump().items() if k in model_fields}
        config_dict.update(model_kwargs)
        return cast(T, config_class(**config_dict))

    # If no config is provided
    else:
        # Create a new config with the model kwargs
        return cast(T, config_class(**model_kwargs))


def standardize_model_config(
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = ModelConfig,
    **kwargs: Any,
) -> T:
    """
    Standardize model configuration.

    This utility function ensures that model configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ModelConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or ModelConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: ModelConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ModelConfig object or subclass

    Examples:
        ```python
        # Create from parameters
        config = standardize_model_config(
            temperature=0.8,
            max_tokens=1000,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from existing config
        existing = ModelConfig(temperature=0.7)
        updated = standardize_model_config(
            config=existing,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from dictionary
        dict_config = {
            "temperature": 0.9,
            "params": {"system_prompt": "You are an expert editor."}
        }
        config = standardize_model_config(config=dict_config)

        # Create specialized config
        from sifaka.models.config import OpenAIConfig
        openai_config = standardize_model_config(
            config_class=OpenAIConfig,
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """
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

    # If config is a ModelConfig
    elif isinstance(config, ModelConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return cast(T, config_class(params=final_params, **kwargs))


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
        # Create from parameters
        config = standardize_chain_config(
            max_attempts=5,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from existing config
        existing = ChainConfig(max_attempts=3)
        updated = standardize_chain_config(
            config=existing,
            params={"system_prompt": "You are an expert editor."}
        )

        # Create from dictionary
        dict_config = {
            "max_attempts": 4,
            "params": {"system_prompt": "You are an expert editor."}
        }
        config = standardize_chain_config(config=dict_config)
        ```
    """
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


def standardize_retry_config(
    config: Optional[Union[Dict[str, Any], RetryConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = RetryConfig,
    **kwargs: Any,
) -> T:
    """
    Standardize retry configuration.

    This utility function ensures that retry configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized RetryConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or RetryConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: RetryConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized RetryConfig object or subclass

    Examples:
        ```python
        # Create from parameters
        config = standardize_retry_config(
            max_attempts=5,
            params={"use_backoff": True}
        )

        # Create from existing config
        existing = RetryConfig(max_attempts=3)
        updated = standardize_retry_config(
            config=existing,
            params={"use_backoff": True}
        )

        # Create from dictionary
        dict_config = {
            "max_attempts": 4,
            "params": {"use_backoff": True}
        }
        config = standardize_retry_config(config=dict_config)

        # Create specialized config
        from sifaka.chain.config import BackoffRetryConfig
        backoff_config = standardize_retry_config(
            config_class=BackoffRetryConfig,
            max_attempts=5,
            initial_backoff=1.0,
            backoff_factor=2.0
        )
        ```
    """
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

    # If config is a RetryConfig
    elif isinstance(config, RetryConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return cast(T, config_class(**config_dict))

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return cast(T, config_class(params=final_params, **kwargs))


def standardize_validation_config(
    config: Optional[Union[Dict[str, Any], ValidationConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ValidationConfig:
    """
    Standardize validation configuration.

    This utility function ensures that validation configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ValidationConfig object.

    Args:
        config: Optional configuration (either a dict or ValidationConfig)
        params: Optional params dictionary to merge with config
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ValidationConfig object

    Examples:
        ```python
        # Create from parameters
        config = standardize_validation_config(
            prioritize_by_cost=True,
            params={"fail_fast": True}
        )

        # Create from existing config
        existing = ValidationConfig(prioritize_by_cost=False)
        updated = standardize_validation_config(
            config=existing,
            params={"fail_fast": True}
        )

        # Create from dictionary
        dict_config = {
            "prioritize_by_cost": True,
            "params": {"fail_fast": True}
        }
        config = standardize_validation_config(config=dict_config)
        ```
    """
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
        return ValidationConfig(**({} if config is None else config), params=final_params, **kwargs)

    # If config is a ValidationConfig
    elif isinstance(config, ValidationConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return ValidationConfig(**config_dict)

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return ValidationConfig(params=final_params, **kwargs)
