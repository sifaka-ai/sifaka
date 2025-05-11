"""
Configuration Utilities

This module provides a unified configuration system for the Sifaka framework,
including base configuration classes and standardization functions for different
component types.

## Overview
The configuration utilities provide a consistent way to configure all components
in the Sifaka framework. They ensure that configuration is handled consistently
across different component types, with standardized parameter handling, validation,
and serialization.

## Components
The module defines a hierarchy of configuration classes:

1. **BaseConfig**: Base configuration class for all components
2. **ModelConfig**: Configuration for model providers
3. **RuleConfig**: Configuration for rules
4. **CriticConfig**: Configuration for critics
5. **ChainConfig**: Configuration for chains
6. **ClassifierConfig**: Configuration for classifiers
7. **RetrieverConfig**: Configuration for retrievers
8. **RetryConfig**: Configuration for retry strategies
9. **ValidationConfig**: Configuration for validation

The module also provides standardization functions for each component type:

1. **standardize_rule_config**: Standardize rule configuration
2. **standardize_critic_config**: Standardize critic configuration
3. **standardize_model_config**: Standardize model provider configuration
4. **standardize_chain_config**: Standardize chain configuration
5. **standardize_classifier_config**: Standardize classifier configuration
6. **standardize_retriever_config**: Standardize retriever configuration
7. **standardize_retry_config**: Standardize retry strategy configuration
8. **standardize_validation_config**: Standardize validation configuration

## Usage Pattern
All standardization functions follow a consistent pattern:

1. Accept configuration in multiple formats (dict, config object, or parameters)
2. Merge parameters from different sources with consistent precedence
3. Return a standardized configuration object

This pattern ensures that configuration is handled consistently across the framework,
regardless of how it's provided by the caller.

## Usage Examples
```python
from sifaka.utils.config import (
    BaseConfig, ModelConfig, RuleConfig, CriticConfig,
    standardize_rule_config, standardize_critic_config, standardize_model_config
)

# Create base configuration
base_config = BaseConfig(
    name="my_component",
    description="A sample component",
    params={"key": "value"}
)

# Create model configuration
model_config = ModelConfig(
    temperature=0.7,
    max_tokens=1000,
    params={"system_prompt": "You are a helpful assistant."}
)

# Create rule configuration using standardization
rule_config = standardize_rule_config(
    priority="HIGH",
    params={"min_length": 10, "max_length": 100}
)

# Create critic configuration using standardization
critic_config = standardize_critic_config(
    min_confidence=0.8,
    params={"system_prompt": "You are an expert editor."}
)

# Create specialized critic configuration
prompt_config = standardize_critic_config(
    config_class=PromptCriticConfig,
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)

# Update existing configuration
existing_config = RuleConfig(priority="MEDIUM")
updated_config = standardize_rule_config(
    config=existing_config,
    params={"min_length": 20}
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

# Type variables for generic configuration handling
T = TypeVar("T", bound=BaseModel)
C = TypeVar("C")  # For ClassifierConfig generic type


class BaseConfig(BaseModel):
    """
    Base configuration for all Sifaka components.

    This class provides a consistent foundation for all configuration classes
    in the Sifaka framework. It defines common fields and methods that are
    shared across all component types.

    ## Architecture
    BaseConfig uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during component initialization and
    remain immutable throughout the component's lifecycle. Components can access
    configuration values through their config property.

    ## Examples
    ```python
    # Create a basic configuration
    config = BaseConfig(
        name="my_component",
        description="A custom component",
        params={"threshold": 0.7}
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Custom threshold: {config.params.get('threshold')}")

    # Create a new configuration with updated parameters
    updated_config = config.with_params(max_length=100, min_length=10)

    # Create a new configuration with updated options
    updated_config = config.with_options(name="new_name")
    ```

    Attributes:
        name: Component name
        description: Component description
        params: Dictionary of additional parameters
    """

    name: str = Field(default="", description="Component name")
    description: str = Field(default="", description="Component description")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    model_config = ConfigDict(frozen=True)

    def with_params(self, **kwargs: Any) -> "BaseConfig":
        """
        Create a new configuration with updated parameters.

        This method creates a new configuration object with the same options as the
        current configuration, but with updated parameters. The original configuration
        remains unchanged due to the immutable nature of configuration objects.

        Args:
            **kwargs: Parameters to update in the params dictionary

        Returns:
            New configuration with updated parameters

        Example:
            ```python
            # Create a configuration with parameters
            config = BaseConfig(
                name="my_component",
                params={"threshold": 0.7}
            )

            # Create a new configuration with updated parameters
            updated_config = config.with_params(
                threshold=0.8,
                max_length=100
            )

            # Original config is unchanged
            assert config.params["threshold"] == 0.7
            assert "max_length" not in config.params

            # New config has updated parameters
            assert updated_config.params["threshold"] == 0.8
            assert updated_config.params["max_length"] == 100
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})

    def with_options(self, **kwargs: Any) -> "BaseConfig":
        """
        Create a new configuration with updated options.

        This method creates a new configuration object with updated options.
        Unlike with_params, which updates the params dictionary, this method
        updates the configuration fields directly. The original configuration
        remains unchanged due to the immutable nature of configuration objects.

        Args:
            **kwargs: Configuration options to update

        Returns:
            New configuration with updated options

        Example:
            ```python
            # Create a configuration
            config = BaseConfig(
                name="my_component",
                description="Original description",
                params={"threshold": 0.7}
            )

            # Create a new configuration with updated options
            updated_config = config.with_options(
                name="new_name",
                description="Updated description"
            )

            # Original config is unchanged
            assert config.name == "my_component"
            assert config.description == "Original description"

            # New config has updated options
            assert updated_config.name == "new_name"
            assert updated_config.description == "Updated description"

            # Params are preserved
            assert updated_config.params == config.params
            ```
        """
        return self.model_copy(update=kwargs)


class ModelConfig(BaseConfig):
    """
    Configuration for model providers.

    This class provides a consistent way to configure model providers across the Sifaka framework.
    It handles common configuration options like temperature and max_tokens, while
    allowing model-specific options through the params dictionary.

    Attributes:
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        api_key: Optional API key for the model provider
        trace_enabled: Whether to enable tracing
    """

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )


class RulePriority(str, Enum):
    """
    Priority levels for rules.

    This enumeration defines the standard priority levels for rules in the Sifaka framework.
    Rules with higher priority are typically executed before rules with lower priority.

    ## Values
    - LOW: Lowest priority level
    - MEDIUM: Default priority level
    - HIGH: High priority level
    - CRITICAL: Highest priority level

    ## Usage
    ```python
    from sifaka.utils.config import RulePriority, RuleConfig

    # Create a rule configuration with HIGH priority
    config = RuleConfig(
        name="important_rule",
        priority=RulePriority.HIGH
    )

    # Priority can also be specified as a string
    config = RuleConfig(
        name="important_rule",
        priority="HIGH"
    )

    # Check priority level
    if config.priority == RulePriority.HIGH:
        print("This is a high-priority rule")
    ```
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleConfig(BaseConfig):
    """
    Configuration for rules.

    This class provides a consistent way to configure rules across the Sifaka framework.
    It handles common configuration options like priority and cost, while
    allowing rule-specific options through the params dictionary.

    Attributes:
        priority: Rule priority level
        cost: Computational cost of the rule
        cache_size: Size of the rule's result cache
    """

    priority: Union[RulePriority, str] = Field(
        default=RulePriority.MEDIUM,
        description="Priority level of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Computational cost of the rule",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of the rule's result cache",
    )


class CriticConfig(BaseConfig):
    """
    Configuration for critics.

    This class provides a consistent way to configure critics across the Sifaka framework.
    It handles common configuration options like min_confidence and max_attempts, while
    allowing critic-specific options through the params dictionary.

    Attributes:
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the critic's result cache
    """

    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of improvement attempts",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of the critic's result cache",
    )


class ChainConfig(BaseConfig):
    """
    Configuration for chains.

    This class provides a consistent way to configure chains across the Sifaka framework.
    It handles common configuration options like max_attempts and cache_enabled, while
    allowing chain-specific options through the params dictionary.

    Attributes:
        max_attempts: Maximum number of generation attempts
        cache_enabled: Whether to enable result caching
        trace_enabled: Whether to enable execution tracing
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of generation attempts",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable result caching",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable execution tracing",
    )


class ClassifierConfig(BaseConfig):
    """
    Configuration for classifiers.

    This class provides a consistent way to configure classifiers across the Sifaka framework.
    It handles common configuration options like min_confidence and cache_size, while
    allowing classifier-specific options through the params dictionary.

    Attributes:
        min_confidence: Minimum confidence threshold
        cache_size: Size of the classifier's result cache
        labels: List of classification labels
        cost: Computational cost of the classifier
    """

    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of the classifier's result cache",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="List of classification labels",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Computational cost of the classifier",
    )


class RetrieverConfig(BaseConfig):
    """
    Configuration for retrievers.

    This class provides a consistent way to configure retrievers across the Sifaka framework.
    It handles common configuration options like top_k and score_threshold, while
    allowing retriever-specific options through the params dictionary.

    Attributes:
        top_k: Number of top results to return
        score_threshold: Minimum score threshold for results
        cache_size: Size of the retriever's result cache
    """

    top_k: int = Field(
        default=3,
        ge=1,
        description="Number of top results to return",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold for results",
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Size of the retriever's result cache",
    )


class RetryConfig(BaseConfig):
    """
    Configuration for retry strategies.

    This class provides a consistent way to configure retry strategies across the Sifaka framework.
    It handles common configuration options like max_attempts and retry_delay, while
    allowing strategy-specific options through the params dictionary.

    Attributes:
        max_attempts: Maximum number of retry attempts
        retry_delay: Delay between retry attempts in seconds
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts",
    )
    retry_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Delay between retry attempts in seconds",
    )


class ValidationConfig(BaseConfig):
    """
    Configuration for validation.

    This class provides a consistent way to configure validation across the Sifaka framework.
    It handles common configuration options like prioritize_by_cost and parallel_validation, while
    allowing validation-specific options through the params dictionary.

    Attributes:
        prioritize_by_cost: Whether to prioritize validators by cost
        parallel_validation: Whether to run validators in parallel
    """

    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize validators by cost",
    )
    parallel_validation: bool = Field(
        default=False,
        description="Whether to run validators in parallel",
    )


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

    ## Workflow
    1. Merges parameters from all sources (config, params, kwargs)
    2. Handles different input formats (dict, RuleConfig, or None)
    3. Creates a new RuleConfig with standardized parameters
    4. Preserves existing configuration when updating

    ## Parameter Precedence
    Parameters are merged with the following precedence (highest to lowest):
    1. Explicit kwargs (e.g., priority="HIGH")
    2. Params dictionary
    3. Params from existing config
    4. Default values from RuleConfig

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

        # Parameter precedence example
        config = standardize_rule_config(
            config={"priority": "LOW", "params": {"threshold": 0.5}},
            params={"threshold": 0.7, "min_length": 10},
            priority="HIGH"
        )
        # Result: priority="HIGH", params={"threshold": 0.7, "min_length": 10}
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
        config_kwargs = {} if config is None else config
        return RuleConfig(**config_kwargs, params=final_params, **kwargs)

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
    # Start with empty params dictionary
    final_params: Dict[str, Any] = {}

    # If params is provided, use it as the base
    if params:
        final_params.update(params)

    # Get the model fields
    model_fields = set(config_class.model_fields.keys())

    # Separate kwargs into model fields and extra params
    model_kwargs = {k: v for k, v in kwargs.items() if k in model_fields}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in model_fields}
    final_params.update(extra_kwargs)

    # If config is a dictionary
    if isinstance(config, dict):
        # Extract params from the dictionary
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)

        # Create config with the remaining options
        config_dict = {} if config is None else config
        # Remove any fields that are in model_kwargs to avoid conflicts
        for k in model_kwargs:
            config_dict.pop(k, None)
        return config_class(**config_dict, params=final_params, **model_kwargs)

    # If config is a CriticConfig
    elif isinstance(config, CriticConfig):
        # Merge the existing params with the new params
        final_params.update(config.params)

        # Create a new config with the updated params
        config_dict = config.model_dump()
        config_dict.pop("params", None)  # Remove params to avoid conflicts
        # Remove any fields that are in model_kwargs to avoid conflicts
        for k in model_kwargs:
            config_dict.pop(k, None)
        return config_class(**config_dict, params=final_params, **model_kwargs)

    # If no config is provided
    else:
        # Create a new config with the params and kwargs
        return config_class(params=final_params, **model_kwargs)


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
        from sifaka.utils.config import standardize_chain_config
        from sifaka.chain.config import ChainConfig
        from sifaka.models.config import OpenAIConfig

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

        # Create with model configuration
        config = standardize_chain_config(
            max_attempts=5,
            params={
                "system_prompt": "You are an expert editor.",
                "model_config": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        )

        # Create with rule configurations
        config = standardize_chain_config(
            max_attempts=5,
            params={
                "system_prompt": "You are an expert editor.",
                "rules": [
                    {"name": "toxicity_rule", "params": {"threshold": 0.8}},
                    {"name": "length_rule", "params": {"min_length": 50, "max_length": 500}}
                ]
            }
        )

        # Create with critic configuration
        config = standardize_chain_config(
            max_attempts=5,
            params={
                "system_prompt": "You are an expert editor.",
                "critic_config": {
                    "name": "prompt_critic",
                    "min_confidence": 0.7,
                    "system_prompt": "You are an expert editor."
                }
            }
        )

        # Access configuration values
        print(f"Max attempts: {config.max_attempts}")
        print(f"System prompt: {config.params.get('system_prompt')}")
        print(f"Rules: {config.params.get('rules', [])}")
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
        from sifaka.utils.config import standardize_validation_config
        from sifaka.chain.config import ValidationConfig

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

        # Create with rule-specific parameters
        config = standardize_validation_config(
            prioritize_by_cost=True,
            params={
                "fail_fast": True,
                "rule_configs": {
                    "toxicity_rule": {"threshold": 0.8},
                    "length_rule": {"min_length": 50, "max_length": 500}
                }
            }
        )

        # Create with validation strategy parameters
        config = standardize_validation_config(
            prioritize_by_cost=True,
            parallel_validation=True,
            params={
                "fail_fast": True,
                "max_parallel_rules": 5
            }
        )

        # Access configuration values
        print(f"Prioritize by cost: {config.prioritize_by_cost}")
        print(f"Fail fast: {config.params.get('fail_fast', False)}")
        print(f"Rule configs: {config.params.get('rule_configs', {})}")
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


def standardize_classifier_config(
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = ClassifierConfig,
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

    Examples:
        ```python
        from sifaka.utils.config import standardize_classifier_config
        from sifaka.utils.config import ClassifierConfig

        # Create from parameters
        config = standardize_classifier_config(
            min_confidence=0.8,
            cache_size=200,
            labels=["positive", "negative", "neutral"],
            params={"threshold": 0.5}
        )

        # Create from existing config
        existing = ClassifierConfig(min_confidence=0.7)
        updated = standardize_classifier_config(
            config=existing,
            params={"threshold": 0.6}
        )

        # Create from dictionary
        dict_config = {
            "min_confidence": 0.9,
            "labels": ["toxic", "non-toxic"],
            "params": {"threshold": 0.7}
        }
        config = standardize_classifier_config(config=dict_config)

        # Create specialized config
        from sifaka.classifiers.config import ToxicityClassifierConfig
        toxicity_config = standardize_classifier_config(
            config_class=ToxicityClassifierConfig,
            min_confidence=0.8,
            labels=["toxic", "non-toxic"],
            params={"threshold": 0.7}
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


def standardize_retriever_config(
    config: Optional[Union[Dict[str, Any], RetrieverConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = RetrieverConfig,
    **kwargs: Any,
) -> T:
    """
    Standardize retriever configuration.

    This utility function ensures that retriever configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized RetrieverConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or RetrieverConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: RetrieverConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized RetrieverConfig object or subclass

    Examples:
        ```python
        from sifaka.utils.config import standardize_retriever_config
        from sifaka.utils.config import RetrieverConfig

        # Create from parameters
        config = standardize_retriever_config(
            top_k=5,
            score_threshold=0.7,
            params={"index_path": "/path/to/index"}
        )

        # Create from existing config
        existing = RetrieverConfig(top_k=3)
        updated = standardize_retriever_config(
            config=existing,
            params={"index_path": "/path/to/index"}
        )

        # Create from dictionary
        dict_config = {
            "top_k": 10,
            "score_threshold": 0.5,
            "params": {"index_path": "/path/to/index"}
        }
        config = standardize_retriever_config(config=dict_config)

        # Create specialized config
        from sifaka.retrieval.config import VectorRetrieverConfig
        vector_config = standardize_retriever_config(
            config_class=VectorRetrieverConfig,
            top_k=5,
            score_threshold=0.7,
            params={"index_path": "/path/to/index", "embedding_model": "all-MiniLM-L6-v2"}
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

    # If config is a RetrieverConfig
    elif isinstance(config, RetrieverConfig):
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
