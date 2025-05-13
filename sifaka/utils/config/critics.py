"""
Critic Configuration Module

This module provides configuration classes and standardization functions for critics.

## Overview
The critic configuration module defines configuration classes for critics in the Sifaka framework.
It provides a consistent approach to configuring critics with standardized parameter handling,
validation, and serialization.

## Components
- **CriticMetadata**: Metadata for critics
- **CriticConfig**: Base configuration for critics
- **PromptCriticConfig**: Configuration for prompt-based critics
- **ReflexionCriticConfig**: Configuration for reflexion critics
- **ConstitutionalCriticConfig**: Configuration for constitutional critics
- **SelfRefineCriticConfig**: Configuration for self-refine critics
- **SelfRAGCriticConfig**: Configuration for self-RAG critics
- **FeedbackCriticConfig**: Configuration for feedback critics
- **ValueCriticConfig**: Configuration for value critics
- **LACCriticConfig**: Configuration for LAC critics
- **standardize_critic_config**: Standardization function for critic configurations
- **Default configurations**: Default configurations for different critic types

## Usage Examples
```python
from sifaka.utils.config.critics import (
    CriticConfig, PromptCriticConfig, standardize_critic_config
)

# Create a basic critic configuration
config = CriticConfig(
    name="my_critic",
    description="A custom critic",
    min_confidence=0.8,
    max_attempts=3
)

# Create a prompt critic configuration
prompt_config = PromptCriticConfig(
    name="prompt_critic",
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)

# Use standardization function
config = standardize_critic_config(
    min_confidence=0.8,
    max_attempts=3,
    params={
        "system_prompt": "You are an expert editor."
    }
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import Field
from .base import BaseConfig

T = TypeVar("T", bound="CriticConfig")


class CriticMetadata(BaseConfig):
    """
    Metadata for critics.

    This class provides a consistent way to store metadata about critics,
    such as performance metrics, usage statistics, and other information.

    ## Architecture
    CriticMetadata extends BaseConfig with critic-specific metadata fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Metadata objects are typically created during critic initialization and
    updated throughout the critic's lifecycle. New metadata objects can be
    created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import CriticMetadata

    # Create critic metadata
    metadata = CriticMetadata(
        name="performance_metrics",
        params={
            "total_calls": 0,
            "successful_calls": 0,
            "average_latency_ms": 0.0
        }
    )

    # Update metadata
    updated_metadata = metadata.with_params(
        total_calls=10,
        successful_calls=8,
        average_latency_ms=150.5
    ) if metadata else ""
    ```

    Attributes:
        name: Metadata name
        description: Metadata description
        params: Dictionary of metadata parameters
    """

    pass


class CriticConfig(BaseConfig):
    """
    Base configuration for critics.

    This class provides a consistent way to configure critics across the Sifaka framework.
    It handles common configuration options like min_confidence and max_attempts, while
    allowing critic-specific options through the params dictionary.

    ## Architecture
    CriticConfig extends BaseConfig with critic-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import CriticConfig

    # Create a critic configuration
    config = CriticConfig(
        name="my_critic",
        description="A custom critic",
        min_confidence=0.8,
        max_attempts=3,
        cache_size=100,
        priority=1,
        cost=0.01,
        track_performance=True,
        system_prompt="You are an expert editor.",
        temperature=0.7,
        max_tokens=1000
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Min confidence: {config.min_confidence}")
    print(f"System prompt: {config.system_prompt}")
    print(f"Temperature: {config.temperature}")

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.9) if config else ""

    # Create a new configuration with updated params
    updated_config = config.with_params(top_p=0.95) if config else ""
    ```

    Attributes:
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the result cache
        priority: Priority of the critic
        cost: Computational cost of the critic
        track_performance: Whether to track performance metrics
        system_prompt: System prompt for the model
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
    """

    min_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    max_attempts: int = Field(default=3, ge=1, description="Maximum number of improvement attempts")
    cache_size: int = Field(default=100, ge=0, description="Size of the result cache")
    priority: int = Field(default=1, description="Priority of the critic")
    cost: Optional[float] = Field(default=None, description="Computational cost of the critic")
    track_performance: bool = Field(
        default=True, description="Whether to track performance metrics"
    )
    system_prompt: str = Field(
        default="You are a helpful critic that evaluates text quality.",
        description="System prompt for the model",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Temperature for text generation"
    )
    max_tokens: int = Field(default=1000, ge=1, description="Maximum number of tokens to generate")


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt-based critics.

    This class extends CriticConfig with prompt-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds prompt-specific options.

    ## Architecture
    PromptCriticConfig extends CriticConfig with prompt-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import PromptCriticConfig

    # Create a prompt critic configuration
    config = PromptCriticConfig(
        name="prompt_critic",
        description="A prompt-based critic",
        system_prompt="You are an expert editor.",
        temperature=0.7,
        max_tokens=1000,
        min_confidence=0.8,
        max_attempts=3
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"System prompt: {config.system_prompt}")
    print(f"Temperature: {config.temperature}")

    # Create a new configuration with updated options
    updated_config = config.with_options(temperature=0.9) if config else ""

    # Create a new configuration with updated params
    updated_config = config.with_params(top_p=0.95) if config else ""
    ```

    Attributes:
        memory_buffer_size: Size of the memory buffer
        eager_initialization: Whether to initialize components eagerly
        track_errors: Whether to track errors
    """

    system_prompt: str = Field(
        default="You are a helpful assistant that provides high-quality feedback and improvements for text.",
        description="System prompt for the model",
    )
    memory_buffer_size: int = Field(default=10, ge=0, description="Size of the memory buffer")
    eager_initialization: bool = Field(
        default=False, description="Whether to initialize components eagerly"
    )
    track_errors: bool = Field(default=True, description="Whether to track errors")


class ReflexionCriticConfig(PromptCriticConfig):
    """
    Configuration for reflexion critics.

    This class extends PromptCriticConfig with reflexion-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds reflexion-specific options like reflection_count and reflection_prompt_template.

    ## Architecture
    ReflexionCriticConfig extends PromptCriticConfig with reflexion-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import ReflexionCriticConfig

    # Create a reflexion critic configuration
    config = ReflexionCriticConfig(
        name="reflexion_critic",
        description="A reflexion-based critic",
        system_prompt="You are an expert editor.",
        reflection_count=3,
        reflection_prompt_template="Reflect on the following text: {text}"
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Reflection count: {config.reflection_count}")
    print(f"Reflection prompt template: {config.reflection_prompt_template}")

    # Create a new configuration with updated options
    updated_config = config.with_options(reflection_count=5) if config else ""
    ```

    Attributes:
        reflection_count: Number of reflections to generate
        reflection_prompt_template: Template for reflection prompts
    """

    reflection_count: int = Field(default=3, ge=1, description="Number of reflections to generate")
    reflection_prompt_template: Optional[str] = Field(
        default=None, description="Template for reflection prompts"
    )


class ConstitutionalCriticConfig(PromptCriticConfig):
    """
    Configuration for constitutional critics.

    This class extends PromptCriticConfig with constitutional-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds constitutional-specific options like principles and constitution_prompt.

    ## Architecture
    ConstitutionalCriticConfig extends PromptCriticConfig with constitutional-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import ConstitutionalCriticConfig

    # Create a constitutional critic configuration
    config = ConstitutionalCriticConfig(
        name="constitutional_critic",
        description="A constitutional critic",
        system_prompt="You are an expert evaluator.",
        principles=["Be helpful", "Be accurate", "Be concise"],
        constitution_prompt="Evaluate the text against these principles."
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Principles: {config.principles}")
    print(f"Constitution prompt: {config.constitution_prompt}")

    # Create a new configuration with updated options
    updated_config = config.with_options(
        principles=["Be helpful", "Be accurate", "Be concise", "Be ethical"]
    ) if config else ""
    ```

    Attributes:
        principles: List of constitutional principles
        constitution_prompt: Prompt for constitutional evaluation
    """

    principles: List[str] = Field(
        default_factory=lambda: ["Be helpful", "Be accurate", "Be concise"],
        description="List of constitutional principles",
    )
    constitution_prompt: str = Field(
        default="Evaluate the text against these constitutional principles.",
        description="Prompt for constitutional evaluation",
    )


class SelfRefineCriticConfig(PromptCriticConfig):
    """
    Configuration for self-refine critics.

    This class extends PromptCriticConfig with self-refine-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds self-refine-specific options like max_iterations and refine_prompt.

    ## Architecture
    SelfRefineCriticConfig extends PromptCriticConfig with self-refine-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import SelfRefineCriticConfig

    # Create a self-refine critic configuration
    config = SelfRefineCriticConfig(
        name="self_refine_critic",
        description="A self-refine critic",
        system_prompt="You are an expert editor.",
        max_iterations=3,
        refine_prompt="Refine the following text to improve its quality."
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Refine prompt: {config.refine_prompt}")

    # Create a new configuration with updated options
    updated_config = config.with_options(max_iterations=5) if config else ""
    ```

    Attributes:
        max_iterations: Maximum number of refinement iterations
        refine_prompt: Prompt for refinement
    """

    max_iterations: int = Field(
        default=3, ge=1, description="Maximum number of refinement iterations"
    )
    refine_prompt: str = Field(
        default="Refine the following text to improve its quality.",
        description="Prompt for refinement",
    )


class SelfRAGCriticConfig(PromptCriticConfig):
    """
    Configuration for self-RAG critics.

    This class extends PromptCriticConfig with self-RAG-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds self-RAG-specific options like retrieval_threshold and retrieval_count.

    ## Architecture
    SelfRAGCriticConfig extends PromptCriticConfig with self-RAG-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import SelfRAGCriticConfig

    # Create a self-RAG critic configuration
    config = SelfRAGCriticConfig(
        name="self_rag_critic",
        description="A self-RAG critic",
        system_prompt="You are an expert assistant.",
        retrieval_threshold=0.7,
        retrieval_count=3
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Retrieval threshold: {config.retrieval_threshold}")
    print(f"Retrieval count: {config.retrieval_count}")

    # Create a new configuration with updated options
    updated_config = config.with_options(retrieval_threshold=0.8) if config else ""
    ```

    Attributes:
        retrieval_threshold: Threshold for retrieval relevance
        retrieval_count: Number of documents to retrieve
    """

    retrieval_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold for retrieval relevance"
    )
    retrieval_count: int = Field(default=3, ge=1, description="Number of documents to retrieve")


class FeedbackCriticConfig(PromptCriticConfig):
    """
    Configuration for feedback critics.

    This class extends PromptCriticConfig with feedback-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds feedback-specific options like feedback_categories and feedback_prompt.

    ## Architecture
    FeedbackCriticConfig extends PromptCriticConfig with feedback-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import FeedbackCriticConfig

    # Create a feedback critic configuration
    config = FeedbackCriticConfig(
        name="feedback_critic",
        description="A feedback critic",
        system_prompt="You are an expert reviewer.",
        feedback_categories=["accuracy", "clarity", "completeness"],
        feedback_prompt="Provide feedback on the following text."
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Feedback categories: {config.feedback_categories}")
    print(f"Feedback prompt: {config.feedback_prompt}")

    # Create a new configuration with updated options
    updated_config = config.with_options(
        feedback_categories=["accuracy", "clarity", "completeness", "relevance"]
    ) if config else ""
    ```

    Attributes:
        feedback_categories: Categories for feedback
        feedback_prompt: Prompt for feedback
    """

    feedback_categories: List[str] = Field(
        default_factory=lambda: ["accuracy", "clarity", "completeness"],
        description="Categories for feedback",
    )
    feedback_prompt: str = Field(
        default="Provide feedback on the following text.", description="Prompt for feedback"
    )


class ValueCriticConfig(PromptCriticConfig):
    """
    Configuration for value critics.

    This class extends PromptCriticConfig with value-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds value-specific options like values and value_prompt.

    ## Architecture
    ValueCriticConfig extends PromptCriticConfig with value-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import ValueCriticConfig

    # Create a value critic configuration
    config = ValueCriticConfig(
        name="value_critic",
        description="A value critic",
        system_prompt="You are an expert evaluator.",
        values=["helpfulness", "accuracy", "harmlessness"],
        value_prompt="Evaluate the text against these values."
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Values: {config.values}")
    print(f"Value prompt: {config.value_prompt}")

    # Create a new configuration with updated options
    updated_config = config.with_options(
        values=["helpfulness", "accuracy", "harmlessness", "fairness"]
    ) if config else ""
    ```

    Attributes:
        values: List of values to evaluate against
        value_prompt: Prompt for value evaluation
    """

    values: List[str] = Field(
        default_factory=lambda: ["helpfulness", "accuracy", "harmlessness"],
        description="List of values to evaluate against",
    )
    value_prompt: str = Field(
        default="Evaluate the text against these values.", description="Prompt for value evaluation"
    )


class LACCriticConfig(PromptCriticConfig):
    """
    Configuration for Language Agent Correction (LAC) critics.

    This class extends PromptCriticConfig with LAC-specific configuration options.
    It inherits all the standard configuration options from PromptCriticConfig and
    adds LAC-specific options like max_iterations and correction_prompt.

    ## Architecture
    LACCriticConfig extends PromptCriticConfig with LAC-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during critic initialization and
    remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.critics import LACCriticConfig

    # Create a LAC critic configuration
    config = LACCriticConfig(
        name="lac_critic",
        description="A LAC critic",
        system_prompt="You are an expert editor.",
        max_iterations=3,
        correction_prompt="Correct any errors in the following text."
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Correction prompt: {config.correction_prompt}")

    # Create a new configuration with updated options
    updated_config = config.with_options(max_iterations=5) if config else ""
    ```

    Attributes:
        max_iterations: Maximum number of correction iterations
        correction_prompt: Prompt for correction
    """

    max_iterations: int = Field(
        default=3, ge=1, description="Maximum number of correction iterations"
    )
    correction_prompt: str = Field(
        default="Correct any errors in the following text.", description="Prompt for correction"
    )


DEFAULT_PROMPT_CRITIC_CONFIG = {
    "system_prompt": "You are a helpful assistant that provides high-quality feedback and improvements for text.",
    "temperature": 0.7,
    "max_tokens": 1000,
    "min_confidence": 0.7,
    "max_attempts": 3,
    "cache_size": 100,
}
DEFAULT_REFLEXION_CRITIC_CONFIG = {**DEFAULT_PROMPT_CRITIC_CONFIG, "reflection_count": 3}
DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "principles": ["Be helpful", "Be accurate", "Be concise"],
    "constitution_prompt": "Evaluate the text against these constitutional principles.",
}
DEFAULT_SELF_REFINE_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "max_iterations": 3,
    "refine_prompt": "Refine the following text to improve its quality.",
}
DEFAULT_SELF_RAG_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "retrieval_threshold": 0.7,
    "retrieval_count": 3,
}
DEFAULT_FEEDBACK_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "feedback_categories": ["accuracy", "clarity", "completeness"],
    "feedback_prompt": "Provide feedback on the following text.",
}
DEFAULT_VALUE_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "values": ["helpfulness", "accuracy", "harmlessness"],
    "value_prompt": "Evaluate the text against these values.",
}
DEFAULT_LAC_CRITIC_CONFIG = {
    **DEFAULT_PROMPT_CRITIC_CONFIG,
    "max_iterations": 3,
    "correction_prompt": "Correct any errors in the following text.",
}


def standardize_critic_config(
    config: Optional[Union[Dict[str, Any], CriticConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[CriticConfig] = CriticConfig,
    **kwargs: Any,
) -> CriticConfig:
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
        from sifaka.utils.config.critics import standardize_critic_config, PromptCriticConfig

        # Create from parameters
        config = standardize_critic_config(
            min_confidence=0.8,
            max_attempts=3,
            params={
                "system_prompt": "You are an expert editor."
            }
        )

        # Create from existing config
        from sifaka.utils.config.critics import CriticConfig
        existing = CriticConfig(min_confidence=0.7)
        updated = standardize_critic_config(
            config=existing,
            params={
                "system_prompt": "You are an expert editor."
            }
        )

        # Create from dictionary
        dict_config = {
            "min_confidence": 0.8,
            "max_attempts": 3,
            "params": {
                "system_prompt": "You are an expert editor."
            }
        }
        config = standardize_critic_config(config=dict_config)

        # Create specialized config
        prompt_config = standardize_critic_config(
            config_class=PromptCriticConfig,
            system_prompt="You are an expert editor.",
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """
    final_params: Dict[str, Any] = {}
    if params:
        final_params.update(params)
    if isinstance(config, dict):
        dict_params = config.pop("params", {}) if config else {}
        final_params.update(dict_params)
        return config_class(**{} if config is None else config, params=final_params, **kwargs)
    elif isinstance(config, CriticConfig):
        final_params.update(config.params)
        config_dict = {**config.model_dump(), "params": final_params, **kwargs}
        return config_class(**config_dict)
    else:
        return config_class(params=final_params, **kwargs)
