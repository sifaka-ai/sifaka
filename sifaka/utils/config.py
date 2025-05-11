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

2. **Model Configuration**
   - **ModelConfig**: Base configuration for model providers
   - **OpenAIConfig**: Configuration for OpenAI model providers
   - **AnthropicConfig**: Configuration for Anthropic model providers
   - **GeminiConfig**: Configuration for Google Gemini model providers

3. **Rule Configuration**
   - **RuleConfig**: Configuration for rules
   - **RulePriority**: Enumeration of rule priority levels

4. **Critic Configuration**
   - **CriticConfig**: Base configuration for critics
   - **PromptCriticConfig**: Configuration for prompt-based critics
   - **ReflexionCriticConfig**: Configuration for reflexion critics
   - **ConstitutionalCriticConfig**: Configuration for constitutional critics
   - **SelfRefineCriticConfig**: Configuration for self-refine critics
   - **SelfRAGCriticConfig**: Configuration for self-RAG critics
   - **FeedbackCriticConfig**: Configuration for feedback critics
   - **ValueCriticConfig**: Configuration for value critics
   - **LACCriticConfig**: Configuration for LAC critics

5. **Chain Configuration**
   - **ChainConfig**: Configuration for chains
   - **EngineConfig**: Configuration for execution engines
   - **ValidatorConfig**: Configuration for validators
   - **ImproverConfig**: Configuration for improvers
   - **FormatterConfig**: Configuration for formatters

6. **Classifier Configuration**
   - **ClassifierConfig**: Configuration for classifiers
   - **ImplementationConfig**: Configuration for classifier implementations

7. **Retrieval Configuration**
   - **RetrieverConfig**: Configuration for retrievers
   - **RankingConfig**: Configuration for ranking strategies
   - **IndexConfig**: Configuration for index management
   - **QueryProcessingConfig**: Configuration for query processing

8. **Other Configuration**
   - **RetryConfig**: Configuration for retry strategies
   - **ValidationConfig**: Configuration for validation

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

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast, Generic
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
from sifaka.core.base import BaseResult

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


class OpenAIConfig(ModelConfig):
    """
    Configuration for OpenAI model providers.

    This class extends ModelConfig with OpenAI-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for OpenAI-specific parameters through the params dictionary.

    ## Architecture
    OpenAIConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## OpenAI-Specific Parameters
    Common OpenAI-specific parameters that can be included in the params dictionary:
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **frequency_penalty**: Penalty for token frequency (0.0 to 2.0)
    - **presence_penalty**: Penalty for token presence (0.0 to 2.0)
    - **stop**: List of strings that stop generation when encountered
    - **logit_bias**: Dictionary of token biases

    Examples:
        ```python
        from sifaka.utils.config import OpenAIConfig

        # Create an OpenAI configuration
        config = OpenAIConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.0,
                "stop": ["\n", "###"],
                "logit_bias": {50256: -100}  # Bias against a specific token
            }
        )

        # Use the configuration with an OpenAI provider
        provider = OpenAIProvider(model_name="gpt-4", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
            top_p=0.95,
            frequency_penalty=0.7
        )
        ```
    """

    pass


class AnthropicConfig(ModelConfig):
    """
    Configuration for Anthropic model providers.

    This class extends ModelConfig with Anthropic-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for Anthropic-specific parameters through the params dictionary.

    ## Architecture
    AnthropicConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Anthropic-Specific Parameters
    Common Anthropic-specific parameters that can be included in the params dictionary:
    - **top_k**: Number of tokens to consider for sampling
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **stop_sequences**: List of strings that stop generation when encountered
    - **system_prompt**: System prompt to control Claude's behavior
    - **anthropic_version**: API version to use

    Examples:
        ```python
        from sifaka.utils.config import AnthropicConfig

        # Create an Anthropic configuration
        config = AnthropicConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 50,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
                "system_prompt": "You are Claude, an AI assistant created by Anthropic.",
                "anthropic_version": "2023-06-01"
            }
        )

        # Use the configuration with an Anthropic provider
        provider = AnthropicProvider(model_name="claude-3-opus", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
            top_p=0.95,
            system_prompt="You are Claude, a helpful AI assistant."
        )
        ```
    """

    pass


class GeminiConfig(ModelConfig):
    """
    Configuration for Google Gemini model providers.

    This class extends ModelConfig with Gemini-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for Gemini-specific parameters through the params dictionary.

    ## Architecture
    GeminiConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Gemini-Specific Parameters
    Common Gemini-specific parameters that can be included in the params dictionary:
    - **top_k**: Number of tokens to consider for sampling
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **stop_sequences**: List of strings that stop generation when encountered
    - **safety_settings**: Dictionary of safety settings
    - **candidate_count**: Number of candidate responses to generate

    Examples:
        ```python
        from sifaka.utils.config import GeminiConfig

        # Create a Gemini configuration
        config = GeminiConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 40,
                "top_p": 0.95,
                "stop_sequences": ["###"],
                "safety_settings": {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
                },
                "candidate_count": 1
            }
        )

        # Use the configuration with a Gemini provider
        provider = GeminiProvider(model_name="gemini-pro", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
            top_p=0.98,
            candidate_count=3
        )
        ```
    """

    pass


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
        cost: Computational cost of the critic
        trace_enabled: Whether to enable tracing
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
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Computational cost of the critic",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )


class PromptCriticConfig(CriticConfig):
    """
    Configuration for prompt-based critics.

    This class extends CriticConfig with prompt-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds prompt-specific options.

    ## Architecture
    PromptCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import PromptCriticConfig

    # Create a prompt critic configuration
    config = PromptCriticConfig(
        min_confidence=0.7,
        system_prompt="You are an expert editor.",
        temperature=0.2
    )

    # Use the configuration with a prompt critic
    critic = PromptCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.8)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        system_prompt="You are an expert reviewer."
    )
    ```

    Attributes:
        system_prompt (str): System prompt for the critic
        temperature (float): Temperature for text generation
        max_tokens (int): Maximum number of tokens to generate
    """

    system_prompt: str = Field(
        default="You are a helpful critic that evaluates text quality.",
        description="System prompt for the critic",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate",
    )


class ReflexionCriticConfig(CriticConfig):
    """
    Configuration for reflexion critics.

    This class extends CriticConfig with reflexion-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds reflexion-specific options.

    ## Architecture
    ReflexionCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import ReflexionCriticConfig

    # Create a reflexion critic configuration
    config = ReflexionCriticConfig(
        min_confidence=0.7,
        max_reflexion_iterations=3,
        reflexion_prompt="Reflect on your previous response."
    )

    # Use the configuration with a reflexion critic
    critic = ReflexionCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(max_reflexion_iterations=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        reflexion_prompt="Carefully analyze your previous response."
    )
    ```

    Attributes:
        max_reflexion_iterations (int): Maximum number of reflexion iterations
        reflexion_prompt (str): Prompt for reflexion
        temperature (float): Temperature for text generation
    """

    max_reflexion_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of reflexion iterations",
    )
    reflexion_prompt: str = Field(
        default="Reflect on your previous response and identify any issues.",
        description="Prompt for reflexion",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class ConstitutionalCriticConfig(CriticConfig):
    """
    Configuration for constitutional critics.

    This class extends CriticConfig with constitutional-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds constitutional-specific options.

    ## Architecture
    ConstitutionalCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import ConstitutionalCriticConfig

    # Create a constitutional critic configuration
    config = ConstitutionalCriticConfig(
        min_confidence=0.7,
        constitutions=["Be helpful", "Be accurate", "Be concise"],
        constitution_prompt="Evaluate the response against these principles."
    )

    # Use the configuration with a constitutional critic
    critic = ConstitutionalCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.8)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        constitutions=["Be helpful", "Be accurate", "Be concise", "Be ethical"]
    )
    ```

    Attributes:
        constitutions (List[str]): List of constitutional principles
        constitution_prompt (str): Prompt for constitutional evaluation
        temperature (float): Temperature for text generation
    """

    constitutions: List[str] = Field(
        default_factory=lambda: ["Be helpful", "Be accurate", "Be concise"],
        description="List of constitutional principles",
    )
    constitution_prompt: str = Field(
        default="Evaluate the response against these constitutional principles.",
        description="Prompt for constitutional evaluation",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class SelfRefineCriticConfig(CriticConfig):
    """
    Configuration for self-refine critics.

    This class extends CriticConfig with self-refine-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds self-refine-specific options.

    ## Architecture
    SelfRefineCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import SelfRefineCriticConfig

    # Create a self-refine critic configuration
    config = SelfRefineCriticConfig(
        min_confidence=0.7,
        max_refine_iterations=3,
        refine_prompt="Refine your previous response."
    )

    # Use the configuration with a self-refine critic
    critic = SelfRefineCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(max_refine_iterations=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        refine_prompt="Improve your previous response."
    )
    ```

    Attributes:
        max_refine_iterations (int): Maximum number of refine iterations
        refine_prompt (str): Prompt for refinement
        temperature (float): Temperature for text generation
    """

    max_refine_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of refine iterations",
    )
    refine_prompt: str = Field(
        default="Refine your previous response to improve its quality.",
        description="Prompt for refinement",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class SelfRAGCriticConfig(CriticConfig):
    """
    Configuration for self-RAG critics.

    This class extends CriticConfig with self-RAG-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds self-RAG-specific options.

    ## Architecture
    SelfRAGCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import SelfRAGCriticConfig

    # Create a self-RAG critic configuration
    config = SelfRAGCriticConfig(
        min_confidence=0.7,
        max_rag_iterations=3,
        retriever_config={
            "top_k": 5,
            "score_threshold": 0.5
        }
    )

    # Use the configuration with a self-RAG critic
    critic = SelfRAGCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(max_rag_iterations=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        retriever_config={
            "top_k": 10,
            "score_threshold": 0.7
        }
    )
    ```

    Attributes:
        max_rag_iterations (int): Maximum number of RAG iterations
        retriever_config (Dict[str, Any]): Configuration for the retriever
        temperature (float): Temperature for text generation
    """

    max_rag_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of RAG iterations",
    )
    retriever_config: Dict[str, Any] = Field(
        default_factory=lambda: {"top_k": 3, "score_threshold": 0.5},
        description="Configuration for the retriever",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class FeedbackCriticConfig(CriticConfig):
    """
    Configuration for feedback critics.

    This class extends CriticConfig with feedback-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds feedback-specific options.

    ## Architecture
    FeedbackCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import FeedbackCriticConfig

    # Create a feedback critic configuration
    config = FeedbackCriticConfig(
        min_confidence=0.7,
        feedback_prompt="Provide feedback on this response.",
        feedback_categories=["accuracy", "clarity", "completeness"]
    )

    # Use the configuration with a feedback critic
    critic = FeedbackCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.8)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        feedback_categories=["accuracy", "clarity", "completeness", "relevance"]
    )
    ```

    Attributes:
        feedback_prompt (str): Prompt for feedback
        feedback_categories (List[str]): Categories for feedback
        temperature (float): Temperature for text generation
    """

    feedback_prompt: str = Field(
        default="Provide feedback on this response in the following categories.",
        description="Prompt for feedback",
    )
    feedback_categories: List[str] = Field(
        default_factory=lambda: ["accuracy", "clarity", "completeness"],
        description="Categories for feedback",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class ValueCriticConfig(CriticConfig):
    """
    Configuration for value critics.

    This class extends CriticConfig with value-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds value-specific options.

    ## Architecture
    ValueCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import ValueCriticConfig

    # Create a value critic configuration
    config = ValueCriticConfig(
        min_confidence=0.7,
        values=["helpfulness", "accuracy", "harmlessness"],
        value_prompt="Evaluate the response against these values."
    )

    # Use the configuration with a value critic
    critic = ValueCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.8)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        values=["helpfulness", "accuracy", "harmlessness", "fairness"]
    )
    ```

    Attributes:
        values (List[str]): List of values to evaluate against
        value_prompt (str): Prompt for value evaluation
        temperature (float): Temperature for text generation
    """

    values: List[str] = Field(
        default_factory=lambda: ["helpfulness", "accuracy", "harmlessness"],
        description="List of values to evaluate against",
    )
    value_prompt: str = Field(
        default="Evaluate the response against these values.",
        description="Prompt for value evaluation",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


class LACCriticConfig(CriticConfig):
    """
    Configuration for Language Agent Correction (LAC) critics.

    This class extends CriticConfig with LAC-specific configuration options.
    It inherits all the standard configuration options from CriticConfig and
    adds LAC-specific options.

    ## Architecture
    LACCriticConfig is a simple extension of CriticConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during critic initialization
    and remain immutable throughout the critic's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config import LACCriticConfig

    # Create a LAC critic configuration
    config = LACCriticConfig(
        min_confidence=0.7,
        max_lac_iterations=3,
        lac_prompt="Correct any errors in the response."
    )

    # Use the configuration with a LAC critic
    critic = LACCritic(config=config)

    # Create a new configuration with updated options
    updated_config = config.with_options(max_lac_iterations=5)

    # Create a new configuration with updated params
    updated_config = config.with_params(
        lac_prompt="Fix any mistakes in the response."
    )
    ```

    Attributes:
        max_lac_iterations (int): Maximum number of LAC iterations
        lac_prompt (str): Prompt for LAC
        temperature (float): Temperature for text generation
    """

    max_lac_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of LAC iterations",
    )
    lac_prompt: str = Field(
        default="Correct any errors in the response.",
        description="Prompt for LAC",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )


# Default configurations
DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="default_prompt_critic",
    description="Default prompt critic configuration",
    system_prompt="You are an expert editor focused on clarity and accuracy.",
    temperature=0.7,
)

DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="default_reflexion_critic",
    description="Default reflexion critic configuration",
    system_prompt="You are an expert editor that learns from past feedback.",
    reflection_depth=2,
)

DEFAULT_CONSTITUTIONAL_CONFIG = ConstitutionalCriticConfig(
    name="default_constitutional_critic",
    description="Default constitutional critic configuration",
    principles=[
        "Responses should be helpful, harmless, and honest.",
        "Responses should be accurate and avoid speculation.",
        "Responses should respect user privacy and autonomy.",
    ],
)

DEFAULT_SELF_REFINE_CONFIG = SelfRefineCriticConfig(
    name="default_self_refine_critic",
    description="Default self-refine critic configuration",
    system_prompt="You are an expert editor focused on clarity and accuracy.",
    max_iterations=3,
)

DEFAULT_SELF_RAG_CONFIG = SelfRAGCriticConfig(
    name="default_self_rag_critic",
    description="Default Self-RAG critic configuration",
    system_prompt="You are an expert assistant that provides accurate information.",
    retrieval_threshold=0.6,
)

DEFAULT_FEEDBACK_CONFIG = FeedbackCriticConfig(
    name="default_feedback_critic",
    description="Default feedback critic configuration",
    system_prompt="You are an expert editor focused on providing constructive feedback.",
)

DEFAULT_VALUE_CONFIG = ValueCriticConfig(
    name="default_value_critic",
    description="Default value critic configuration",
    system_prompt="You are an expert evaluator focused on assessing value alignment.",
    values=["helpfulness", "accuracy", "safety"],
)

DEFAULT_LAC_CONFIG = LACCriticConfig(
    name="default_lac_critic",
    description="Default LAC critic configuration",
    system_prompt="You are an expert editor focused on correcting errors.",
    max_lac_iterations=3,
)


class CriticMetadata(BaseResult):
    """
    Metadata for critic results.

    This class provides a standardized structure for critic metadata,
    including scores, feedback, issues, and suggestions.
    It extends BaseResult to provide a consistent result structure
    across the Sifaka framework.

    ## Overview
    The class provides:
    - Score tracking (0.0 to 1.0)
    - Human-readable feedback
    - Issue identification
    - Improvement suggestions
    - Additional metadata storage

    ## Usage Examples
    ```python
    from sifaka.utils.config import CriticMetadata

    # Create basic metadata
    metadata = CriticMetadata(
        score=0.85,
        feedback="Good text quality",
        passed=True,
        message="Critique completed successfully",
        issues=["Could be more concise"],
        suggestions=["Remove redundant phrases"]
    )

    # Create metadata with additional data
    metadata = CriticMetadata(
        score=0.75,
        feedback="Text needs improvement",
        passed=False,
        message="Text needs improvement",
        issues=["Too verbose", "Unclear structure"],
        suggestions=["Simplify language", "Add clear sections"],
        metadata={
            "processing_time": 1.5,
            "confidence": 0.9
        }
    )
    ```

    ## Error Handling
    The class implements:
    - Score range validation (0.0 to 1.0)
    - Required field validation
    - Type checking for all fields
    - Default value handling

    Attributes:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        issues: List of identified issues (inherited from BaseResult)
        suggestions: List of improvement suggestions (inherited from BaseResult)
        metadata: Additional metadata (inherited from BaseResult)
        passed: Whether the critique passed (inherited from BaseResult)
        message: Human-readable message (inherited from BaseResult)
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for the critique (0.0 to 1.0)",
    )
    feedback: str = Field(
        description="Human-readable feedback",
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
        async_enabled: Whether to enable async execution
        timeout: Timeout for chain operations in seconds
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
    async_enabled: bool = Field(
        default=False,
        description="Whether to enable async execution",
    )
    timeout: float = Field(
        default=60.0,
        ge=0.0,
        description="Timeout for chain operations in seconds",
    )


class EngineConfig(BaseConfig):
    """
    Configuration for the execution engine.

    This class provides a standardized way to configure the execution engine
    in the Sifaka framework, with immutable configuration values to ensure
    consistency. It includes settings for retry behavior and backoff strategies.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = EngineConfig(
        max_attempts=3,
        retry_delay=1.0,
        backoff_factor=2.0
    )

    # Create a configuration with jitter
    config = EngineConfig(
        max_attempts=3,
        retry_delay=1.0,
        backoff_factor=2.0,
        max_retry_delay=30.0,
        jitter=True
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_attempts=5)
    ```

    Attributes:
        max_attempts (int): Maximum number of generation attempts
        retry_delay (float): Delay between retry attempts in seconds
        backoff_factor (float): Factor to multiply retry delay by each attempt
        max_retry_delay (float): Maximum retry delay in seconds
        jitter (bool): Whether to add random jitter to retry delays
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of generation attempts",
    )
    retry_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Delay between retry attempts in seconds",
    )
    backoff_factor: float = Field(
        default=1.0,
        ge=1.0,
        description="Factor to multiply retry delay by each attempt",
    )
    max_retry_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum retry delay in seconds",
    )
    jitter: bool = Field(
        default=False,
        description="Whether to add random jitter to retry delays",
    )


class ValidatorConfig(BaseConfig):
    """
    Configuration for validators.

    This class provides a standardized way to configure validators in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ValidatorConfig(
        timeout=10.0
    )

    # Create a configuration with params
    config = ValidatorConfig(
        timeout=10.0,
        params={
            "min_length": 10,
            "max_length": 1000
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(timeout=20.0)
    ```

    Attributes:
        timeout (float): Timeout for validation operations in seconds
        prioritize_by_cost (bool): Whether to prioritize validators by cost
        parallel_validation (bool): Whether to run validators in parallel
    """

    timeout: float = Field(
        default=10.0,
        ge=0.0,
        description="Timeout for validation operations in seconds",
    )
    prioritize_by_cost: bool = Field(
        default=False,
        description="Whether to prioritize validators by cost",
    )
    parallel_validation: bool = Field(
        default=False,
        description="Whether to run validators in parallel",
    )


class ImproverConfig(BaseConfig):
    """
    Configuration for improvers.

    This class provides a standardized way to configure improvers in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ImproverConfig(
        timeout=30.0,
        max_improvement_attempts=3
    )

    # Create a configuration with params
    config = ImproverConfig(
        timeout=30.0,
        max_improvement_attempts=3,
        params={
            "system_prompt": "You are an expert editor.",
            "improvement_strategy": "iterative"
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_improvement_attempts=5)
    ```

    Attributes:
        timeout (float): Timeout for improvement operations in seconds
        max_improvement_attempts (int): Maximum number of improvement attempts
    """

    timeout: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout for improvement operations in seconds",
    )
    max_improvement_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of improvement attempts",
    )


class FormatterConfig(BaseConfig):
    """
    Configuration for formatters.

    This class provides a standardized way to configure formatters in the Sifaka
    framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = FormatterConfig(
        include_metadata=True,
        include_validation_results=True
    )

    # Create a configuration with params
    config = FormatterConfig(
        include_metadata=True,
        include_validation_results=True,
        params={
            "format": "json",
            "pretty_print": True
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(include_metadata=False)
    ```

    Attributes:
        include_metadata (bool): Whether to include metadata in results
        include_validation_results (bool): Whether to include validation results in results
    """

    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results",
    )
    include_validation_results: bool = Field(
        default=True,
        description="Whether to include validation results in results",
    )


class ClassifierConfig(BaseConfig, Generic[C]):
    """
    Configuration for classifiers.

    This class provides a consistent way to configure classifiers across the Sifaka framework.
    It handles common configuration options like min_confidence and cache_size, while
    allowing classifier-specific options through the params dictionary.

    The class is generic over the type of labels (C) that the classifier can produce.
    This allows for type-safe configuration of classifiers with different label types.

    Attributes:
        min_confidence: Minimum confidence threshold
        cache_size: Size of the classifier's result cache
        labels: List of classification labels
        cost: Computational cost of the classifier
        multi_label: Whether to allow multiple labels
        return_probabilities: Whether to return probabilities
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
    multi_label: bool = Field(
        default=False,
        description="Whether to allow multiple labels",
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return probabilities",
    )


class ImplementationConfig(BaseConfig):
    """
    Configuration for classifier implementations.

    This class provides a standardized way to configure classifier implementations
    in the Sifaka framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = ImplementationConfig(
        min_confidence=0.7,
        cache_size=100
    )

    # Create a configuration with params
    config = ImplementationConfig(
        min_confidence=0.7,
        cache_size=100,
        params={
            "model_path": "/path/to/model",
            "tokenizer_path": "/path/to/tokenizer"
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(min_confidence=0.8)
    ```

    Attributes:
        min_confidence (float): Minimum confidence threshold (0.0 to 1.0)
        cache_size (int): Size of the classifier's result cache
        batch_size (int): Batch size for processing multiple inputs
        labels (List[str]): List of classification labels
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
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for processing multiple inputs",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="List of classification labels",
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
        rerank_results: Whether to rerank results
        include_metadata: Whether to include metadata in results
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
    rerank_results: bool = Field(
        default=False,
        description="Whether to rerank results",
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results",
    )


class RankingConfig(BaseConfig):
    """
    Configuration for ranking strategies.

    This class provides a standardized way to configure ranking strategies
    in the Sifaka framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = RankingConfig(
        top_k=5,
        score_threshold=0.5
    )

    # Create a configuration with params
    config = RankingConfig(
        top_k=5,
        score_threshold=0.5,
        params={
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "normalize_scores": True
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(top_k=10)
    ```

    Attributes:
        top_k (int): Number of top results to return
        score_threshold (Optional[float]): Minimum score threshold for results
        normalize_scores (bool): Whether to normalize scores
        rerank_results (bool): Whether to rerank results
    """

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top results to return",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold for results",
    )
    normalize_scores: bool = Field(
        default=False,
        description="Whether to normalize scores",
    )
    rerank_results: bool = Field(
        default=False,
        description="Whether to rerank results",
    )


class IndexConfig(BaseConfig):
    """
    Configuration for index management.

    This class provides a standardized way to configure index management
    in the Sifaka framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = IndexConfig(
        index_name="my-index",
        dimension=768
    )

    # Create a configuration with params
    config = IndexConfig(
        index_name="my-index",
        dimension=768,
        params={
            "metric": "cosine",
            "index_type": "hnsw",
            "ef_construction": 200,
            "M": 16
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(dimension=1024)
    ```

    Attributes:
        index_name (str): Name of the index
        dimension (int): Dimension of the vectors
        metric (str): Distance metric to use
        index_type (str): Type of index to use
    """

    index_name: str = Field(
        default="default",
        description="Name of the index",
    )
    dimension: int = Field(
        default=768,
        ge=1,
        description="Dimension of the vectors",
    )
    metric: str = Field(
        default="cosine",
        description="Distance metric to use",
    )
    index_type: str = Field(
        default="hnsw",
        description="Type of index to use",
    )


class QueryProcessingConfig(BaseConfig):
    """
    Configuration for query processing.

    This class provides a standardized way to configure query processing
    in the Sifaka framework, with immutable configuration values to ensure consistency.

    ## Architecture
    Extends BaseConfig to provide consistent configuration
    handling across all Sifaka components.

    ## Examples
    ```python
    # Create a basic configuration
    config = QueryProcessingConfig(
        expand_query=True,
        max_query_length=100
    )

    # Create a configuration with params
    config = QueryProcessingConfig(
        expand_query=True,
        max_query_length=100,
        params={
            "expansion_model": "doc2query-t5-base",
            "num_expansions": 3
        }
    )

    # Create a new configuration with updated options
    updated_config = config.with_options(max_query_length=200)
    ```

    Attributes:
        expand_query (bool): Whether to expand the query
        max_query_length (int): Maximum length of the query
        remove_stopwords (bool): Whether to remove stopwords
        apply_stemming (bool): Whether to apply stemming
    """

    expand_query: bool = Field(
        default=False,
        description="Whether to expand the query",
    )
    max_query_length: int = Field(
        default=100,
        ge=1,
        description="Maximum length of the query",
    )
    remove_stopwords: bool = Field(
        default=False,
        description="Whether to remove stopwords",
    )
    apply_stemming: bool = Field(
        default=False,
        description="Whether to apply stemming",
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
        from sifaka.utils.config import PromptCriticConfig
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
        from sifaka.utils.config import OpenAIConfig
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
        from sifaka.utils.config import standardize_chain_config, ChainConfig, OpenAIConfig

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
        # VectorRetrieverConfig would be defined in utils/config.py
        VectorRetrieverConfig = RetrieverConfig  # Using base class as example
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
