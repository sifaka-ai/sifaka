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
The module defines a hierarchy of configuration classes organized into separate modules:

1. **base.py**: Base configuration class for all components
2. **models.py**: Model provider configurations
3. **rules.py**: Rule configurations
4. **critics.py**: Critic configurations
5. **chain.py**: Chain configurations
6. **classifiers.py**: Classifier configurations
7. **retrieval.py**: Retrieval configurations

## Usage Examples
```python
# Import directly from specific modules
from sifaka.utils.config.base import BaseConfig
from sifaka.utils.config.models import ModelConfig, OpenAIConfig, standardize_model_config

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

# Create specialized model configuration
openai_config = OpenAIConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Use standardization function
config = standardize_model_config(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

# Re-export standardization functions for backward compatibility
from sifaka.utils.config.base import BaseConfig
from sifaka.utils.config.models import ModelConfig, OpenAIConfig, AnthropicConfig, GeminiConfig
from sifaka.utils.config.rules import RuleConfig, RulePriority
from sifaka.utils.config.critics import (
    CriticConfig,
    CriticMetadata,
    PromptCriticConfig,
    ReflexionCriticConfig,
    ConstitutionalCriticConfig,
    SelfRefineCriticConfig,
    SelfRAGCriticConfig,
    FeedbackCriticConfig,
    ValueCriticConfig,
    LACCriticConfig,
    DEFAULT_PROMPT_CRITIC_CONFIG as DEFAULT_PROMPT_CONFIG,
    DEFAULT_REFLEXION_CRITIC_CONFIG as DEFAULT_REFLEXION_CONFIG,
    DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG as DEFAULT_CONSTITUTIONAL_CONFIG,
    DEFAULT_SELF_REFINE_CRITIC_CONFIG as DEFAULT_SELF_REFINE_CONFIG,
    DEFAULT_SELF_RAG_CRITIC_CONFIG as DEFAULT_SELF_RAG_CONFIG,
    DEFAULT_FEEDBACK_CRITIC_CONFIG as DEFAULT_FEEDBACK_CONFIG,
    DEFAULT_VALUE_CRITIC_CONFIG as DEFAULT_VALUE_CONFIG,
    DEFAULT_LAC_CRITIC_CONFIG as DEFAULT_LAC_CONFIG,
)
from sifaka.utils.config.chain import ChainConfig, EngineConfig, ValidatorConfig, ImproverConfig
from sifaka.utils.config.classifiers import ClassifierConfig
from sifaka.utils.config.retrieval import (
    RetrieverConfig,
    QueryProcessingConfig,
    RankingConfig,
    IndexConfig,
)


# Additional functions needed for backward compatibility
def extract_classifier_config_params(*args, **kwargs):
    """Extract classifier configuration parameters."""
    from sifaka.utils.config.classifiers import (
        extract_classifier_config_params as _extract_classifier_config_params,
    )

    return _extract_classifier_config_params(*args, **kwargs)


# Define what symbols to export
__all__ = [
    # Base config
    "BaseConfig",
    # Model configs
    "ModelConfig",
    "OpenAIConfig",
    "AnthropicConfig",
    "GeminiConfig",
    # Rule configs
    "RuleConfig",
    "RulePriority",
    # Critic configs
    "CriticConfig",
    "CriticMetadata",
    "PromptCriticConfig",
    "ReflexionCriticConfig",
    "ConstitutionalCriticConfig",
    "SelfRefineCriticConfig",
    "SelfRAGCriticConfig",
    "FeedbackCriticConfig",
    "ValueCriticConfig",
    "LACCriticConfig",
    # Chain configs
    "ChainConfig",
    "EngineConfig",
    "ValidatorConfig",
    "ImproverConfig",
    # Classifier configs
    "ClassifierConfig",
    "extract_classifier_config_params",
    # Retrieval configs
    "RetrieverConfig",
    "QueryProcessingConfig",
    "RankingConfig",
    "IndexConfig",
    # Default configs
    "DEFAULT_PROMPT_CONFIG",
    "DEFAULT_REFLEXION_CONFIG",
    "DEFAULT_CONSTITUTIONAL_CONFIG",
    "DEFAULT_SELF_REFINE_CONFIG",
    "DEFAULT_SELF_RAG_CONFIG",
    "DEFAULT_FEEDBACK_CONFIG",
    "DEFAULT_VALUE_CONFIG",
    "DEFAULT_LAC_CONFIG",
    # Standardization functions
    "standardize_rule_config",
    "standardize_critic_config",
    "standardize_model_config",
    "standardize_chain_config",
    "standardize_classifier_config",
    "standardize_retriever_config",
]


# Re-export standardization functions
# These will need to be implemented in the respective modules
def standardize_rule_config(*args, **kwargs):
    """Standardize rule configuration."""
    from sifaka.utils.config.rules import standardize_rule_config as _standardize_rule_config

    return _standardize_rule_config(*args, **kwargs)


def standardize_critic_config(*args, **kwargs):
    """Standardize critic configuration."""
    from sifaka.utils.config.critics import standardize_critic_config as _standardize_critic_config

    return _standardize_critic_config(*args, **kwargs)


def standardize_model_config(*args, **kwargs):
    """Standardize model provider configuration."""
    from sifaka.utils.config.models import standardize_model_config as _standardize_model_config

    return _standardize_model_config(*args, **kwargs)


def standardize_chain_config(*args, **kwargs):
    """Standardize chain configuration."""
    from sifaka.utils.config.chain import standardize_chain_config as _standardize_chain_config

    return _standardize_chain_config(*args, **kwargs)


def standardize_classifier_config(*args, **kwargs):
    """Standardize classifier configuration."""
    from sifaka.utils.config.classifiers import (
        standardize_classifier_config as _standardize_classifier_config,
    )

    return _standardize_classifier_config(*args, **kwargs)


def standardize_retriever_config(*args, **kwargs):
    """Standardize retriever configuration."""
    from sifaka.utils.config.retrieval import (
        standardize_retriever_config as _standardize_retriever_config,
    )

    return _standardize_retriever_config(*args, **kwargs)
