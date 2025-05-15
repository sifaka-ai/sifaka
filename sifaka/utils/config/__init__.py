from typing import Any, List

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
from .base import BaseConfig
from .chain import (
    ChainConfig,
    EngineConfig,
    ValidatorConfig,
    ImproverConfig,
    FormatterConfig,
    standardize_chain_config,
)
from .classifiers import (
    ClassifierConfig,
    ImplementationConfig,
    standardize_classifier_config,
    extract_classifier_config_params,
)
from .critics import (
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
    standardize_critic_config,
)
from .critics import (
    DEFAULT_PROMPT_CRITIC_CONFIG,
    DEFAULT_REFLEXION_CRITIC_CONFIG,
    DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG,
    DEFAULT_SELF_REFINE_CRITIC_CONFIG,
    DEFAULT_SELF_RAG_CRITIC_CONFIG,
    DEFAULT_FEEDBACK_CRITIC_CONFIG,
    DEFAULT_VALUE_CRITIC_CONFIG,
    DEFAULT_LAC_CRITIC_CONFIG,
)
from .models import (
    ModelConfig,
    OpenAIConfig,
    AnthropicConfig,
    GeminiConfig,
    standardize_model_config,
)
from .retrieval import (
    RetrieverConfig,
    RankingConfig,
    IndexConfig,
    QueryProcessingConfig,
    standardize_retrieval_config,
)
from .rules import RuleConfig, standardize_rule_config

__all__: List[Any] = [
    "BaseConfig",
    "ChainConfig",
    "EngineConfig",
    "ValidatorConfig",
    "ImproverConfig",
    "FormatterConfig",
    "standardize_chain_config",
    "ClassifierConfig",
    "ImplementationConfig",
    "standardize_classifier_config",
    "extract_classifier_config_params",
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
    "standardize_critic_config",
    "DEFAULT_PROMPT_CRITIC_CONFIG",
    "DEFAULT_REFLEXION_CRITIC_CONFIG",
    "DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG",
    "DEFAULT_SELF_REFINE_CRITIC_CONFIG",
    "DEFAULT_SELF_RAG_CRITIC_CONFIG",
    "DEFAULT_FEEDBACK_CRITIC_CONFIG",
    "DEFAULT_VALUE_CRITIC_CONFIG",
    "DEFAULT_LAC_CRITIC_CONFIG",
    "ModelConfig",
    "OpenAIConfig",
    "AnthropicConfig",
    "GeminiConfig",
    "standardize_model_config",
    "RetrieverConfig",
    "RankingConfig",
    "IndexConfig",
    "QueryProcessingConfig",
    "standardize_retrieval_config",
    "RuleConfig",
    "standardize_rule_config",
]
