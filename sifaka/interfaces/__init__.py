"""
Interfaces for Sifaka

This package provides the core interfaces for the Sifaka framework.
"""

from typing import List

# Core interfaces
from .core import (
    IdentifiableProtocol,
    ConfigurableProtocol,
    StatefulProtocol,
    LoggableProtocol,
    TraceableProtocol,
    ComponentProtocol,
    PluginProtocol,
)

# Component protocols
from .component import ComponentProtocol as ComponentBaseProtocol
from .model import ModelProviderProtocol, LanguageModelProviderProtocol
from .rule import SimpleRuleProtocol, RuleProtocol, RuleResultHandlerProtocol, ValidatableProtocol
from .critic import (
    CriticProtocol,
    TextValidator,
    TextImprover,
    TextCritic,
    LLMProvider,
    PromptFactory,
    CritiqueResult,
)
from .chain_components import ChainPluginProtocol
from .validator import ValidatorProtocol
from .improver import ImproverProtocol

# Chain component protocols
from .chain_components import (
    ValidationResult,
    ModelProtocol,
    ValidatorProtocol as ChainValidatorProtocol,
    ImproverProtocol as ChainImproverProtocol,
    FormatterProtocol,
)

# Classifier protocols
from .classifier import (
    ClassifierImplementationProtocol,
    ClassifierPluginProtocol,
)

# API client and token counter protocols
from .client import APIClientProtocol
from .counter import TokenCounterProtocol

# Retrieval protocols
from .retrieval import (
    RetrieverProtocol,
    DocumentStoreProtocol,
    IndexManagerProtocol,
    QueryProcessorProtocol,
)

__all__: List[str] = [
    # Core interfaces
    "IdentifiableProtocol",
    "ConfigurableProtocol",
    "StatefulProtocol",
    "LoggableProtocol",
    "TraceableProtocol",
    "ComponentProtocol",
    "PluginProtocol",
    # Component protocols
    "ComponentBaseProtocol",
    "ModelProviderProtocol",
    "LanguageModelProviderProtocol",
    "RuleProtocol",
    "SimpleRuleProtocol",
    "CriticProtocol",
    "ChainPluginProtocol",
    "ValidatorProtocol",
    "ImproverProtocol",
    # Chain component protocols
    "ValidationResult",
    "ModelProtocol",
    "ChainValidatorProtocol",
    "ChainImproverProtocol",
    "FormatterProtocol",
    # Classifier protocols
    "ClassifierImplementationProtocol",
    "ClassifierPluginProtocol",
    # API client and token counter protocols
    "APIClientProtocol",
    "TokenCounterProtocol",
    # Retrieval protocols
    "RetrieverProtocol",
    "DocumentStoreProtocol",
    "IndexManagerProtocol",
    "QueryProcessorProtocol",
    # Additional interfaces
    "RuleResultHandlerProtocol",
    "ValidatableProtocol",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    "CritiqueResult",
]
