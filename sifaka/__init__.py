"""
Sifaka: A framework for building reliable and reflective AI systems.
"""

from typing import List
import os

# Define version
__version__: str = "0.1.0"

# Core interfaces and base classes
# These should have minimal dependencies themselves
from sifaka.interfaces import (
    ModelProviderProtocol,
    RuleProtocol,
    CriticProtocol,
    ChainPluginProtocol,
    ImproverProtocol,
    ValidatorProtocol,
    ComponentProtocol,
)

# Base result types with no circular dependencies
from sifaka.core.results import (
    BaseResult,
    ChainResult,
    ValidationResult,
    RuleResult,
    GenerationResult,
    ModelResult,
    CriticResult,
    ErrorResult,
    RetrievalResult,
    ClassificationResult,
)

# Initialize component registry if not disabled
# This loads all component implementations and registers their factory functions
# The environment variable SIFAKA_SKIP_REGISTRY_INIT can be set to disable this behavior
if os.environ.get("SIFAKA_SKIP_REGISTRY_INIT", "").lower() != "true":
    try:
        from sifaka.core.initialize_registry import initialize_registry

        initialize_registry()
    except ImportError:
        # If the registry module is not available, skip initialization
        # This can happen during early stages of package installation
        pass

# Expose the core interfaces in __all__
__all__: List[str] = [
    # Protocols
    "ModelProviderProtocol",
    "RuleProtocol",
    "CriticProtocol",
    "ChainPluginProtocol",
    "ImproverProtocol",
    "ValidatorProtocol",
    "ComponentProtocol",
    # Result types
    "BaseResult",
    "ChainResult",
    "ValidationResult",
    "RuleResult",
    "GenerationResult",
    "ModelResult",
    "CriticResult",
    "ErrorResult",
    "RetrievalResult",
    "ClassificationResult",
]

# Import concrete implementations separately to avoid circular dependencies
# Users should import these directly from their respective modules
# Examples:
# from sifaka.chain import Chain
# from sifaka.models.providers.openai import OpenAIProvider
# from sifaka.rules.base import Rule
