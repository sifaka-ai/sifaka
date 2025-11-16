"""Type definitions and enums for Sifaka.

This module provides type-safe enumerations and type aliases to replace
magic strings throughout the codebase.
"""

import sys
from enum import Enum
from typing import Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class CriticType(str, Enum):
    """Available critic types.

    Use these constants instead of strings when specifying critics:
        config = Config(critics=[CriticType.REFLEXION, CriticType.SELF_RAG])
    """

    REFLEXION = "reflexion"
    SELF_RAG = "self_rag"
    SELF_REFINE = "self_refine"
    SELF_CONSISTENCY = "self_consistency"
    CONSTITUTIONAL = "constitutional"
    META_REWARDING = "meta_rewarding"
    N_CRITICS = "n_critics"
    SELF_TAUGHT_EVALUATOR = "self_taught_evaluator"
    AGENT4DEBATE = "agent4debate"
    STYLE = "style"
    PROMPT = "prompt"

    @classmethod
    def values(cls) -> list[str]:
        """Get all critic type values as strings."""
        return [c.value for c in cls]

    @classmethod
    def from_string(cls, value: str) -> "CriticType":
        """Convert string to CriticType, with helpful error message."""
        try:
            return cls(value)
        except ValueError:
            available = ", ".join(cls.values())
            raise ValueError(
                f"Unknown critic type: '{value}'. Available critics: {available}"
            )


class ValidatorType(str, Enum):
    """Available validator types.

    Use these constants instead of strings when specifying validators:
        config = Config(validators=[ValidatorType.LENGTH, ValidatorType.FORMAT])
    """

    LENGTH = "length"
    FORMAT = "format"
    CONTENT = "content"
    READABILITY = "readability"
    GUARDRAILS = "guardrails"
    CUSTOM = "custom"
    COMPOSABLE = "composable"

    @classmethod
    def values(cls) -> list[str]:
        """Get all validator type values as strings."""
        return [v.value for v in cls]

    @classmethod
    def from_string(cls, value: str) -> "ValidatorType":
        """Convert string to ValidatorType, with helpful error message."""
        try:
            return cls(value)
        except ValueError:
            available = ", ".join(cls.values())
            raise ValueError(
                f"Unknown validator type: '{value}'. Available validators: {available}"
            )


class StorageType(str, Enum):
    """Available storage backend types.

    Use these constants instead of strings when specifying storage:
        storage_config = StorageConfig(backend=StorageType.REDIS)
    """

    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    CUSTOM = "custom"

    @classmethod
    def values(cls) -> list[str]:
        """Get all storage type values as strings."""
        return [s.value for s in cls]


class Provider(str, Enum):
    """LLM provider types.

    Already defined in llm_client.py but re-exported here for convenience.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


# Type aliases for clarity
CriticName = Union[str, CriticType]
ValidatorName = Union[str, ValidatorType]
StorageBackendName = Union[str, StorageType]
ProviderName = Union[str, Provider]

# Literal types for stricter typing where needed
CriticTypeLiteral = Literal[
    "reflexion",
    "self_rag",
    "self_refine",
    "self_consistency",
    "constitutional",
    "meta_rewarding",
    "n_critics",
    "self_taught_evaluator",
    "agent4debate",
    "style",
    "prompt",
]

ValidatorTypeLiteral = Literal[
    "length", "format", "content", "readability", "guardrails", "custom", "composable"
]

StorageTypeLiteral = Literal["memory", "file", "redis", "custom"]

ProviderLiteral = Literal["openai", "anthropic", "gemini", "groq", "ollama"]


# Export all types
__all__ = [
    "CriticType",
    "ValidatorType",
    "StorageType",
    "Provider",
    "CriticName",
    "ValidatorName",
    "StorageBackendName",
    "ProviderName",
    "CriticTypeLiteral",
    "ValidatorTypeLiteral",
    "StorageTypeLiteral",
    "ProviderLiteral",
]
