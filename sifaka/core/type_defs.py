"""Type definitions for improved type safety throughout Sifaka.

This module provides TypedDict and other type definitions to replace
generic Dict[str, Any] usage with more specific, type-safe alternatives.
"""

import sys
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 12):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

from .types import CriticType


class ToolResultItem(TypedDict):
    """Individual result item from a tool query."""

    title: Optional[str]
    content: str
    url: Optional[str]
    snippet: Optional[str]
    source: Optional[str]
    relevance: Optional[float]


class ToolUsageMetadata(TypedDict):
    """Metadata about tool usage."""

    tool_name: str
    query: str
    result_count: int
    execution_time: float
    status: Literal["success", "error", "timeout"]
    error_message: Optional[str]


class CriticMetadata(TypedDict, total=False):
    """Metadata specific to critics."""

    principles_used: Optional[List[str]]  # For constitutional critic
    consensus_data: Optional[Dict[str, float]]  # For self-consistency
    tools_used: Optional[List[str]]  # For critics that use tools
    confidence_factors: Optional[Dict[str, float]]  # Confidence breakdown


class MiddlewareContext(TypedDict, total=False):
    """Context passed through middleware pipeline."""

    critics: List[Union[str, CriticType]]
    validators: Optional[List[Any]]  # TODO: Replace with ValidatorProtocol
    config: Any  # Circular import if we use Config
    storage: Optional[Any]  # TODO: Replace with StorageBackend protocol
    model: Optional[str]
    temperature: Optional[float]
    max_iterations: Optional[int]
    metadata: Dict[str, Any]


class CriticSettings(TypedDict, total=False):
    """Settings for configuring individual critics."""

    enable_tools: bool
    tool_timeout: float
    base_confidence: float
    context_window: int
    # Critic-specific settings
    num_samples: Optional[int]  # self_consistency
    citation_style: Optional[str]  # academic writing
    formality_level: Optional[str]  # academic writing
    programming_language: Optional[str]  # technical docs
    doc_style: Optional[str]  # technical docs
    constitutional_principles: Optional[List[str]]  # constitutional


class ValidatorSettings(TypedDict, total=False):
    """Settings for configuring individual validators."""

    # Length validator
    min_length: Optional[int]
    max_length: Optional[int]
    # Format validator
    min_paragraphs: Optional[int]
    max_paragraphs: Optional[int]
    min_sentences: Optional[int]
    max_sentences: Optional[int]
    # Content validator
    required_terms: Optional[List[str]]
    forbidden_terms: Optional[List[str]]
    # Custom settings
    custom_rules: Optional[Dict[str, Any]]


class StorageBackendSettings(TypedDict, total=False):
    """Settings for storage backend configuration."""

    # Redis settings
    url: Optional[str]
    host: Optional[str]
    port: Optional[int]
    db: Optional[int]
    password: Optional[str]
    ttl: Optional[int]
    key_prefix: Optional[str]
    # File storage settings
    storage_path: Optional[str]
    file_format: Optional[Literal["json", "yaml", "pickle"]]
    # Common settings
    enable_compression: Optional[bool]
    max_size: Optional[int]
    auto_cleanup: Optional[bool]


class GenerationMetadata(TypedDict):
    """Metadata for text generation results."""

    model: str
    temperature: float
    tokens_used: int
    processing_time: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_estimate: Optional[float]


class ValidationResult(TypedDict):
    """Result from a validator check."""

    validator_name: str
    passed: bool
    message: str
    severity: Literal["error", "warning", "info"]
    details: Optional[Dict[str, Any]]


class PerformanceMetrics(TypedDict):
    """Performance tracking metrics."""

    operation: str
    duration_ms: float
    timestamp: str
    success: bool
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


class ToolCallParams(TypedDict, total=False):
    """Optional parameters for tool calls."""

    max_results: Optional[int]
    filters: Optional[Dict[str, Any]]
    timeout: Optional[float]
    cache: Optional[bool]
    metadata: Optional[Dict[str, Any]]


class LLMCompleteParams(TypedDict, total=False):
    """Optional parameters for LLM completion calls."""

    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stream: Optional[bool]
    response_format: Optional[Dict[str, Any]]
    seed: Optional[int]
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]


class CriticFactoryParams(TypedDict, total=False):
    """Optional parameters for critic factory functions."""

    provider: Optional[str]  # LLM provider override
    api_key: Optional[str]  # API key override
    config: Optional[Any]  # Full configuration object


# Re-export for convenience
__all__ = [
    "ToolResultItem",
    "ToolUsageMetadata",
    "CriticMetadata",
    "MiddlewareContext",
    "CriticSettings",
    "ValidatorSettings",
    "StorageBackendSettings",
    "GenerationMetadata",
    "ValidationResult",
    "PerformanceMetrics",
    "ToolCallParams",
    "LLMCompleteParams",
    "CriticFactoryParams",
]
