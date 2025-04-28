"""
Base classes and protocols for content validation.

This module provides the core abstractions for content validation rules,
including protocols for content analysis and validation.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.rules.base import BaseValidator, RuleConfig, RuleResult

# Type variables
T = TypeVar("T")
C = TypeVar("C", bound="ContentConfig")


@runtime_checkable
class ContentAnalyzer(Protocol[T]):
    """Protocol for content analysis."""

    @abstractmethod
    def analyze(self, content: str) -> T:
        """Analyze content and return results."""
        ...

    @abstractmethod
    def can_analyze(self, content: str) -> bool:
        """Check if content can be analyzed."""
        ...


class BaseContentValidator(BaseValidator[str]):
    """Base class for content validators."""

    def __init__(self, config: Optional[RuleConfig] = None) -> None:
        """Initialize validator with optional configuration."""
        super().__init__()
        self.config = config or RuleConfig()

    @property
    def validation_type(self) -> type:
        """Get the type this validator can validate."""
        return str

    def can_validate(self, content: str) -> bool:
        """Check if content can be validated."""
        return isinstance(content, str) and bool(content.strip())


class BaseContentAnalyzer(ContentAnalyzer[T]):
    """Base implementation of content analyzer."""

    def __init__(self, config: Optional[RuleConfig] = None) -> None:
        """Initialize analyzer with optional configuration."""
        self.config = config or RuleConfig()

    def can_analyze(self, content: str) -> bool:
        """Check if content can be analyzed."""
        return isinstance(content, str) and bool(content.strip())


@runtime_checkable
class ContentValidator(Protocol):
    """Protocol for content validation."""

    @abstractmethod
    def validate(self, content: str, **kwargs) -> RuleResult:
        """Validate content and return results."""
        ...

    @abstractmethod
    def get_validation_errors(self, content: str) -> List[str]:
        """Get list of validation errors for content."""
        ...

    @property
    @abstractmethod
    def config(self) -> RuleConfig:
        """Get validator configuration."""
        ...
