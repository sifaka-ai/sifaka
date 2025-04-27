"""
Fact-checking rules for Sifaka.
"""

from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final, TypeVar
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator
import re


@dataclass(frozen=True)
class FactualConsistencyConfig(RuleConfig):
    """Configuration for factual consistency rules."""

    contradiction_indicators: List[str] = field(
        default_factory=lambda: [
            "but",
            "however",
            "although",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "despite",
            "yet",
            "while",
            "whereas",
        ]
    )
    confidence_threshold: float = 0.7
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class ConfidenceConfig(RuleConfig):
    """Configuration for confidence rules."""

    confidence_indicators: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "high": ["definitely", "certainly", "always", "never", "must", "will"],
            "medium": ["likely", "probably", "usually", "often", "generally"],
            "low": ["maybe", "possibly", "sometimes", "occasionally", "might"],
            "uncertain": ["perhaps", "could", "may", "seems", "appears"],
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not all(
            isinstance(level, str) and isinstance(indicators, list)
            for level, indicators in self.confidence_indicators.items()
        ):
            raise ValueError("confidence_indicators must be a Dict[str, List[str]]")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class CitationConfig(RuleConfig):
    """Configuration for citation rules."""

    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\[[\d]+\]",  # [1], [2], etc.
            r"\([A-Za-z]+ et al., \d{4}\)",  # (Smith et al., 2020)
            r"\([A-Za-z]+, \d{4}\)",  # (Smith, 2020)
            r"https?://[^\s]+",  # URLs
        ]
    )
    required_citations: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not all(isinstance(pattern, str) for pattern in self.citation_patterns):
            raise ValueError("citation_patterns must be a List[str]")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class FactualAccuracyConfig(RuleConfig):
    """Configuration for factual accuracy rules."""

    knowledge_base: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "earth_shape": {"round", "spherical", "geoid"},
            "gravity": {"9.8 m/s²", "9.8 meters per second squared"},
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not all(
            isinstance(key, str) and isinstance(facts, set)
            for key, facts in self.knowledge_base.items()
        ):
            raise ValueError("knowledge_base must be a Dict[str, Set[str]]")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@runtime_checkable
class FactualConsistencyValidator(Protocol):
    """Protocol for factual consistency validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> FactualConsistencyConfig: ...


@runtime_checkable
class ConfidenceValidator(Protocol):
    """Protocol for confidence validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> ConfidenceConfig: ...


@runtime_checkable
class CitationValidator(Protocol):
    """Protocol for citation validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> CitationConfig: ...


@runtime_checkable
class FactualAccuracyValidator(Protocol):
    """Protocol for factual accuracy validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> FactualAccuracyConfig: ...


class FactualConsistencyRule(Rule):
    """Rule that checks for factual consistency within the text."""

    def __init__(
        self,
        name: str,
        description: str,
        config: FactualConsistencyConfig,
        validator: FactualConsistencyValidator,
    ) -> None:
        """Initialize the factual consistency rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> FactualConsistencyConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> FactualConsistencyValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text maintains factual consistency.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


class ConfidenceRule(Rule):
    """Rule that checks for appropriate confidence levels in statements."""

    def __init__(
        self,
        name: str,
        description: str,
        config: ConfidenceConfig,
        validator: ConfidenceValidator,
    ) -> None:
        """Initialize the confidence rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> ConfidenceConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> ConfidenceValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text uses appropriate confidence levels.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


class CitationRule(Rule):
    """Rule that checks for proper citations and references."""

    def __init__(
        self,
        name: str,
        description: str,
        config: CitationConfig,
        validator: CitationValidator,
    ) -> None:
        """Initialize the citation rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> CitationConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> CitationValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text contains proper citations.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)


class FactualAccuracyRule(Rule):
    """Rule that checks for factual accuracy using a knowledge base."""

    def __init__(
        self,
        name: str,
        description: str,
        config: FactualAccuracyConfig,
        validator: FactualAccuracyValidator,
    ) -> None:
        """Initialize the factual accuracy rule."""
        super().__init__(name=name, description=description)
        self._config = config
        self._validator = validator

    @property
    def config(self) -> FactualAccuracyConfig:
        """Get the rule configuration."""
        return self._config

    @property
    def validator(self) -> FactualAccuracyValidator:
        """Get the rule validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text contains accurate facts.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        return self._validator.validate(text)
