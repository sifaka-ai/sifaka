"""
Citation validation rules for Sifaka.

This module provides rules for validating citations in text, including:
- Citation pattern validation
- Citation count validation
- Citation format validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.factual.citation import create_citation_rule

    # Create a citation rule
    rule = create_citation_rule(
        citation_patterns=[
            r"\(\d{4}\)",  # (2024)
            r"\[.*?\]",    # [Author, 2024]
            r"\(.*?\)"     # (Author, 2024)
        ],
        min_citations=1,
        max_citations=5
    )
"""

from typing import Any, Dict, List, Optional
import re

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.factual.base import BaseFactualValidator


# Default citation patterns
DEFAULT_CITATION_PATTERNS: List[str] = [
    r"\(\d{4}\)",      # (2024)
    r"\[.*?\]",        # [Author, 2024]
    r"\(.*?\)",        # (Author, 2024)
    r"\d{4}",          # 2024
    r"\[.*?\]\(.*?\)", # [Author](URL)
]


class CitationConfig(BaseModel):
    """Configuration for citation validation."""

    model_config = ConfigDict(frozen=True)

    citation_patterns: List[str] = Field(
        default_factory=lambda: DEFAULT_CITATION_PATTERNS,
        description="List of regex patterns for citation validation",
    )
    min_citations: int = Field(
        default=1,
        ge=0,
        description="Minimum number of citations required",
    )
    max_citations: int = Field(
        default=5,
        ge=1,
        description="Maximum number of citations allowed",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )

    @field_validator("citation_patterns")
    @classmethod
    def validate_citation_patterns(cls, v: List[str]) -> List[str]:
        """Validate that citation patterns are not empty."""
        if not v:
            raise ValueError("Citation patterns cannot be empty")
        return v

    @field_validator("max_citations")
    @classmethod
    def validate_max_citations(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that max_citations is greater than or equal to min_citations."""
        if "min_citations" in values and v < values["min_citations"]:
            raise ValueError("max_citations must be greater than or equal to min_citations")
        return v


class DefaultCitationValidator(BaseFactualValidator):
    """Default validator for citation validation."""

    def __init__(self, config: CitationConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(config)
        self._patterns = [re.compile(fmt) for fmt in config.citation_patterns]
        self._min_citations = config.min_citations
        self._max_citations = config.max_citations

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for citations.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        # Find all citations
        citations = []
        for pattern in self._patterns:
            citations.extend(pattern.findall(text))

        # Count citations
        citation_count = len(citations)
        is_valid = self._min_citations <= citation_count <= self._max_citations

        return RuleResult(
            is_valid=is_valid,
            score=1.0 if is_valid else 0.0,
            message=f"Found {citation_count} citations (min: {self._min_citations}, max: {self._max_citations})",
        )


class CitationRule(Rule):
    """Rule for validating citations."""

    def __init__(self, config: CitationConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the rule
        """
        super().__init__(config)
        self._validator = DefaultCitationValidator(config)

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for citations.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        return self._validator.validate(text)


def create_citation_validator(
    citation_patterns: Optional[List[str]] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> DefaultCitationValidator:
    """Create a citation validator.

    Args:
        citation_patterns: List of regex patterns for citation validation
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultCitationValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if citation_patterns is not None:
        config_params["citation_patterns"] = citation_patterns
    if min_citations is not None:
        config_params["min_citations"] = min_citations
    if max_citations is not None:
        config_params["max_citations"] = max_citations

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = CitationConfig(**config_params)

    # Create validator
    return DefaultCitationValidator(config)


def create_citation_rule(
    name: str = "citation_rule",
    description: str = "Validates text for citations",
    citation_patterns: Optional[List[str]] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> CitationRule:
    """Create a citation rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        citation_patterns: List of regex patterns for citation validation
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the rule

    Returns:
        CitationRule: The created rule
    """
    # Create config dictionary
    config_dict = {
        "citation_patterns": citation_patterns or DEFAULT_CITATION_PATTERNS,
        "min_citations": min_citations or 1,
        "max_citations": max_citations or 5,
        **kwargs,
    }

    # Create config
    config = CitationConfig(**config_dict)

    # Create rule
    return CitationRule(config)
