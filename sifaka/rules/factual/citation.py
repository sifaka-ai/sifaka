"""
Citation validation rules for Sifaka.

This module provides validators and rules for checking citations in text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The CitationConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_citation_rule, create_citation_validator) handle configuration

Usage Example:
    from sifaka.rules.factual.citation import create_citation_rule

    # Create a citation rule using the factory function
    rule = create_citation_rule(
        citation_patterns=[
            r"\[[\d]+\]",  # [1], [2], etc.
            r"\([A-Za-z]+, \d{4}\)",  # (Smith, 2020)
        ],
        required_citations=True
    )

    # Validate text
    result = rule.validate("According to Smith (2020), this approach is effective.")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = CitationRule(
        config=RuleConfig(
            params={
                "citation_patterns": [
                    r"\[[\d]+\]",  # [1], [2], etc.
                    r"\([A-Za-z]+, \d{4}\)",  # (Smith, 2020)
                ],
                "required_citations": True
            }
        )
    )
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


__all__ = [
    # Config classes
    "CitationConfig",
    # Validator classes
    "DefaultCitationValidator",
    # Rule classes
    "CitationRule",
    # Factory functions
    "create_citation_validator",
    "create_citation_rule",
]


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
        """Validate configuration."""
        super().__post_init__()
        if not self.citation_patterns:
            raise ValueError("Must provide at least one citation pattern")


class DefaultCitationValidator(BaseFactualValidator):
    """Default implementation of citation validation."""

    def __init__(self, config: CitationConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self._patterns = [re.compile(pattern) for pattern in config.citation_patterns]

    @property
    def config(self) -> CitationConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for citations."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        found_citations = []
        for pattern in self._patterns:
            found_citations.extend(pattern.findall(text))

        if not found_citations and self.config.required_citations:
            return RuleResult(
                passed=False,
                message="No citations found",
                metadata={
                    "found_citations": [],
                    "required": self.config.required_citations,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Found {len(found_citations)} citation(s)",
            metadata={
                "found_citations": found_citations,
                "required": self.config.required_citations,
            },
        )


class CitationRule(Rule[str, RuleResult, DefaultCitationValidator, Any]):
    """Rule that checks for citations in the text."""

    def __init__(
        self,
        name: str = "citation_rule",
        description: str = "Checks for citations",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultCitationValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the citation rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultCitationValidator:
        """Create a default validator from config."""
        citation_config = CitationConfig(**self._rule_params)
        return DefaultCitationValidator(citation_config)


def create_citation_validator(
    citation_patterns: Optional[List[str]] = None,
    required_citations: Optional[bool] = None,
    **kwargs,
) -> DefaultCitationValidator:
    """
    Create a citation validator with the specified configuration.

    This factory function creates a configured citation validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        citation_patterns: List of regex patterns for detecting citations
        required_citations: Whether citations are required in the text
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured citation validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if citation_patterns is not None:
        config_params["citation_patterns"] = citation_patterns
    if required_citations is not None:
        config_params["required_citations"] = required_citations

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = CitationConfig(**config_params)

    # Return configured validator
    return DefaultCitationValidator(config)


def create_citation_rule(
    name: str = "citation_rule",
    description: str = "Validates text for citations",
    citation_patterns: Optional[List[str]] = None,
    required_citations: Optional[bool] = None,
    **kwargs,
) -> CitationRule:
    """
    Create a citation rule with configuration.

    This factory function creates a configured CitationRule instance.
    It uses create_citation_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        citation_patterns: List of regex patterns for detecting citations
        required_citations: Whether citations are required in the text
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured CitationRule instance
    """
    # Create validator using the validator factory
    validator = create_citation_validator(
        citation_patterns=citation_patterns,
        required_citations=required_citations,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return CitationRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
