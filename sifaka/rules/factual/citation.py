"""
Citation validation rules for Sifaka.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


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
    ) -> None:
        """
        Initialize the citation rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultCitationValidator:
        """Create a default validator from config."""
        citation_config = CitationConfig(**self._rule_params)
        return DefaultCitationValidator(citation_config)


def create_citation_rule(
    name: str = "citation_rule",
    description: str = "Validates text for citations",
    config: Optional[Dict[str, Any]] = None,
) -> CitationRule:
    """
    Create a citation rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured CitationRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return CitationRule(
        name=name,
        description=description,
        config=rule_config,
    )
