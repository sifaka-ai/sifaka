"""
Prohibited content validation.

This module provides validation rules for detecting and filtering prohibited content,
with support for customizable term lists and case sensitivity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sifaka.rules.base import RuleConfig, RuleResult
from .base import BaseContentValidator
from .analyzers import ProhibitedContentAnalyzer


@dataclass
class ProhibitedContentConfig(RuleConfig):
    """Configuration for prohibited content validation."""

    # Core settings
    terms: Set[str] = field(default_factory=set)
    case_sensitive: bool = False
    max_matches: int = 0  # 0 means any match fails validation

    # Performance settings
    cache_size: int = 1000
    priority: int = 2
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        super().__post_init__()
        if self.max_matches < 0:
            raise ValueError("max_matches must be non-negative")

        # For consistency, copy configuration values to params
        if not self.params:
            object.__setattr__(
                self,
                "params",
                {
                    "prohibited_terms": self.terms,
                    "case_sensitive": self.case_sensitive,
                    "max_matches": self.max_matches,
                    "cache_size": self.cache_size,
                    "priority": self.priority,
                    "cost": self.cost,
                },
            )

    def add_terms(self, terms: Set[str]) -> None:
        """Add prohibited terms."""
        self.terms.update(terms)
        if "prohibited_terms" in self.params:
            self.params["prohibited_terms"].update(terms)

    def remove_terms(self, terms: Set[str]) -> None:
        """Remove prohibited terms."""
        self.terms.difference_update(terms)
        if "prohibited_terms" in self.params:
            self.params["prohibited_terms"].difference_update(terms)


class ProhibitedContentValidator(BaseContentValidator):
    """Validator for prohibited content."""

    def __init__(self, config: Optional[ProhibitedContentConfig] = None) -> None:
        """Initialize validator with configuration."""
        super().__init__(config or ProhibitedContentConfig())
        self._analyzer = ProhibitedContentAnalyzer(config=self.config)

    def validate(self, content: str, **kwargs) -> RuleResult:
        """Validate content for prohibited terms."""
        if not self.can_validate(content):
            return RuleResult(
                passed=False,
                message="Invalid content",
                metadata={"error": "Content must be a non-empty string"},
            )

        analysis = self._analyzer.analyze(content)
        found_terms = analysis["found_terms"]
        total_matches = analysis["total_matches"]

        # Check if we found any prohibited terms
        if not found_terms:
            return RuleResult(
                passed=True,
                message="No prohibited terms found",
                metadata={"analysis": analysis},
            )

        # Check against max_matches if configured
        max_matches = self.config.params.get("max_matches", 0)
        if max_matches > 0 and total_matches <= max_matches:
            return RuleResult(
                passed=True,
                message=f"Found {total_matches} prohibited terms (within limit of {max_matches})",
                metadata={"analysis": analysis},
            )

        return RuleResult(
            passed=False,
            message=f"Found {total_matches} prohibited terms",
            metadata={"analysis": analysis},
        )

    def get_validation_errors(self, content: str) -> List[str]:
        """Get list of validation errors."""
        if not self.can_validate(content):
            return ["Content must be a non-empty string"]

        analysis = self._analyzer.analyze(content)
        found_terms = analysis["found_terms"]

        if not found_terms:
            return []

        max_matches = self.config.params.get("max_matches", 0)
        if max_matches > 0 and len(found_terms) <= max_matches:
            return []

        return [f"Found prohibited terms: {', '.join(found_terms)}"]
