"""
Legal domain validation configuration.

This module provides configuration classes for customizing legal content
validation rules, including citation formats and legal terminology requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sifaka.rules.base import RuleConfig


@dataclass
class LegalConfig(RuleConfig):
    """Configuration for legal content validation."""

    # Citation settings
    citation_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "case": r"\b\d+\s+[A-Z]\.[a-zA-Z0-9]+\s+\d+\b",  # e.g., "123 U.S. 456"
            "statute": r"\b\d+\s+U\.S\.C\.\s+§\s*\d+\b",  # e.g., "42 U.S.C. § 1983"
            "regulation": r"\b\d+\s+C\.F\.R\.\s+§\s*\d+\b",  # e.g., "17 C.F.R. § 240"
        }
    )
    min_citations: int = 0
    max_citations: int = 0  # 0 means no limit

    # Legal term requirements
    required_terms: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "required": {
                "jurisdiction",
                "pursuant to",
                "herein",
                "thereof",
            },
            "prohibited": {
                "guarantee",
                "promise",
                "ensure",
                "always",
            },
            "warning": {
                "shall",
                "must",
                "will",
                "may",
            },
        }
    )
    min_required_terms: int = 0
    max_prohibited_terms: int = 0  # 0 means any prohibited term fails validation

    # Disclaimer settings
    disclaimer_required: bool = True
    disclaimer_patterns: Set[str] = field(
        default_factory=lambda: {
            r"not\s+legal\s+advice",
            r"consult.*attorney",
            r"for\s+informational\s+purposes\s+only",
        }
    )

    # Performance settings
    cache_size: int = 1000
    priority: int = 2
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        super().__post_init__()
        if self.min_citations < 0:
            raise ValueError("min_citations must be non-negative")
        if self.max_citations < 0:
            raise ValueError("max_citations must be non-negative")
        if self.max_citations > 0 and self.min_citations > self.max_citations:
            raise ValueError("min_citations cannot be greater than max_citations")

        if self.min_required_terms < 0:
            raise ValueError("min_required_terms must be non-negative")
        if self.max_prohibited_terms < 0:
            raise ValueError("max_prohibited_terms must be non-negative")

        # For consistency, copy configuration values to params
        if not self.params:
            object.__setattr__(
                self,
                "params",
                {
                    "citation_patterns": self.citation_patterns,
                    "min_citations": self.min_citations,
                    "max_citations": self.max_citations,
                    "required_terms": self.required_terms,
                    "min_required_terms": self.min_required_terms,
                    "max_prohibited_terms": self.max_prohibited_terms,
                    "disclaimer_required": self.disclaimer_required,
                    "disclaimer_patterns": self.disclaimer_patterns,
                    "cache_size": self.cache_size,
                    "priority": self.priority,
                    "cost": self.cost,
                },
            )

    def add_citation_pattern(self, name: str, pattern: str) -> None:
        """Add a new citation pattern."""
        self.citation_patterns[name] = pattern
        if "citation_patterns" in self.params:
            self.params["citation_patterns"][name] = pattern

    def add_legal_terms(self, category: str, terms: Set[str]) -> None:
        """Add legal terms to a category."""
        if category not in self.required_terms:
            raise ValueError(f"Invalid category: {category}")
        self.required_terms[category].update(terms)
        if "required_terms" in self.params:
            self.params["required_terms"][category].update(terms)

    def add_disclaimer_pattern(self, pattern: str) -> None:
        """Add a new disclaimer pattern."""
        self.disclaimer_patterns.add(pattern)
        if "disclaimer_patterns" in self.params:
            self.params["disclaimer_patterns"].add(pattern)
