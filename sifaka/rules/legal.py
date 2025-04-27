"""
Legal-specific rules for Sifaka.

This module provides rules for validating legal content, including citation formats,
legal terms, and other legal-specific validation requirements.
"""

from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final, Optional
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
import re
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator


@dataclass(frozen=True)
class LegalCitationConfig(RuleConfig):
    """Configuration for legal citation validation."""

    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
            r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
            r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
            r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
        ]
    )
    require_citations: bool = True
    min_citations: int = 0
    max_citations: int = 100
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.citation_patterns, list):
            raise ValueError("citation_patterns must be a list")
        if not all(isinstance(p, str) for p in self.citation_patterns):
            raise ValueError("citation_patterns must contain only strings")
        if self.min_citations < 0:
            raise ValueError("min_citations must be non-negative")
        if self.max_citations < self.min_citations:
            raise ValueError("max_citations must be greater than or equal to min_citations")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class LegalTermsConfig(RuleConfig):
    """Configuration for legal terms validation."""

    legal_terms: Set[str] = field(
        default_factory=lambda: {
            "confidential",
            "proprietary",
            "classified",
            "restricted",
            "private",
            "sensitive",
        }
    )
    warning_terms: Set[str] = field(
        default_factory=lambda: {
            "warning",
            "caution",
            "notice",
            "disclaimer",
            "privileged",
        }
    )
    required_terms: Set[str] = field(default_factory=set)
    prohibited_terms: Set[str] = field(default_factory=set)
    case_sensitive: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.legal_terms, set):
            raise ValueError("legal_terms must be a set")
        if not all(isinstance(t, str) for t in self.legal_terms):
            raise ValueError("legal_terms must contain only strings")
        if not isinstance(self.warning_terms, set):
            raise ValueError("warning_terms must be a set")
        if not all(isinstance(t, str) for t in self.warning_terms):
            raise ValueError("warning_terms must contain only strings")
        if not isinstance(self.required_terms, set):
            raise ValueError("required_terms must be a set")
        if not all(isinstance(t, str) for t in self.required_terms):
            raise ValueError("required_terms must contain only strings")
        if not isinstance(self.prohibited_terms, set):
            raise ValueError("prohibited_terms must be a set")
        if not all(isinstance(t, str) for t in self.prohibited_terms):
            raise ValueError("prohibited_terms must contain only strings")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@runtime_checkable
class LegalCitationValidator(Protocol):
    """Protocol for legal citation validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalCitationConfig: ...


@runtime_checkable
class LegalTermsValidator(Protocol):
    """Protocol for legal terms validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalTermsConfig: ...


class DefaultLegalCitationValidator(RuleValidator[str]):
    """Default implementation of legal citation validation."""

    def __init__(self, config: LegalCitationConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._compiled_patterns = [re.compile(pattern) for pattern in config.citation_patterns]

    @property
    def config(self) -> LegalCitationConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate legal citations."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            # Find all citations
            found_citations = []
            for pattern in self._compiled_patterns:
                found_citations.extend(pattern.findall(text))

            # Check if citations are properly formatted
            invalid_citations = []
            for citation in found_citations:
                if not any(pattern.match(citation) for pattern in self._compiled_patterns):
                    invalid_citations.append(citation)

            # Check citation count requirements
            total_citations = len(found_citations)
            if self.config.require_citations and total_citations == 0:
                return RuleResult(
                    passed=False,
                    message="No citations found when citations are required",
                    metadata={
                        "found_citations": found_citations,
                        "total_citations": total_citations,
                        "requirement": "required",
                    },
                )

            if total_citations < self.config.min_citations:
                return RuleResult(
                    passed=False,
                    message=f"Found {total_citations} citations, minimum required is {self.config.min_citations}",
                    metadata={
                        "found_citations": found_citations,
                        "total_citations": total_citations,
                        "requirement": "minimum",
                    },
                )

            if total_citations > self.config.max_citations:
                return RuleResult(
                    passed=False,
                    message=f"Found {total_citations} citations, maximum allowed is {self.config.max_citations}",
                    metadata={
                        "found_citations": found_citations,
                        "total_citations": total_citations,
                        "requirement": "maximum",
                    },
                )

            if invalid_citations:
                return RuleResult(
                    passed=False,
                    message=f"Found {len(invalid_citations)} invalid citations",
                    metadata={
                        "found_citations": found_citations,
                        "invalid_citations": invalid_citations,
                        "total_citations": total_citations,
                    },
                )

            return RuleResult(
                passed=True,
                message=f"Found {total_citations} valid citations",
                metadata={
                    "found_citations": found_citations,
                    "total_citations": total_citations,
                },
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error validating citations: {str(e)}",
                metadata={"error": str(e)},
            )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultLegalTermsValidator(RuleValidator[str]):
    """Default implementation of legal terms validation."""

    def __init__(self, config: LegalTermsConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> LegalTermsConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate legal terms."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            # Process text based on case sensitivity
            if not self.config.case_sensitive:
                text = text.lower()
                legal_terms = {t.lower() for t in self.config.legal_terms}
                warning_terms = {t.lower() for t in self.config.warning_terms}
                required_terms = {t.lower() for t in self.config.required_terms}
                prohibited_terms = {t.lower() for t in self.config.prohibited_terms}
            else:
                legal_terms = self.config.legal_terms
                warning_terms = self.config.warning_terms
                required_terms = self.config.required_terms
                prohibited_terms = self.config.prohibited_terms

            # Find terms
            found_legal_terms = {term for term in legal_terms if term in text}
            found_warning_terms = {term for term in warning_terms if term in text}
            found_required_terms = {term for term in required_terms if term in text}
            found_prohibited_terms = {term for term in prohibited_terms if term in text}

            # Check requirements
            missing_required = required_terms - found_required_terms
            if missing_required:
                return RuleResult(
                    passed=False,
                    message=f"Missing required terms: {', '.join(missing_required)}",
                    metadata={
                        "missing_required": list(missing_required),
                        "found_legal": list(found_legal_terms),
                        "found_warning": list(found_warning_terms),
                        "found_prohibited": list(found_prohibited_terms),
                    },
                )

            if found_prohibited_terms:
                return RuleResult(
                    passed=False,
                    message=f"Found prohibited terms: {', '.join(found_prohibited_terms)}",
                    metadata={
                        "found_prohibited": list(found_prohibited_terms),
                        "found_legal": list(found_legal_terms),
                        "found_warning": list(found_warning_terms),
                    },
                )

            return RuleResult(
                passed=True,
                message="All legal terms requirements met",
                metadata={
                    "found_legal": list(found_legal_terms),
                    "found_warning": list(found_warning_terms),
                },
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error validating legal terms: {str(e)}",
                metadata={"error": str(e)},
            )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class LegalCitationRule(Rule):
    """Rule for validating legal citations."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with legal citation validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        citation_config = LegalCitationConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultLegalCitationValidator(citation_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output citations."""
        return self._validator.validate(output)


class LegalTermsRule(Rule):
    """Rule for validating legal terms."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with legal terms validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        terms_config = LegalTermsConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultLegalTermsValidator(terms_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output legal terms."""
        return self._validator.validate(output)


def create_legal_citation_rule(
    name: str = "legal_citation_rule",
    description: str = "Validates legal citations",
    config: Optional[Dict[str, Any]] = None,
) -> LegalCitationRule:
    """
    Create a legal citation rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured LegalCitationRule instance
    """
    if config is None:
        config = {
            "citation_patterns": [
                r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
                r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
                r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
                r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
            ],
            "require_citations": True,
            "min_citations": 0,
            "max_citations": 100,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return LegalCitationRule(
        name=name,
        description=description,
        config=config,
    )


def create_legal_terms_rule(
    name: str = "legal_terms_rule",
    description: str = "Validates legal terms",
    config: Optional[Dict[str, Any]] = None,
) -> LegalTermsRule:
    """
    Create a legal terms rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured LegalTermsRule instance
    """
    if config is None:
        config = {
            "legal_terms": {
                "confidential",
                "proprietary",
                "classified",
                "restricted",
                "private",
                "sensitive",
            },
            "warning_terms": {
                "warning",
                "caution",
                "notice",
                "disclaimer",
                "privileged",
            },
            "required_terms": set(),
            "prohibited_terms": set(),
            "case_sensitive": False,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return LegalTermsRule(
        name=name,
        description=description,
        config=config,
    )


# Export public classes and functions
__all__ = [
    "LegalCitationRule",
    "LegalCitationConfig",
    "LegalCitationValidator",
    "DefaultLegalCitationValidator",
    "LegalTermsRule",
    "LegalTermsConfig",
    "LegalTermsValidator",
    "DefaultLegalTermsValidator",
    "create_legal_citation_rule",
    "create_legal_terms_rule",
]
