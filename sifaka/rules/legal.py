"""
Legal-specific rules for Sifaka.

This module provides rules for validating legal content, including citation formats,
legal terms, and other legal-specific validation requirements.
"""

from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final
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


class DefaultLegalCitationValidator:
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
                message=f"All {total_citations} citations are valid",
                metadata={
                    "found_citations": found_citations,
                    "total_citations": total_citations,
                },
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during citation validation: {str(e)}",
                metadata={"error": str(e)},
            )


class DefaultLegalTermsValidator:
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
            text_to_check = text if self.config.case_sensitive else text.lower()
            terms_to_check = lambda t: t if self.config.case_sensitive else t.lower()

            # Check for required terms
            missing_terms = [
                term
                for term in self.config.required_terms
                if terms_to_check(term) not in text_to_check
            ]
            if missing_terms:
                return RuleResult(
                    passed=False,
                    message=f"Missing required legal terms: {', '.join(missing_terms)}",
                    metadata={
                        "missing_terms": missing_terms,
                        "requirement": "required",
                    },
                )

            # Check for prohibited terms
            found_prohibited = [
                term
                for term in self.config.prohibited_terms
                if terms_to_check(term) in text_to_check
            ]
            if found_prohibited:
                return RuleResult(
                    passed=False,
                    message=f"Found prohibited legal terms: {', '.join(found_prohibited)}",
                    metadata={
                        "found_prohibited": found_prohibited,
                        "requirement": "prohibited",
                    },
                )

            # Check for legal terms
            found_legal_terms = [
                term for term in self.config.legal_terms if terms_to_check(term) in text_to_check
            ]

            # Check for warning terms
            found_warning_terms = [
                term for term in self.config.warning_terms if terms_to_check(term) in text_to_check
            ]

            return RuleResult(
                passed=True,
                message="Legal terms validation passed",
                metadata={
                    "found_legal_terms": found_legal_terms,
                    "found_warning_terms": found_warning_terms,
                },
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during legal terms validation: {str(e)}",
                metadata={"error": str(e)},
            )


class LegalCitationRule(Rule):
    """Rule that validates legal citations in the output."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: LegalCitationValidator,
    ) -> None:
        """Initialize the legal citation rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> LegalCitationValidator:
        """Get the citation validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate legal citations."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during citation validation: {str(e)}",
                metadata={"error": str(e)},
            )


class LegalTermsRule(Rule):
    """Rule that validates legal terms in the output."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: LegalTermsValidator,
    ) -> None:
        """Initialize the legal terms rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> LegalTermsValidator:
        """Get the terms validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate legal terms."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during legal terms validation: {str(e)}",
                metadata={"error": str(e)},
            )


def create_legal_citation_rule(
    name: str,
    description: str,
    config: LegalCitationConfig | None = None,
) -> LegalCitationRule:
    """Create a legal citation rule with default configuration."""
    validator = DefaultLegalCitationValidator(config or LegalCitationConfig())
    return LegalCitationRule(name=name, description=description, validator=validator)


def create_legal_terms_rule(
    name: str,
    description: str,
    config: LegalTermsConfig | None = None,
) -> LegalTermsRule:
    """Create a legal terms rule with default configuration."""
    validator = DefaultLegalTermsValidator(config or LegalTermsConfig())
    return LegalTermsRule(name=name, description=description, validator=validator)
