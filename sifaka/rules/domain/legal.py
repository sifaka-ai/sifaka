"""
Legal domain-specific validation rules for Sifaka.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator


@dataclass(frozen=True)
class LegalConfig(RuleConfig):
    """Configuration for legal rules."""

    legal_terms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "jurisdiction": ["jurisdiction", "court", "venue", "forum", "tribunal"],
            "statute": ["statute", "law", "regulation", "code", "act", "bill", "ordinance"],
            "precedent": ["precedent", "case law", "ruling", "decision", "holding", "opinion"],
            "liability": ["liability", "responsibility", "duty", "obligation", "negligence"],
            "procedure": ["procedure", "motion", "pleading", "filing", "petition", "appeal"],
            "evidence": ["evidence", "proof", "exhibit", "testimony", "witness", "document"],
        }
    )
    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
            r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
            r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
            r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
            r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
            r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
            r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
        ]
    )
    disclaimers: List[str] = field(
        default_factory=lambda: [
            r"(?i)not\s+(?:intended\s+as\s+)?legal\s+advice",
            r"(?i)consult\s+(?:(?:a|your)\s+)?(?:qualified\s+)?(?:attorney|lawyer|legal\s+counsel)",
            r"(?i)seek\s+legal\s+(?:counsel|advice|representation)",
            r"(?i)legal\s+disclaimer\s*[:\-]?",
            r"(?i)for\s+informational\s+purposes\s+only",
            r"(?i)does\s+not\s+constitute\s+(?:a|an)\s+attorney-client\s+relationship",
            r"(?i)not\s+a\s+substitute\s+for\s+legal\s+(?:counsel|advice)",
        ]
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.legal_terms:
            raise ValueError("Must provide at least one legal term category")
        if not self.citation_patterns:
            raise ValueError("Must provide at least one citation pattern")
        if not self.disclaimers:
            raise ValueError("Must provide at least one disclaimer pattern")


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
        super().__post_init__()
        if not isinstance(self.citation_patterns, list):
            raise ValueError("citation_patterns must be a list")
        if not all(isinstance(p, str) for p in self.citation_patterns):
            raise ValueError("citation_patterns must contain only strings")
        if self.min_citations < 0:
            raise ValueError("min_citations must be non-negative")
        if self.max_citations < self.min_citations:
            raise ValueError("max_citations must be greater than or equal to min_citations")


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
        super().__post_init__()
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


@runtime_checkable
class LegalValidator(Protocol):
    """Protocol for legal content validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LegalConfig: ...


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


class DefaultLegalValidator(BaseDomainValidator):
    """Default implementation of legal content validation."""

    def __init__(self, config: LegalConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self._citation_patterns = [re.compile(pattern) for pattern in config.citation_patterns]
        self._disclaimer_patterns = [re.compile(pattern) for pattern in config.disclaimers]

    @property
    def config(self) -> LegalConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate legal content."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            # Check for disclaimer if required
            has_disclaimer = any(pattern.search(text) for pattern in self._disclaimer_patterns)
            if self.config.disclaimer_required and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="No legal disclaimer found when required",
                    metadata={"found_disclaimer": False, "requirement": "disclaimer"},
                )

            # Check for legal terms
            legal_term_counts = {}
            for category, terms in self.config.legal_terms.items():
                count = 0
                for term in terms:
                    # Case-insensitive search
                    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
                    count += len(pattern.findall(text))
                legal_term_counts[category] = count

            return RuleResult(
                passed=True,
                message="Legal content validation passed",
                metadata={
                    "legal_term_counts": legal_term_counts,
                    "has_disclaimer": has_disclaimer,
                    "disclaimer_required": self.config.disclaimer_required,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error validating legal content: {str(e)}",
                metadata={"error": str(e)},
            )


class DefaultLegalCitationValidator(BaseDomainValidator):
    """Default implementation of legal citation validation."""

    def __init__(self, config: LegalCitationConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self._compiled_patterns = [re.compile(pattern) for pattern in config.citation_patterns]

    @property
    def config(self) -> LegalCitationConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
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

            # All checks passed
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


class DefaultLegalTermsValidator(BaseDomainValidator):
    """Default implementation of legal terms validation."""

    def __init__(self, config: LegalTermsConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> LegalTermsConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate legal terms."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            # Check for legal terms
            flags = 0 if self.config.case_sensitive else re.IGNORECASE
            legal_terms_found = set()
            for term in self.config.legal_terms:
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", flags)
                if pattern.search(text):
                    legal_terms_found.add(term)

            # Check for warning terms
            warning_terms_found = set()
            for term in self.config.warning_terms:
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", flags)
                if pattern.search(text):
                    warning_terms_found.add(term)

            # Check for required terms
            missing_required_terms = set()
            for term in self.config.required_terms:
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", flags)
                if not pattern.search(text):
                    missing_required_terms.add(term)

            # Check for prohibited terms
            prohibited_terms_found = set()
            for term in self.config.prohibited_terms:
                pattern = re.compile(r"\b" + re.escape(term) + r"\b", flags)
                if pattern.search(text):
                    prohibited_terms_found.add(term)

            # Determine if validation passes
            passed = True
            message = "Legal terms validation passed"

            if missing_required_terms:
                passed = False
                message = f"Missing required legal terms: {', '.join(missing_required_terms)}"

            if prohibited_terms_found:
                passed = False
                message = f"Found prohibited legal terms: {', '.join(prohibited_terms_found)}"

            return RuleResult(
                passed=passed,
                message=message,
                metadata={
                    "legal_terms_found": list(legal_terms_found),
                    "warning_terms_found": list(warning_terms_found),
                    "missing_required_terms": list(missing_required_terms),
                    "prohibited_terms_found": list(prohibited_terms_found),
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error validating legal terms: {str(e)}",
                metadata={"error": str(e)},
            )


class LegalRule(Rule[str, RuleResult, DefaultLegalValidator, Any]):
    """Rule that validates legal content."""

    def __init__(
        self,
        name: str = "legal_rule",
        description: str = "Validates legal content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalValidator] = None,
    ) -> None:
        """
        Initialize the legal rule.

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

    def _create_default_validator(self) -> DefaultLegalValidator:
        """Create a default validator from config."""
        legal_config = LegalConfig(**self._rule_params)
        return DefaultLegalValidator(legal_config)


class LegalCitationRule(Rule[str, RuleResult, DefaultLegalCitationValidator, Any]):
    """Rule that checks for legal citations."""

    def __init__(
        self,
        name: str = "legal_citation_rule",
        description: str = "Checks for legal citations",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalCitationValidator] = None,
    ) -> None:
        """
        Initialize the legal citation rule.

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

    def _create_default_validator(self) -> DefaultLegalCitationValidator:
        """Create a default validator from config."""
        citation_config = LegalCitationConfig(**self._rule_params)
        return DefaultLegalCitationValidator(citation_config)


class LegalTermsRule(Rule[str, RuleResult, DefaultLegalTermsValidator, Any]):
    """Rule that validates legal terminology."""

    def __init__(
        self,
        name: str = "legal_terms_rule",
        description: str = "Validates legal terminology",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultLegalTermsValidator] = None,
    ) -> None:
        """
        Initialize the legal terms rule.

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

    def _create_default_validator(self) -> DefaultLegalTermsValidator:
        """Create a default validator from config."""
        terms_config = LegalTermsConfig(**self._rule_params)
        return DefaultLegalTermsValidator(terms_config)


def create_legal_rule(
    name: str = "legal_rule",
    description: str = "Validates text for legal content",
    config: Optional[Dict[str, Any]] = None,
) -> LegalRule:
    """
    Create a legal rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured LegalRule instance
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

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return LegalRule(
        name=name,
        description=description,
        config=rule_config,
    )


def create_legal_citation_rule(
    name: str = "legal_citation_rule",
    description: str = "Validates legal citations",
    config: Optional[Dict[str, Any]] = None,
) -> LegalCitationRule:
    """
    Create a legal citation validation rule.

    Args:
        name: Name of the rule
        description: Description of the rule
        config: Optional configuration dictionary with:
            - citation_patterns: List of regex patterns for legal citations
            - require_citations: Whether citations are required
            - min_citations: Minimum number of citations required
            - max_citations: Maximum number of citations allowed
            - cache_size: Size of internal cache
            - priority: Priority of the rule
            - cost: Computational cost factor

    Returns:
        A configured LegalCitationRule
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return LegalCitationRule(
        name=name,
        description=description,
        config=rule_config,
    )


def create_legal_terms_rule(
    name: str = "legal_terms_rule",
    description: str = "Validates legal terms",
    config: Optional[Dict[str, Any]] = None,
) -> LegalTermsRule:
    """
    Create a legal terms validation rule.

    Args:
        name: Name of the rule
        description: Description of the rule
        config: Optional configuration dictionary with:
            - legal_terms: Set of legal terms to check for
            - warning_terms: Set of warning terms to check for
            - required_terms: Set of terms that must be present
            - prohibited_terms: Set of terms that must not be present
            - case_sensitive: Whether to make checks case-sensitive
            - cache_size: Size of internal cache
            - priority: Priority of the rule
            - cost: Computational cost factor

    Returns:
        A configured LegalTermsRule
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return LegalTermsRule(
        name=name,
        description=description,
        config=rule_config,
    )
