"""
Legal domain-specific validation rules for Sifaka.

This module provides validators and rules for checking legal content, citations, and terminology.

Usage Example:
    from sifaka.rules.domain.legal import create_legal_rule, create_legal_citation_rule, create_legal_terms_rule

    # Create a legal rule
    legal_rule = create_legal_rule(
        disclaimer_required=True,
        legal_terms={
            "jurisdiction": ["court", "venue", "forum"],
            "liability": ["liability", "responsibility", "duty"]
        }
    )

    # Create a legal citation rule
    citation_rule = create_legal_citation_rule(
        citation_patterns=[r"\d+\s+U\.S\.\s+\d+", r"\d+\s+S\.\s*Ct\.\s+\d+"],
        require_citations=True,
        min_citations=1
    )

    # Create a legal terms rule
    terms_rule = create_legal_terms_rule(
        required_terms={"disclaimer", "notice"},
        prohibited_terms={"guarantee", "warranty"}
    )

    # Validate text
    result = legal_rule.validate("This legal document is subject to the jurisdiction of the court.")
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator


__all__ = [
    # Config classes
    "LegalConfig",
    "LegalCitationConfig",
    "LegalTermsConfig",
    # Protocol classes
    "LegalValidator",
    "LegalCitationValidator",
    "LegalTermsValidator",
    # Validator classes
    "DefaultLegalValidator",
    "DefaultLegalCitationValidator",
    "DefaultLegalTermsValidator",
    # Rule classes
    "LegalRule",
    "LegalCitationRule",
    "LegalTermsRule",
    # Factory functions
    "create_legal_validator",
    "create_legal_rule",
    "create_legal_citation_validator",
    "create_legal_citation_rule",
    "create_legal_terms_validator",
    "create_legal_terms_rule",
]


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
        **kwargs,
    ) -> None:
        """
        Initialize the legal rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

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
        **kwargs,
    ) -> None:
        """
        Initialize the legal citation rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

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
        **kwargs,
    ) -> None:
        """
        Initialize the legal terms rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultLegalTermsValidator:
        """Create a default validator from config."""
        terms_config = LegalTermsConfig(**self._rule_params)
        return DefaultLegalTermsValidator(terms_config)


def create_legal_validator(
    legal_terms: Optional[Dict[str, List[str]]] = None,
    citation_patterns: Optional[List[str]] = None,
    disclaimers: Optional[List[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> DefaultLegalValidator:
    """
    Create a legal validator with the specified configuration.

    This factory function creates a configured legal validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        legal_terms: Dictionary mapping categories to lists of legal terms
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of regex patterns for legal disclaimers
        disclaimer_required: Whether a disclaimer is required
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if legal_terms is not None:
        config_params["legal_terms"] = legal_terms
    if citation_patterns is not None:
        config_params["citation_patterns"] = citation_patterns
    if disclaimers is not None:
        config_params["disclaimers"] = disclaimers
    if disclaimer_required is not None:
        config_params["disclaimer_required"] = disclaimer_required

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalConfig(**config_params)

    # Return configured validator
    return DefaultLegalValidator(config)


def create_legal_rule(
    name: str = "legal_rule",
    description: str = "Validates text for legal content",
    legal_terms: Optional[Dict[str, List[str]]] = None,
    citation_patterns: Optional[List[str]] = None,
    disclaimers: Optional[List[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> LegalRule:
    """
    Create a legal rule with configuration.

    This factory function creates a configured LegalRule instance.
    It uses create_legal_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        legal_terms: Dictionary mapping categories to lists of legal terms
        citation_patterns: List of regex patterns for legal citations
        disclaimers: List of regex patterns for legal disclaimers
        disclaimer_required: Whether a disclaimer is required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured LegalRule instance
    """
    # Create validator using the validator factory
    validator = create_legal_validator(
        legal_terms=legal_terms,
        citation_patterns=citation_patterns,
        disclaimers=disclaimers,
        disclaimer_required=disclaimer_required,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_legal_citation_validator(
    citation_patterns: Optional[List[str]] = None,
    require_citations: Optional[bool] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> DefaultLegalCitationValidator:
    """
    Create a legal citation validator with the specified configuration.

    This factory function creates a configured legal citation validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        citation_patterns: List of regex patterns for legal citations
        require_citations: Whether citations are required
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal citation validator
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
    if require_citations is not None:
        config_params["require_citations"] = require_citations
    if min_citations is not None:
        config_params["min_citations"] = min_citations
    if max_citations is not None:
        config_params["max_citations"] = max_citations

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalCitationConfig(**config_params)

    # Return configured validator
    return DefaultLegalCitationValidator(config)


def create_legal_citation_rule(
    name: str = "legal_citation_rule",
    description: str = "Validates legal citations",
    citation_patterns: Optional[List[str]] = None,
    require_citations: Optional[bool] = None,
    min_citations: Optional[int] = None,
    max_citations: Optional[int] = None,
    **kwargs,
) -> LegalCitationRule:
    """
    Create a legal citation validation rule.

    This factory function creates a configured LegalCitationRule instance.
    It uses create_legal_citation_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        citation_patterns: List of regex patterns for legal citations
        require_citations: Whether citations are required
        min_citations: Minimum number of citations required
        max_citations: Maximum number of citations allowed
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured LegalCitationRule
    """
    # Create validator using the validator factory
    validator = create_legal_citation_validator(
        citation_patterns=citation_patterns,
        require_citations=require_citations,
        min_citations=min_citations,
        max_citations=max_citations,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalCitationRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_legal_terms_validator(
    legal_terms: Optional[Set[str]] = None,
    warning_terms: Optional[Set[str]] = None,
    required_terms: Optional[Set[str]] = None,
    prohibited_terms: Optional[Set[str]] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs,
) -> DefaultLegalTermsValidator:
    """
    Create a legal terms validator with the specified configuration.

    This factory function creates a configured legal terms validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        legal_terms: Set of legal terms to check for
        warning_terms: Set of warning terms to check for
        required_terms: Set of terms that must be present
        prohibited_terms: Set of terms that must not be present
        case_sensitive: Whether to make checks case-sensitive
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured legal terms validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if legal_terms is not None:
        config_params["legal_terms"] = legal_terms
    if warning_terms is not None:
        config_params["warning_terms"] = warning_terms
    if required_terms is not None:
        config_params["required_terms"] = required_terms
    if prohibited_terms is not None:
        config_params["prohibited_terms"] = prohibited_terms
    if case_sensitive is not None:
        config_params["case_sensitive"] = case_sensitive

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = LegalTermsConfig(**config_params)

    # Return configured validator
    return DefaultLegalTermsValidator(config)


def create_legal_terms_rule(
    name: str = "legal_terms_rule",
    description: str = "Validates legal terms",
    legal_terms: Optional[Set[str]] = None,
    warning_terms: Optional[Set[str]] = None,
    required_terms: Optional[Set[str]] = None,
    prohibited_terms: Optional[Set[str]] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs,
) -> LegalTermsRule:
    """
    Create a legal terms validation rule.

    This factory function creates a configured LegalTermsRule instance.
    It uses create_legal_terms_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        legal_terms: Set of legal terms to check for
        warning_terms: Set of warning terms to check for
        required_terms: Set of terms that must be present
        prohibited_terms: Set of terms that must not be present
        case_sensitive: Whether to make checks case-sensitive
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured LegalTermsRule
    """
    # Create validator using the validator factory
    validator = create_legal_terms_validator(
        legal_terms=legal_terms,
        warning_terms=warning_terms,
        required_terms=required_terms,
        prohibited_terms=prohibited_terms,
        case_sensitive=case_sensitive,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return LegalTermsRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
