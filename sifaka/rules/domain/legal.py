"""
Legal domain-specific validation rules for Sifaka.

This module provides validators and rules for checking legal content, citations, and terminology.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The LegalConfig, LegalCitationConfig, and LegalTermsConfig classes extend RuleConfig
      and provide type-safe access to parameters
    - Factory functions (create_legal_rule, create_legal_citation_rule, create_legal_terms_rule)
      handle configuration

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

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = LegalRule(
        config=RuleConfig(
            params={
                "disclaimer_required": True,
                "legal_terms": {
                    "jurisdiction": ["court", "venue", "forum"],
                    "liability": ["liability", "responsibility", "duty"]
                }
            }
        )
    )
"""

# Standard library
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
    Pattern,
)

# Third-party
from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

# Sifaka
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
    # Internal helpers (non-exported)
    "_DisclaimerAnalyzer",
    "_LegalTermAnalyzer",
    "_CitationAnalyzer",
]


class LegalConfig(BaseModel):
    """Configuration for legal rules."""

    model_config = ConfigDict(frozen=True)

    legal_terms: List[str] = Field(
        default_factory=lambda: [
            "copyright",
            "trademark",
            "patent",
            "license",
            "agreement",
            "contract",
            "terms",
            "conditions",
            "warranty",
            "liability",
        ],
        description="List of legal terms to validate",
    )
    citation_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\d+\s*(?:U\.?S\.?|F\.?(?:2d|3d)?|S\.?Ct\.?)\s*\d+",  # Federal cases
            r"\d+\s*[A-Z][a-z]*\.?\s*(?:2d|3d)?\s*\d+",  # State cases
            r"(?:\d+\s*)?U\.?S\.?C\.?\s*§*\s*\d+(?:\([a-z]\))?",  # U.S. Code
            r"\d+\s*(?:Cal\.?|N\.?Y\.?|Tex\.?)\s*(?:2d|3d|4th)?\s*\d+",  # State reporters
            r"(?:pub\.?\s*l\.?|P\.?L\.?)\s*\d+[-‐]\d+",  # Public Laws
            r"(?:CFR|C\.F\.R\.)\s*§*\s*\d+\.\d+",  # Code of Federal Regulations
            r"\d+\s*L\.?\s*Ed\.?\s*(?:2d)?\s*\d+",  # Supreme Court (Lawyers' Edition)
        ],
        description="List of regex patterns for legal citations",
    )
    disclaimers: List[str] = Field(
        default_factory=lambda: [
            "This is not legal advice",
            "Consult an attorney",
            "For informational purposes only",
            "Not a substitute for legal counsel",
        ],
        description="List of acceptable legal disclaimers",
    )
    disclaimer_required: bool = Field(
        default=True,
        description="Whether to require a legal disclaimer",
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

    @field_validator("legal_terms")
    @classmethod
    def validate_legal_terms(cls, v: List[str]) -> List[str]:
        """Validate that legal terms are not empty."""
        if not v:
            raise ValueError("Legal terms cannot be empty")
        return v

    @field_validator("citation_patterns")
    @classmethod
    def validate_citation_patterns(cls, v: List[str]) -> List[str]:
        """Validate that citation patterns are not empty."""
        if not v:
            raise ValueError("Must provide at least one citation pattern")
        return v

    @field_validator("disclaimers")
    @classmethod
    def validate_disclaimers(cls, v: List[str]) -> List[str]:
        """Validate that disclaimers are not empty."""
        if not v:
            raise ValueError("Must provide at least one disclaimer pattern")
        return v


class LegalCitationConfig(BaseModel):
    """Configuration for legal citation validation."""

    citation_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\d+\s+U\.S\.\s+\d+",  # US Reports citations
            r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
            r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter citations
            r"\d+\s+F\.\s*Supp\.\s+\d+",  # Federal Supplement
        ],
        description="List of regex patterns for legal citations",
    )
    require_citations: bool = Field(
        default=True,
        description="Whether citations are required in the text",
    )
    min_citations: int = Field(
        default=0,
        ge=0,
        description="Minimum number of citations required",
    )
    max_citations: int = Field(
        default=100,
        ge=0,
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
        """Validate that citation patterns are valid strings."""
        if not all(isinstance(p, str) for p in v):
            raise ValueError("citation_patterns must contain only strings")
        return v

    @field_validator("max_citations")
    @classmethod
    def validate_max_citations(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that max_citations is greater than or equal to min_citations."""
        if "min_citations" in values and v < values["min_citations"]:
            raise ValueError("max_citations must be greater than or equal to min_citations")
        return v


class LegalTermsConfig(BaseModel):
    """Configuration for legal terms validation."""

    legal_terms: Set[str] = Field(
        default_factory=lambda: {
            "confidential",
            "proprietary",
            "restricted",
            "private",
            "sensitive",
        },
        description="Set of legal terms to check for",
    )
    warning_terms: Set[str] = Field(
        default_factory=lambda: {
            "warning",
            "caution",
            "notice",
            "disclaimer",
            "privileged",
        },
        description="Set of warning terms to check for",
    )
    required_terms: Set[str] = Field(
        default_factory=set,
        description="Set of terms that must be present",
    )
    prohibited_terms: Set[str] = Field(
        default_factory=set,
        description="Set of terms that must not be present",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether term matching should be case sensitive",
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


# ---------------------------------------------------------------------------
# Analyzer helpers (Single Responsibility, re-usable)
# ---------------------------------------------------------------------------


class _DisclaimerAnalyzer(BaseModel):
    """Detect whether a text contains at least one required disclaimer pattern."""

    patterns: List[str] = Field(default_factory=list)

    _compiled: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [re.compile(pat, re.IGNORECASE) for pat in self.patterns]

    # Public API -----------------------------------------------------------
    def contains_disclaimer(self, text: str) -> bool:
        return any(pat.search(text) for pat in self._compiled)


class _LegalTermAnalyzer(BaseModel):
    """Count occurrences of legal terms grouped by category."""

    terms: Dict[str, List[str]] = Field(default_factory=dict)

    _compiled: Dict[str, List[re.Pattern[str]]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = {
            cat: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in term_list]
            for cat, term_list in self.terms.items()
        }

    def analyze(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cat, patterns in self._compiled.items():
            counts[cat] = sum(len(p.findall(text)) for p in patterns)
        return counts


class _CitationAnalyzer(BaseModel):
    """Locate citations, validate formatting, and compute totals."""

    patterns: List[str] = Field(default_factory=list)

    _compiled: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [re.compile(p) for p in self.patterns]

    def extract(self, text: str) -> List[str]:
        citations: List[str] = []
        for pat in self._compiled:
            citations.extend(pat.findall(text))
        return citations

    def invalid(self, citations: List[str]) -> List[str]:
        return [c for c in citations if not any(p.match(c) for p in self._compiled)]


class DefaultLegalValidator(BaseDomainValidator):
    """Default implementation of legal content validation (delegates to analyzers)."""

    def __init__(self, config: LegalConfig) -> None:
        super().__init__(config)

        self._disclaimer_analyzer = _DisclaimerAnalyzer(patterns=config.disclaimers)
        self._term_analyzer = _LegalTermAnalyzer(terms=config.legal_terms)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> LegalConfig:  # type: ignore[override]
        return self._config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401 – simple desc
        """Validate *text* for legal content consistency and disclaimers."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            has_disclaimer = self._disclaimer_analyzer.contains_disclaimer(text)
            term_counts = self._term_analyzer.analyze(text)

            if self.config.disclaimer_required and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="No legal disclaimer found when required",
                    metadata={
                        "legal_term_counts": term_counts,
                        "has_disclaimer": False,
                        "disclaimer_required": True,
                    },
                )

            return RuleResult(
                passed=True,
                message="Legal content validation passed",
                metadata={
                    "legal_term_counts": term_counts,
                    "has_disclaimer": has_disclaimer,
                    "disclaimer_required": self.config.disclaimer_required,
                },
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating legal content: {e}",
                metadata={"error": str(e)},
            )


class DefaultLegalCitationValidator(BaseDomainValidator):
    """Default implementation of legal citation validation using _CitationAnalyzer."""

    def __init__(self, config: LegalCitationConfig) -> None:
        super().__init__(config)
        self._citation_analyzer = _CitationAnalyzer(patterns=config.citation_patterns)

    @property
    def config(self) -> LegalCitationConfig:  # type: ignore[override]
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Validate *text* for citation presence, count, and correctness."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            citations = self._citation_analyzer.extract(text)
            invalid = self._citation_analyzer.invalid(citations)
            total = len(citations)

            # Requirements checks ------------------------------------------------
            if self.config.require_citations and total == 0:
                return RuleResult(
                    passed=False,
                    message="No citations found when required",
                    metadata={"total_citations": 0},
                )

            if total < self.config.min_citations:
                return RuleResult(
                    passed=False,
                    message=(
                        f"Found {total} citations; minimum required is {self.config.min_citations}"
                    ),
                    metadata={"total_citations": total},
                )

            if total > self.config.max_citations:
                return RuleResult(
                    passed=False,
                    message=(
                        f"Found {total} citations; maximum allowed is {self.config.max_citations}"
                    ),
                    metadata={"total_citations": total},
                )

            if invalid:
                return RuleResult(
                    passed=False,
                    message=f"Found {len(invalid)} invalid citations",
                    metadata={"invalid_citations": invalid, "total_citations": total},
                )

            return RuleResult(
                passed=True,
                message=f"Found {total} valid citations",
                metadata={"total_citations": total},
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating citations: {e}",
                metadata={"error": str(e)},
            )


class DefaultLegalTermsValidator(BaseDomainValidator):
    """Default implementation of legal terms validation with analyzers."""

    def __init__(self, config: LegalTermsConfig) -> None:
        super().__init__(config)

        flags = 0 if config.case_sensitive else re.IGNORECASE

        # Pre-compile sets for quick membership checks
        self._legal_patterns = [re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.legal_terms]
        self._warning_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.warning_terms
        ]
        self._required_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.required_terms
        ]
        self._prohibited_patterns = [
            re.compile(r"\b" + re.escape(t) + r"\b", flags) for t in config.prohibited_terms
        ]

    @property
    def config(self) -> LegalTermsConfig:  # type: ignore[override]
        return self._config

    def _matches(self, patterns: List[re.Pattern[str]], text: str) -> Set[str]:
        return {p.pattern.strip("\\b").strip("\\b") for p in patterns if p.search(text)}

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Validate legal term usage in *text*."""

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            legal_found = self._matches(self._legal_patterns, text)
            warning_found = self._matches(self._warning_patterns, text)
            missing_required = {
                p.pattern.strip("\\b").strip("\\b")
                for p in self._required_patterns
                if not p.search(text)
            }
            prohibited_found = self._matches(self._prohibited_patterns, text)

            if missing_required:
                return RuleResult(
                    passed=False,
                    message="Missing required legal terms",
                    metadata={
                        "missing_required_terms": missing_required,
                        "legal_terms_found": legal_found,
                        "warning_terms_found": warning_found,
                        "prohibited_terms_found": prohibited_found,
                    },
                )

            if prohibited_found:
                return RuleResult(
                    passed=False,
                    message="Found prohibited legal terms",
                    metadata={
                        "missing_required_terms": missing_required,
                        "legal_terms_found": legal_found,
                        "warning_terms_found": warning_found,
                        "prohibited_terms_found": prohibited_found,
                    },
                )

            return RuleResult(
                passed=True,
                message="Legal terms validation passed",
                metadata={
                    "legal_terms_found": legal_found,
                    "warning_terms_found": warning_found,
                },
            )
        except Exception as e:  # pragma: no cover
            return RuleResult(
                passed=False,
                message=f"Error validating legal terms: {e}",
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
