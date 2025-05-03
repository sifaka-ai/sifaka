"""
Medical domain-specific validation rules for Sifaka.

This module provides rules for validating medical content, including:
- Medical terminology validation
- Medical disclaimer validation
- Medical warning validation

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.domain.medical import create_medical_rule

    # Create a medical rule
    rule = create_medical_rule(
        medical_terms=["diagnosis", "treatment", "symptom"],
        require_disclaimer=True
    )
"""

import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.domain.base import DomainValidator as BaseDomainValidator


# Default medical terms
DEFAULT_MEDICAL_TERMS: List[str] = [
    "diagnosis",
    "treatment",
    "symptom",
    "condition",
    "disease",
    "medication",
    "prescription",
    "therapy",
    "procedure",
    "surgery",
]

# Default medical warning terms
DEFAULT_WARNING_TERMS: List[str] = [
    "emergency",
    "urgent",
    "critical",
    "severe",
    "life-threatening",
    "dangerous",
    "risk",
    "warning",
    "caution",
    "alert",
]

# Default medical disclaimers
DEFAULT_DISCLAIMERS: List[str] = [
    "This is not medical advice",
    "Consult a healthcare professional",
    "For informational purposes only",
    "Not a substitute for professional medical advice",
]


class MedicalConfig(BaseModel):
    """Configuration for medical content validation."""

    model_config = ConfigDict(frozen=True)

    medical_terms: List[str] = Field(
        default_factory=lambda: DEFAULT_MEDICAL_TERMS,
        description="List of medical terms to validate",
    )
    warning_terms: List[str] = Field(
        default_factory=lambda: DEFAULT_WARNING_TERMS,
        description="List of medical warning terms",
    )
    require_disclaimer: bool = Field(
        default=True,
        description="Whether to require a medical disclaimer",
    )
    disclaimers: List[str] = Field(
        default_factory=lambda: DEFAULT_DISCLAIMERS,
        description="List of acceptable medical disclaimers",
    )

    @field_validator("medical_terms")
    @classmethod
    def validate_medical_terms(cls, v: List[str]) -> List[str]:
        """Validate that medical terms are not empty."""
        if not v:
            raise ValueError("Medical terms cannot be empty")
        return v


__all__ = [
    # Config classes
    "MedicalConfig",
    # Protocol classes
    "MedicalValidator",
    # Validator classes
    "DefaultMedicalValidator",
    # Rule classes
    "MedicalRule",
    # Factory functions
    "create_medical_validator",
    "create_medical_rule",
    # Internal helpers
    "_MedicalTermAnalyzer",
    "_MedicalDisclaimerAnalyzer",
]


class _MedicalDisclaimerAnalyzer(BaseModel):
    """Detect medical disclaimers in text."""

    patterns: List[str] = Field(
        default_factory=lambda: [
            r"(?i)not\s+medical\s+advice",
            r"(?i)consult\s+(?:a|your)\s+(?:doctor|physician|healthcare\s+provider)",
            r"(?i)seek\s+medical\s+(?:attention|advice|care)",
            r"(?i)for\s+informational\s+purposes\s+only",
        ]
    )

    _compiled: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [re.compile(pat) for pat in self.patterns]

    def contains_disclaimer(self, text: str) -> bool:
        return any(p.search(text) for p in self._compiled)


class _MedicalTermAnalyzer(BaseModel):
    """Identify medical terms and warning terms within text."""

    term_categories: Dict[str, List[str]] = Field(default_factory=dict)
    warning_terms: Set[str] = Field(default_factory=set)

    _compiled_categories: Dict[str, List[re.Pattern[str]]] = PrivateAttr(default_factory=dict)
    _compiled_warning: List[re.Pattern[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled_categories = {
            cat: [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in terms]
            for cat, terms in self.term_categories.items()
        }
        self._compiled_warning = [
            re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in self.warning_terms
        ]

    def analyze(self, text: str) -> tuple[Dict[str, List[str]], List[str]]:
        found_terms: Dict[str, List[str]] = {}
        for cat, patterns in self._compiled_categories.items():
            matches = [p.pattern.strip("\\b").strip("\\b") for p in patterns if p.search(text)]
            if matches:
                found_terms[cat] = matches

        warnings = [
            p.pattern.strip("\\b").strip("\\b") for p in self._compiled_warning if p.search(text)
        ]
        return found_terms, warnings


class DefaultMedicalValidator(BaseDomainValidator):
    """Default implementation of medical content validation with analyzers."""

    def __init__(self, config: MedicalConfig) -> None:
        super().__init__(config)

        self._term_analyzer = _MedicalTermAnalyzer(
            term_categories=config.medical_terms, warning_terms=config.warning_terms
        )
        self._disc_analyzer = _MedicalDisclaimerAnalyzer()

    @property
    def config(self) -> MedicalConfig:  # type: ignore[override]
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:  # noqa: D401
        """Check *text* for medical terms, warnings, and required disclaimer."""

        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        found_terms, warning_terms = self._term_analyzer.analyze(text)
        has_disclaimer = self._disc_analyzer.contains_disclaimer(text)

        # Decision logic
        if found_terms:
            if self.config.require_disclaimer and not has_disclaimer:
                return RuleResult(
                    passed=False,
                    message="Medical content requires a disclaimer",
                    metadata={
                        "found_terms": found_terms,
                        "warning_terms": warning_terms,
                        "has_disclaimer": has_disclaimer,
                    },
                )

            if warning_terms:
                return RuleResult(
                    passed=False,
                    message="Contains potentially unsafe medical terms",
                    metadata={
                        "found_terms": found_terms,
                        "warning_terms": warning_terms,
                        "has_disclaimer": has_disclaimer,
                    },
                )

        return RuleResult(
            passed=True,
            message="No unsafe medical content detected",
            metadata={
                "found_terms": found_terms,
                "warning_terms": warning_terms,
                "has_disclaimer": has_disclaimer,
            },
        )


class MedicalRule(Rule[str, RuleResult, DefaultMedicalValidator, Any]):
    """Rule that checks for medical content accuracy and safety."""

    def __init__(
        self,
        name: str = "medical_rule",
        description: str = "Checks for medical content accuracy and safety",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultMedicalValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the medical rule.

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

    def _create_default_validator(self) -> DefaultMedicalValidator:
        """Create a default validator from config."""
        medical_config = MedicalConfig(**self._rule_params)
        return DefaultMedicalValidator(medical_config)


def create_medical_validator(
    medical_terms: Optional[Dict[str, List[str]]] = None,
    warning_terms: Optional[Set[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> DefaultMedicalValidator:
    """
    Create a medical validator with the specified configuration.

    This factory function creates a configured medical validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        medical_terms: Dictionary mapping categories to lists of medical terms
        warning_terms: Set of terms that should trigger warnings
        disclaimer_required: Whether a medical disclaimer is required
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured medical validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if medical_terms is not None:
        config_params["medical_terms"] = medical_terms
    if warning_terms is not None:
        config_params["warning_terms"] = warning_terms
    if disclaimer_required is not None:
        config_params["require_disclaimer"] = disclaimer_required

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = MedicalConfig(**config_params)

    # Return configured validator
    return DefaultMedicalValidator(config)


def create_medical_rule(
    name: str = "medical_rule",
    description: str = "Validates text for medical content",
    medical_terms: Optional[Dict[str, List[str]]] = None,
    warning_terms: Optional[Set[str]] = None,
    disclaimer_required: Optional[bool] = None,
    **kwargs,
) -> MedicalRule:
    """
    Create a medical rule with configuration.

    This factory function creates a configured MedicalRule instance.
    It uses create_medical_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        medical_terms: Dictionary mapping categories to lists of medical terms
        warning_terms: Set of terms that should trigger warnings
        disclaimer_required: Whether a medical disclaimer is required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured MedicalRule instance
    """
    # Create validator using the validator factory
    validator = create_medical_validator(
        medical_terms=medical_terms,
        warning_terms=warning_terms,
        disclaimer_required=disclaimer_required,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return MedicalRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
