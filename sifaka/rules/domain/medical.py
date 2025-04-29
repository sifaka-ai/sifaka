"""
Medical domain-specific validation rules for Sifaka.

This module provides validators and rules for checking medical content.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The MedicalConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_medical_rule, create_medical_validator) handle configuration

Usage Example:
    from sifaka.rules.domain.medical import create_medical_rule

    # Create a medical rule using the factory function
    rule = create_medical_rule(
        medical_terms={
            "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
            "treatment": ["treatment", "therapy"]
        },
        warning_terms={"diagnosis", "treatment", "cure"},
        disclaimer_required=True
    )

    # Validate text
    result = rule.validate("This treatment may help with symptoms. Consult your doctor for medical advice.")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = MedicalRule(
        config=RuleConfig(
            params={
                "medical_terms": {
                    "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
                    "treatment": ["treatment", "therapy"]
                },
                "warning_terms": {"diagnosis", "treatment", "cure"},
                "disclaimer_required": True
            }
        )
    )
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator


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
]


@dataclass(frozen=True)
class MedicalConfig(RuleConfig):
    """Configuration for medical rules."""

    medical_terms: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
            "treatment": ["treatment", "treat", "treating", "therapy"],
            "medication": ["medication", "drug", "prescription", "medicine"],
            "symptom": ["symptom", "symptoms", "sign", "signs"],
        }
    )
    warning_terms: Set[str] = field(
        default_factory=lambda: {
            "diagnosis",
            "treatment",
            "medication",
            "prescription",
            "therapy",
            "cure",
            "heal",
            "remedy",
        }
    )
    disclaimer_required: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.medical_terms:
            raise ValueError("Must provide at least one medical term category")
        if not self.warning_terms:
            raise ValueError("Must provide at least one warning term")


@runtime_checkable
class MedicalValidator(Protocol):
    """Protocol for medical content validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> MedicalConfig: ...


class DefaultMedicalValidator(BaseDomainValidator):
    """Default implementation of medical content validation."""

    def __init__(self, config: MedicalConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> MedicalConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for medical content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        found_terms: Dict[str, List[str]] = {}
        warning_terms: List[str] = []

        # Check for medical terms
        for category, terms in self.config.medical_terms.items():
            matches = [term for term in terms if term in text_lower]
            if matches:
                found_terms[category] = matches

        # Check for warning terms
        warning_terms = [term for term in self.config.warning_terms if term in text_lower]

        # Check for disclaimer if required
        has_disclaimer = False
        if self.config.disclaimer_required:
            disclaimer_patterns = [
                r"(?i)not\s+medical\s+advice",
                r"(?i)consult\s+(?:a|your)\s+(?:doctor|physician|healthcare\s+provider)",
                r"(?i)seek\s+medical\s+(?:attention|advice|care)",
                r"(?i)for\s+informational\s+purposes\s+only",
            ]
            has_disclaimer = any(re.search(pattern, text) for pattern in disclaimer_patterns)

        if found_terms:
            if self.config.disclaimer_required and not has_disclaimer:
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
        config_params["disclaimer_required"] = disclaimer_required

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
