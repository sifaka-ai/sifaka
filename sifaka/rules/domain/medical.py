"""
Medical domain-specific validation rules for Sifaka.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator


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
    ) -> None:
        """
        Initialize the medical rule.

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

    def _create_default_validator(self) -> DefaultMedicalValidator:
        """Create a default validator from config."""
        medical_config = MedicalConfig(**self._rule_params)
        return DefaultMedicalValidator(medical_config)


def create_medical_rule(
    name: str = "medical_rule",
    description: str = "Validates text for medical content",
    config: Optional[Dict[str, Any]] = None,
) -> MedicalRule:
    """
    Create a medical rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured MedicalRule instance
    """
    if config is None:
        config = {
            "medical_terms": {
                "diagnosis": ["diagnosis", "diagnose", "diagnosed"],
                "treatment": ["treatment", "treat", "treating", "therapy"],
                "medication": ["medication", "drug", "prescription", "medicine"],
                "symptom": ["symptom", "symptoms", "sign", "signs"],
            },
            "warning_terms": {
                "diagnosis",
                "treatment",
                "medication",
                "prescription",
                "therapy",
                "cure",
                "heal",
                "remedy",
            },
            "disclaimer_required": True,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return MedicalRule(
        name=name,
        description=description,
        config=rule_config,
    )
