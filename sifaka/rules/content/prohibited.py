"""
Prohibited content validation rules for Sifaka.

This module provides validators and rules for checking text against prohibited content.

Usage Example:
    from sifaka.rules.content.prohibited import create_prohibited_content_rule

    # Create a prohibited content rule
    rule = create_prohibited_content_rule(
        terms=["inappropriate", "offensive", "vulgar"],
        case_sensitive=False
    )

    # Validate text
    result = rule.validate("This is a test.")
"""

from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Sequence

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import ContentAnalyzer, ContentValidator, DefaultContentAnalyzer


__all__ = [
    # Data classes
    "ProhibitedContentConfig",
    "ProhibitedTerms",
    # Validator classes
    "ProhibitedContentValidator",
    "DefaultProhibitedContentValidator",
    # Rule classes
    "ProhibitedContentRule",
    # Factory functions
    "create_prohibited_content_validator",
    "create_prohibited_content_rule",
]


@dataclass(frozen=True)
class ProhibitedContentConfig:
    """Configuration for prohibited content validation."""

    terms: List[str]
    case_sensitive: bool = False
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.terms:
            raise ConfigurationError("Prohibited terms list cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "terms": self.terms,
            "case_sensitive": self.case_sensitive,
            "priority": self.priority,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class ProhibitedTerms:
    """Immutable container for prohibited terms configuration."""

    terms: frozenset[str]
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.terms:
            raise ConfigurationError("Prohibited terms list cannot be empty")

    def with_terms(self, terms: Sequence[str]) -> "ProhibitedTerms":
        """Create new instance with updated terms."""
        return ProhibitedTerms(terms=frozenset(terms), case_sensitive=self.case_sensitive)

    def with_case_sensitivity(self, case_sensitive: bool) -> "ProhibitedTerms":
        """Create new instance with updated case sensitivity."""
        return ProhibitedTerms(terms=self.terms, case_sensitive=case_sensitive)


class ProhibitedContentValidator(ContentValidator):
    """Validator that checks for prohibited content."""

    def __init__(
        self,
        terms: ProhibitedTerms,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._terms: Final[ProhibitedTerms] = terms

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output does not contain prohibited terms."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            check_output = output if self._terms.case_sensitive else output.lower()
            check_terms = (
                self._terms.terms
                if self._terms.case_sensitive
                else frozenset(t.lower() for t in self._terms.terms)
            )

            found_terms = [term for term in check_terms if term in check_output]

            return RuleResult(
                passed=not found_terms,
                message=(
                    "No prohibited terms found"
                    if not found_terms
                    else f"Found prohibited terms: {', '.join(found_terms)}"
                ),
                metadata={
                    "found_terms": found_terms,
                    "case_sensitive": self._terms.case_sensitive,
                    "analysis": self._analyzer.analyze(output),
                },
            )

        except Exception as e:
            raise ValidationError(f"Content validation failed: {str(e)}") from e


class DefaultProhibitedContentValidator(BaseValidator[str]):
    """Default implementation of prohibited content validation."""

    def __init__(self, config: RuleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for prohibited content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params
        case_sensitive = self.config.params.get("case_sensitive", False)
        terms = self.config.params.get(
            "terms",
            [
                "profanity",
                "obscenity",
                "hate speech",
                "explicit content",
                "adult content",
                "nsfw",
                "inappropriate",
            ],
        )

        check_text = text if case_sensitive else text.lower()
        found_terms = []

        for term in terms:
            check_term = term if case_sensitive else term.lower()
            if check_term in check_text:
                found_terms.append(term)

        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Found prohibited terms: {', '.join(found_terms)}",
                metadata={
                    "found_terms": found_terms,
                    "case_sensitive": case_sensitive,
                },
            )

        return RuleResult(
            passed=True,
            message="No prohibited terms found",
            metadata={
                "found_terms": [],
                "case_sensitive": case_sensitive,
            },
        )


class ProhibitedContentRule(
    Rule[str, RuleResult, DefaultProhibitedContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for prohibited content in the output."""

    def __init__(
        self,
        name: str = "prohibited_content_rule",
        description: str = "Checks for prohibited content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultProhibitedContentValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the prohibited content rule.

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
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultProhibitedContentValidator:
        """Create a default validator from config."""
        rule_config = RuleConfig(params=self._rule_params)
        return DefaultProhibitedContentValidator(rule_config)


def create_prohibited_content_validator(
    terms: Optional[List[str]] = None,
    case_sensitive: bool = False,
    **kwargs,
) -> DefaultProhibitedContentValidator:
    """
    Create a prohibited content validator with the specified configuration.

    This factory function creates a configured prohibited content validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        terms: List of prohibited terms to check for
        case_sensitive: Whether to perform case-sensitive matching
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured prohibited content validator
    """
    # Set default values if not provided
    if terms is None:
        terms = [
            "profanity",
            "obscenity",
            "hate speech",
            "explicit content",
            "adult content",
            "nsfw",
            "inappropriate",
        ]

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config dictionary
    config_dict = {
        "terms": terms,
        "case_sensitive": case_sensitive,
        **rule_config_params,
    }

    # Create RuleConfig
    rule_config = RuleConfig(params=config_dict)

    # Return configured validator
    return DefaultProhibitedContentValidator(rule_config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    terms: Optional[List[str]] = None,
    case_sensitive: bool = False,
    **kwargs,
) -> ProhibitedContentRule:
    """
    Create a prohibited content rule with configuration.

    This factory function creates a configured ProhibitedContentRule instance.
    It uses create_prohibited_content_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        terms: List of prohibited terms to check for
        case_sensitive: Whether to perform case-sensitive matching
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured ProhibitedContentRule instance
    """
    # Create validator using the validator factory
    validator = create_prohibited_content_validator(
        terms=terms,
        case_sensitive=case_sensitive,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return ProhibitedContentRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
