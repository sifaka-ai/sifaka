"""
Prohibited content validation rules for Sifaka.
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
    ) -> None:
        """
        Initialize the prohibited content rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
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
        )

    def _create_default_validator(self) -> DefaultProhibitedContentValidator:
        """Create a default validator from config."""
        rule_config = RuleConfig(params=self._rule_params)
        return DefaultProhibitedContentValidator(rule_config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    config: Optional[Dict[str, Any]] = None,
) -> ProhibitedContentRule:
    """
    Create a prohibited content rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured ProhibitedContentRule instance
    """
    if config is None:
        config = {
            "terms": [
                "profanity",
                "obscenity",
                "hate speech",
                "explicit content",
                "adult content",
                "nsfw",
                "inappropriate",
            ],
            "case_sensitive": False,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return ProhibitedContentRule(
        name=name,
        description=description,
        config=rule_config,
    )
