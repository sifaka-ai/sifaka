"""
Prohibited content validation rules for Sifaka.

This module provides rules for validating text against prohibited content, supporting both
case-sensitive and case-insensitive matching of prohibited terms.

Architecture Notes:
- Uses immutable configuration via dataclasses
- Implements Protocol-based validation for dependency injection
- Provides factory functions for easy rule creation
- Includes comprehensive error handling and validation
- Follows single responsibility principle with clear separation of concerns
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult, Any


@dataclass(frozen=True)
class ProhibitedContentConfig:
    """Immutable configuration for prohibited content validation."""

    prohibited_terms: Set[str]
    case_sensitive: bool = False
    cache_size: int = 10
    priority: int = 2
    cost: float = 1.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.prohibited_terms, (set, frozenset, list)):
            object.__setattr__(self, "prohibited_terms", set(self.prohibited_terms))

        if not self.prohibited_terms:
            raise ValueError("prohibited_terms set cannot be empty")

        if not all(isinstance(term, str) for term in self.prohibited_terms):
            raise ValueError("all prohibited terms must be strings")

        if not all(term.strip() for term in self.prohibited_terms):
            raise ValueError("prohibited terms cannot be empty or whitespace-only")


@runtime_checkable
class ProhibitedContentValidator(Protocol):
    """Protocol for prohibited content validation components."""

    @property
    def config(self) -> ProhibitedContentConfig: ...

    def validate(self, text: str) -> RuleResult: ...


class DefaultProhibitedContentValidator(BaseValidator[str]):
    """Default implementation of prohibited content validation."""

    def __init__(self, config: ProhibitedContentConfig) -> None:
        """Initialize the validator with configuration."""
        self._config = config

    @property
    def config(self) -> ProhibitedContentConfig:
        """Get the validator configuration."""
        return self._config

    def _find_prohibited_terms(self, text: str) -> list[str]:
        """
        Find all prohibited terms in the text.

        Args:
            text: The text to check

        Returns:
            List of found prohibited terms
        """
        if not self.config.case_sensitive:
            text = text.lower()
            terms = {term.lower() for term in self.config.prohibited_terms}
        else:
            terms = self.config.prohibited_terms

        return [term for term in terms if term in text]

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate that the text does not contain prohibited terms.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValueError: If text is not a string
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        found_terms = self._find_prohibited_terms(text)
        metadata = {
            "found_terms": found_terms,
            "total_terms": len(found_terms),
            "case_sensitive": self.config.case_sensitive,
        }

        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Found prohibited terms: {', '.join(found_terms)}",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message="No prohibited terms found",
            metadata=metadata,
        )


class ProhibitedContentRule(Rule[str, RuleResult, DefaultProhibitedContentValidator, Any]):
    """Rule that checks if the text contains any prohibited terms."""

    def __init__(
        self,
        name: str = "prohibited_content_rule",
        description: str = "Checks if text contains prohibited terms",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultProhibitedContentValidator] = None,
    ) -> None:
        """
        Initialize the rule with prohibited content validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Custom validator implementation
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultProhibitedContentValidator:
        """Create a default validator from config."""
        rule_config = ProhibitedContentConfig(**self._rule_params)
        return DefaultProhibitedContentValidator(rule_config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    config: Optional[Dict[str, Any]] = None,
) -> ProhibitedContentRule:
    """
    Factory function to create a prohibited content rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Dictionary of configuration parameters

    Returns:
        Configured ProhibitedContentRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return ProhibitedContentRule(
        name=name,
        description=description,
        config=rule_config,
    )


# Export public classes and functions
__all__ = [
    "ProhibitedContentRule",
    "ProhibitedContentConfig",
    "ProhibitedContentValidator",
    "DefaultProhibitedContentValidator",
    "create_prohibited_content_rule",
]
