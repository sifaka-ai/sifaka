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

from typing import Dict, Any, Protocol, runtime_checkable, Final, Set, Optional, Type
from dataclasses import dataclass

from sifaka.rules.base import Rule, RuleResult


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


class DefaultProhibitedContentValidator:
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

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text does not contain prohibited terms.

        Args:
            text: The text to validate

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

    def can_validate(self, text: Any) -> bool:
        """Check if the validator can handle the input."""
        return isinstance(text, str)

    @property
    def validation_type(self) -> Type[str]:
        """Get the type of input this validator can handle."""
        return str


class ProhibitedContentRule(Rule):
    """Rule that checks if the text contains any prohibited terms."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[ProhibitedContentValidator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with prohibited content validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Custom prohibited content validator implementation
            config: Prohibited content validation configuration dictionary
        """
        # Create the config object first
        prohibited_config = ProhibitedContentConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultProhibitedContentValidator(prohibited_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, text: str) -> RuleResult:
        """
        Validate that the text does not contain prohibited terms.

        Args:
            text: The text to validate

        Returns:
            RuleResult with validation results
        """
        return self._validator.validate(text)


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
    prohibited_config = ProhibitedContentConfig(**(config or {}))
    return ProhibitedContentRule(
        name=name,
        description=description,
        config=prohibited_config,
    )


# Export public classes and functions
__all__ = [
    "ProhibitedContentRule",
    "ProhibitedContentConfig",
    "ProhibitedContentValidator",
    "DefaultProhibitedContentValidator",
    "create_prohibited_content_rule",
]
