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
        # Convert to frozenset if not already a frozenset
        if not isinstance(self.prohibited_terms, frozenset):
            object.__setattr__(self, "prohibited_terms", frozenset(self.prohibited_terms))

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
        # For case-insensitive matching, we need to preserve the original terms
        # but do the comparison in lowercase
        original_terms = list(self.config.prohibited_terms)

        if not self.config.case_sensitive:
            check_text = text.lower()
            check_terms = [term.lower() for term in original_terms]
            # Return the original terms that match when lowercased
            return [original_terms[i] for i, term in enumerate(check_terms) if term in check_text]
        else:
            # For case-sensitive matching, just check if the terms are in the text
            return [term for term in original_terms if term in text]

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
        # Check for None or non-string input
        if text is None or not isinstance(text, str):
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
        # Default prohibited terms if not provided
        if "prohibited_terms" not in self._rule_params:
            self._rule_params["prohibited_terms"] = [
                "profanity",
                "obscenity",
                "hate speech",
                "explicit content",
                "adult content",
                "nsfw",
                "inappropriate",
                "offensive",
            ]

        # Create config with parameters
        rule_config = ProhibitedContentConfig(**self._rule_params)
        return DefaultProhibitedContentValidator(rule_config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    config: Optional[Any] = None,
    validator: Optional[DefaultProhibitedContentValidator] = None,
) -> ProhibitedContentRule:
    """
    Factory function to create a prohibited content rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Configuration parameters (dict, RuleConfig, or ProhibitedContentConfig)
        validator: Optional custom validator

    Returns:
        Configured ProhibitedContentRule instance
    """
    rule_config = None

    # Handle different config types
    if config is None:
        # Default configuration with common prohibited terms
        default_terms = [
            "profanity",
            "obscenity",
            "hate speech",
            "explicit content",
            "adult content",
            "nsfw",
            "inappropriate",
            "offensive",
        ]
        rule_config = RuleConfig(params={"prohibited_terms": default_terms})
    elif isinstance(config, dict):
        # Dictionary config
        rule_config = RuleConfig(params=config)
    elif isinstance(config, RuleConfig):
        # RuleConfig object
        rule_config = config
    elif isinstance(config, ProhibitedContentConfig):
        # ProhibitedContentConfig object - convert to RuleConfig
        rule_config = RuleConfig(
            params={
                "prohibited_terms": config.prohibited_terms,
                "case_sensitive": config.case_sensitive,
                "cache_size": config.cache_size,
                "priority": config.priority,
                "cost": config.cost,
            }
        )
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    return ProhibitedContentRule(
        name=name,
        description=description,
        config=rule_config,
        validator=validator,
    )


# Export public classes and functions
__all__ = [
    "ProhibitedContentRule",
    "ProhibitedContentConfig",
    "ProhibitedContentValidator",
    "DefaultProhibitedContentValidator",
    "create_prohibited_content_rule",
]
