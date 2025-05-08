"""
Prohibited content validation rules for Sifaka.

This module provides rules for validating that text does not contain prohibited
content, such as specific terms or phrases.
"""

import re
from typing import List, Optional, Any, Dict, Union, Pattern

from pydantic import Field, PrivateAttr

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    BaseValidator,
    create_rule,
)
from sifaka.utils.state import StateManager, RuleState, create_rule_state


class ProhibitedContentValidator(BaseValidator[str]):
    """
    Validator for prohibited content.

    This validator checks if text contains prohibited terms or phrases.

    Attributes:
        terms: List of prohibited terms or phrases
        case_sensitive: Whether the matching should be case-sensitive
        threshold: Threshold for the number of matches allowed (0 means no matches allowed)
    """

    terms: List[str] = Field(
        default=[],
        description="List of prohibited terms or phrases",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the matching should be case-sensitive",
    )
    threshold: int = Field(
        default=0,
        ge=0,
        description="Threshold for the number of matches allowed (0 means no matches allowed)",
    )

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self,
        terms: Optional[List[str]] = None,
        case_sensitive: bool = False,
        threshold: int = 0,
        **kwargs,
    ):
        """Initialize the validator."""
        super().__init__(**kwargs)
        if terms is not None:
            self.terms = terms
        self.case_sensitive = case_sensitive
        self.threshold = threshold

    def warm_up(self) -> None:
        """Initialize the validator if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            # Compile regex patterns for each term
            flags = 0 if self.case_sensitive else re.IGNORECASE
            state.patterns = [re.compile(re.escape(term), flags) for term in self.terms]
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate that text does not contain prohibited terms.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Check for prohibited terms
        matches = []
        for pattern in state.patterns:
            matches.extend(pattern.findall(text))

        # Check if the number of matches exceeds the threshold
        if len(matches) <= self.threshold:
            return RuleResult(
                passed=True,
                message=f"Text does not contain prohibited terms (or is within threshold)",
                metadata={
                    "matches": matches,
                    "match_count": len(matches),
                    "threshold": self.threshold,
                },
            )
        else:
            return RuleResult(
                passed=False,
                message=f"Text contains prohibited terms: {', '.join(set(matches))}",
                metadata={
                    "matches": matches,
                    "match_count": len(matches),
                    "threshold": self.threshold,
                },
            )


class ProhibitedContentRule(Rule[str, RuleResult, ProhibitedContentValidator, Any]):
    """
    Rule that validates text does not contain prohibited content.

    This rule ensures that text does not contain prohibited terms or phrases.
    It checks for exact matches of the specified terms in the text.

    Attributes:
        _name: The name of the rule
        _description: Description of the rule
        _config: Rule configuration
        _validator: The validator used by this rule
    """

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            state.validator = self._create_default_validator()
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """Validate the text."""
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Check cache
        cache_key = text
        if cache_key in state.cache:
            return state.cache[cache_key]

        # Delegate to validator
        result = state.validator.validate(text, **kwargs)

        # Cache result
        state.cache[cache_key] = result

        return result

    def _create_default_validator(self) -> ProhibitedContentValidator:
        """
        Create a default validator.

        Returns:
            A ProhibitedContentValidator with default settings
        """
        params = self._config.params
        return ProhibitedContentValidator(
            terms=params.get("terms", []),
            case_sensitive=params.get("case_sensitive", False),
            threshold=params.get("threshold", 0),
        )


def create_prohibited_content_validator(
    terms: Optional[List[str]] = None,
    case_sensitive: bool = False,
    threshold: int = 0,
    **kwargs: Any,
) -> ProhibitedContentValidator:
    """
    Create a prohibited content validator.

    This factory function creates a validator that ensures text does not contain
    prohibited terms or phrases. It checks for exact matches of the specified
    terms in the text.

    Args:
        terms: List of prohibited terms or phrases
        case_sensitive: Whether the matching should be case-sensitive
        threshold: Threshold for the number of matches allowed (0 means no matches allowed)
        **kwargs: Additional parameters for the validator

    Returns:
        A validator that validates text does not contain prohibited content

    Examples:
        ```python
        from sifaka.rules.content.prohibited import create_prohibited_content_validator

        # Create a validator that prohibits specific terms
        validator = create_prohibited_content_validator(
            terms=["inappropriate", "offensive"],
            case_sensitive=False
        )

        # Validate text
        result = validator.validate("This is appropriate text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    if terms is None:
        terms = []

    validator = ProhibitedContentValidator(
        terms=terms,
        case_sensitive=case_sensitive,
        threshold=threshold,
        **kwargs,
    )

    # Initialize the validator
    validator.warm_up()

    return validator


def create_prohibited_content_rule(
    terms: Optional[List[str]] = None,
    case_sensitive: bool = False,
    threshold: int = 0,
    name: str = "prohibited_content_rule",
    description: str = "Validates that text does not contain prohibited terms",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule that validates text does not contain prohibited content.

    This factory function creates a rule that ensures text does not contain
    prohibited terms or phrases. It checks for exact matches of the specified
    terms in the text.

    Args:
        terms: List of prohibited terms or phrases
        case_sensitive: Whether the matching should be case-sensitive
        threshold: Threshold for the number of matches allowed (0 means no matches allowed)
        name: The name of the rule
        description: Description of the rule
        config: Rule configuration
        **kwargs: Additional parameters for the rule

    Returns:
        A rule that validates text does not contain prohibited content

    Examples:
        ```python
        from sifaka.rules.content.prohibited import create_prohibited_content_rule

        # Create a rule that prohibits specific terms
        rule = create_prohibited_content_rule(
            terms=["inappropriate", "offensive"],
            case_sensitive=False,
            name="appropriate_content_rule"
        )

        # Validate text
        result = rule.validate("This is appropriate text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    if terms is None:
        terms = []

    # Create rule configuration
    rule_config = config or RuleConfig()
    rule_config = rule_config.with_params(
        terms=terms,
        case_sensitive=case_sensitive,
        threshold=threshold,
        **{k: v for k, v in kwargs.items() if k not in ["name", "description"]},
    )

    # Create rule
    rule = ProhibitedContentRule(
        name=name,
        description=description,
        config=rule_config,
    )

    # Initialize the rule
    rule.warm_up()

    return rule
