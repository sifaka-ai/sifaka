"""
Safety validation rules for Sifaka.

This module provides rules for validating the safety of text content,
ensuring that it does not contain harmful or unsafe content.
"""

import importlib
from typing import List, Optional, Any, Dict, Union

from pydantic import Field, PrivateAttr

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    BaseValidator,
    create_rule,
)
from sifaka.utils.state import StateManager, RuleState, create_rule_state


class SafetyValidator(BaseValidator[str]):
    """
    Validator for text safety.

    This validator checks if text contains unsafe or harmful content
    using a safety classifier.

    Attributes:
        categories: List of safety categories to check
        threshold: Confidence threshold for safety detection
    """

    categories: List[str] = Field(
        default=["toxicity", "profanity", "sexual", "hate"],
        description="List of safety categories to check",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for safety detection",
    )

    # State management
    _state: StateManager[RuleState] = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, categories: Optional[List[str]] = None, threshold: float = 0.7, **kwargs):
        """Initialize the validator."""
        super().__init__(**kwargs)
        if categories is not None:
            self.categories = categories
        self.threshold = threshold

    def warm_up(self) -> None:
        """Initialize the validator if needed."""
        if not self._state.is_initialized:
            state = self._state.initialize()
            try:
                # Try to import detoxify
                state.detoxify = importlib.import_module("detoxify")
                state.model = state.detoxify.Detoxify("original")
            except ImportError:
                state.detoxify = None
                state.model = None
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate that text does not contain unsafe content.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state.get_state()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Check if detoxify is available
        if state.detoxify is None or state.model is None:
            return RuleResult(
                passed=False,
                message="detoxify package is required for safety validation. Install with: pip install detoxify",
                metadata={"reason": "missing_dependency"},
            )

        try:
            # Predict toxicity scores
            results = state.model.predict(text)

            # Check if any category exceeds the threshold
            violations = []
            for category in self.categories:
                if category in results and results[category] >= self.threshold:
                    violations.append(
                        {
                            "category": category,
                            "score": float(results[category]),
                        }
                    )

            # Create metadata with all scores
            metadata = {
                "scores": {k: float(v) for k, v in results.items()},
                "threshold": self.threshold,
                "categories": self.categories,
            }

            if violations:
                # Failed validation
                message = (
                    f"Text contains unsafe content: {', '.join(v['category'] for v in violations)}"
                )
                metadata["violations"] = violations
                return RuleResult(
                    passed=False,
                    message=message,
                    metadata=metadata,
                )
            else:
                # Passed validation
                return RuleResult(
                    passed=True,
                    message="Text does not contain unsafe content",
                    metadata=metadata,
                )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Safety validation failed: {str(e)}",
                metadata={"error": str(e)},
            )


class SafetyRule(Rule[str, RuleResult, SafetyValidator, Any]):
    """
    Rule that validates text safety.

    This rule ensures that text does not contain unsafe or harmful content.
    It uses a safety classifier to detect potentially harmful content.

    Attributes:
        _name: The name of the rule
        _description: Description of the rule
        _config: Rule configuration
        _validator: The validator used by this rule
    """

    # State management
    _state: StateManager[RuleState] = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._state.is_initialized:
            state = self._state.initialize()
            state.validator = self._create_default_validator()
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """Validate the text."""
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state.get_state()

        # Check cache
        cache_key = text
        if cache_key in state.cache:
            return state.cache[cache_key]

        # Delegate to validator
        result = state.validator.validate(text, **kwargs)

        # Cache result
        state.cache[cache_key] = result

        return result

    def _create_default_validator(self) -> SafetyValidator:
        """
        Create a default validator.

        Returns:
            A SafetyValidator with default settings
        """
        params = self._config.params
        return SafetyValidator(
            categories=params.get("categories", ["toxicity", "profanity", "sexual", "hate"]),
            threshold=params.get("threshold", 0.7),
        )


def create_safety_validator(
    categories: Optional[List[str]] = None,
    threshold: float = 0.7,
    **kwargs: Any,
) -> SafetyValidator:
    """
    Create a safety validator.

    This factory function creates a validator that ensures text does not contain
    unsafe or harmful content. It uses a safety classifier to detect potentially
    harmful content.

    Args:
        categories: List of safety categories to check
        threshold: Confidence threshold for safety detection
        **kwargs: Additional parameters for the validator

    Returns:
        A validator that validates text safety

    Examples:
        ```python
        from sifaka.rules.content.safety import create_safety_validator

        # Create a validator that checks for toxicity and profanity
        validator = create_safety_validator(
            categories=["toxicity", "profanity"],
            threshold=0.8
        )

        # Validate text
        result = validator.validate("This is safe text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'detoxify' package to be installed:
    pip install detoxify
    """
    if categories is None:
        categories = ["toxicity", "profanity", "sexual", "hate"]

    validator = SafetyValidator(
        categories=categories,
        threshold=threshold,
        **kwargs,
    )

    # Initialize the validator
    validator.warm_up()

    return validator


def create_safety_rule(
    categories: Optional[List[str]] = None,
    threshold: float = 0.7,
    name: str = "safety_rule",
    description: str = "Validates that text does not contain unsafe content",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule that validates text safety.

    This factory function creates a rule that ensures text does not contain
    unsafe or harmful content. It uses a safety classifier to detect potentially
    harmful content.

    Args:
        categories: List of safety categories to check
        threshold: Confidence threshold for safety detection
        name: The name of the rule
        description: Description of the rule
        config: Rule configuration
        **kwargs: Additional parameters for the rule

    Returns:
        A rule that validates text safety

    Examples:
        ```python
        from sifaka.rules.content.safety import create_safety_rule

        # Create a rule that checks for toxicity and profanity
        rule = create_safety_rule(
            categories=["toxicity", "profanity"],
            threshold=0.8,
            name="content_safety_rule"
        )

        # Validate text
        result = rule.validate("This is safe text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'detoxify' package to be installed:
    pip install detoxify
    """
    if categories is None:
        categories = ["toxicity", "profanity", "sexual", "hate"]

    # Create rule configuration
    rule_config = config or RuleConfig()
    rule_config = rule_config.with_params(
        categories=categories,
        threshold=threshold,
        **{k: v for k, v in kwargs.items() if k not in ["name", "description"]},
    )

    # Create rule
    rule = SafetyRule(
        name=name,
        description=description,
        config=rule_config,
    )

    # Initialize the rule
    rule.warm_up()

    return rule
