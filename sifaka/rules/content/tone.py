"""
Tone validation rules for Sifaka.

This module provides rules for validating the tone of text, ensuring
that text has the expected tone (formal, informal, etc.).
"""

from typing import List, Optional, Any, Dict, Union, Literal

from pydantic import Field, PrivateAttr

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    BaseValidator,
    create_rule,
)
from sifaka.utils.state import StateManager, RuleState, create_rule_state


class ToneValidator(BaseValidator[str]):
    """
    Validator for text tone.

    This validator checks if text has the expected tone (formal, informal, etc.).
    It uses a simple heuristic approach by default.

    Attributes:
        expected_tone: Expected tone of the text
        threshold: Threshold for tone confidence
        model_name: Name of the model to use for tone analysis
    """

    expected_tone: Literal["formal", "informal", "technical", "conversational"] = Field(
        default="formal",
        description="Expected tone of the text",
    )
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for tone confidence",
    )
    model_name: str = Field(
        default="heuristic",
        description="Name of the model to use for tone analysis",
    )

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self,
        expected_tone: Literal["formal", "informal", "technical", "conversational"] = "formal",
        threshold: float = 0.6,
        model_name: str = "heuristic",
        **kwargs,
    ):
        """Initialize the validator."""
        super().__init__(**kwargs)
        self.expected_tone = expected_tone
        self.threshold = threshold
        self.model_name = model_name

    def warm_up(self) -> None:
        """Initialize the validator if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()

            # Initialize tone indicators
            state.tone_indicators = {
                "formal": [
                    "furthermore",
                    "nevertheless",
                    "however",
                    "therefore",
                    "thus",
                    "consequently",
                    "hence",
                    "accordingly",
                    "subsequently",
                    "moreover",
                    "in addition",
                    "in conclusion",
                    "in summary",
                    "in regard to",
                ],
                "informal": [
                    "yeah",
                    "nope",
                    "cool",
                    "awesome",
                    "gonna",
                    "wanna",
                    "gotta",
                    "kinda",
                    "sorta",
                    "y'all",
                    "ain't",
                    "dunno",
                    "gimme",
                    "lemme",
                    "btw",
                    "lol",
                    "omg",
                    "tbh",
                    "fyi",
                    "asap",
                ],
                "technical": [
                    "algorithm",
                    "implementation",
                    "functionality",
                    "interface",
                    "parameter",
                    "variable",
                    "function",
                    "method",
                    "class",
                    "object",
                    "instance",
                    "module",
                    "component",
                    "system",
                    "architecture",
                    "framework",
                    "infrastructure",
                    "protocol",
                    "specification",
                ],
                "conversational": [
                    "hey",
                    "hi",
                    "hello",
                    "so",
                    "well",
                    "anyway",
                    "like",
                    "you know",
                    "I mean",
                    "right",
                    "actually",
                    "basically",
                    "literally",
                    "honestly",
                    "seriously",
                    "absolutely",
                    "totally",
                    "really",
                    "just",
                    "pretty",
                ],
            }

            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate that text has the expected tone.

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

        try:
            # Analyze tone using heuristic approach
            text_lower = text.lower()

            # Count indicators for each tone
            tone_scores = {}
            for tone, indicators in state.tone_indicators.items():
                count = sum(1 for indicator in indicators if indicator.lower() in text_lower)
                # Normalize score to 0-1 range
                tone_scores[tone] = min(count / 5.0, 1.0)

            # Determine detected tone
            detected_tone = max(tone_scores, key=tone_scores.get)
            confidence = tone_scores[detected_tone]

            # Check if detected tone matches expected tone and confidence is above threshold
            if detected_tone == self.expected_tone and confidence >= self.threshold:
                return RuleResult(
                    passed=True,
                    message=f"Text has the expected {self.expected_tone} tone",
                    metadata={
                        "scores": tone_scores,
                        "detected_tone": detected_tone,
                        "confidence": confidence,
                        "threshold": self.threshold,
                    },
                )
            else:
                if detected_tone != self.expected_tone:
                    message = f"Text has {detected_tone} tone, expected {self.expected_tone}"
                else:
                    message = (
                        f"Tone confidence ({confidence:.2f}) is below threshold ({self.threshold})"
                    )

                return RuleResult(
                    passed=False,
                    message=message,
                    metadata={
                        "scores": tone_scores,
                        "detected_tone": detected_tone,
                        "confidence": confidence,
                        "threshold": self.threshold,
                    },
                )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Tone analysis failed: {str(e)}",
                metadata={"error": str(e)},
            )


class ToneRule(Rule[str, RuleResult, ToneValidator, Any]):
    """
    Rule that validates text tone.

    This rule ensures that text has the expected tone (formal, informal, etc.).
    It uses a simple heuristic approach by default.

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

    def _create_default_validator(self) -> ToneValidator:
        """
        Create a default validator.

        Returns:
            A ToneValidator with default settings
        """
        params = self._config.params
        return ToneValidator(
            expected_tone=params.get("expected_tone", "formal"),
            threshold=params.get("threshold", 0.6),
            model_name=params.get("model_name", "heuristic"),
        )


def create_tone_validator(
    expected_tone: Literal["formal", "informal", "technical", "conversational"] = "formal",
    threshold: float = 0.6,
    model_name: str = "heuristic",
    **kwargs: Any,
) -> ToneValidator:
    """
    Create a tone validator.

    This factory function creates a validator that ensures text has the expected
    tone (formal, informal, etc.). It uses a simple heuristic approach by default.

    Args:
        expected_tone: Expected tone of the text
        threshold: Threshold for tone confidence
        model_name: Name of the model to use for tone analysis
        **kwargs: Additional parameters for the validator

    Returns:
        A validator that validates text tone

    Examples:
        ```python
        from sifaka.rules.content.tone import create_tone_validator

        # Create a validator that checks for formal tone
        validator = create_tone_validator(
            expected_tone="formal",
            threshold=0.5
        )

        # Validate text
        result = validator.validate("Furthermore, it is imperative to consider...")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    validator = ToneValidator(
        expected_tone=expected_tone,
        threshold=threshold,
        model_name=model_name,
        **kwargs,
    )

    # Initialize the validator
    validator.warm_up()

    return validator


def create_tone_rule(
    expected_tone: Literal["formal", "informal", "technical", "conversational"] = "formal",
    threshold: float = 0.6,
    model_name: str = "heuristic",
    name: str = "tone_rule",
    description: str = "Validates that text has the expected tone",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a rule that validates text tone.

    This factory function creates a rule that ensures text has the expected
    tone (formal, informal, etc.). It uses a simple heuristic approach by default.

    Args:
        expected_tone: Expected tone of the text
        threshold: Threshold for tone confidence
        model_name: Name of the model to use for tone analysis
        name: The name of the rule
        description: Description of the rule
        config: Rule configuration
        **kwargs: Additional parameters for the rule

    Returns:
        A rule that validates text tone

    Examples:
        ```python
        from sifaka.rules.content.tone import create_tone_rule

        # Create a rule that checks for formal tone
        rule = create_tone_rule(
            expected_tone="formal",
            threshold=0.5,
            name="formal_tone_rule"
        )

        # Validate text
        result = rule.validate("Furthermore, it is imperative to consider...")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    # Create rule configuration
    rule_config = config or RuleConfig()
    rule_config = rule_config.with_params(
        expected_tone=expected_tone,
        threshold=threshold,
        model_name=model_name,
        **{k: v for k, v in kwargs.items() if k not in ["name", "description"]},
    )

    # Create rule
    rule = ToneRule(
        name=name,
        description=description,
        config=rule_config,
    )

    # Initialize the rule
    rule.warm_up()

    return rule


# Export public classes and functions
__all__ = [
    # Helper classes
    "ToneConfig",
    "ToneAnalyzer",
    # Validator classes
    "ToneValidator",
    # Rule classes
    "ToneRule",
    # Validator factory functions
    "create_tone_validator",
    # Rule factory functions
    "create_tone_rule",
]
