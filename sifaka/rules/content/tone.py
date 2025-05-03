"""
Tone consistency validation rules for Sifaka.

This module provides validators and rules for checking tone consistency in text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions (create_tone_consistency_rule, create_tone_consistency_validator) handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.content.tone import create_tone_consistency_rule

    # Create a tone consistency rule using the factory function
    rule = create_tone_consistency_rule(
        expected_tone="formal",
        threshold=0.8,
        tone_indicators={
            "formal": {
                "positive": ["therefore", "consequently", "furthermore"],
                "negative": ["yo", "hey", "cool"]
            }
        }
    )

    # Validate text
    result = rule.validate("Therefore, we can conclude that the hypothesis is valid.")

    # Create standalone validator
    from sifaka.rules.content.tone import create_tone_consistency_validator
    validator = create_tone_consistency_validator(
        expected_tone="formal",
        threshold=0.8
    )
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import ContentValidator


# Default tone indicators for different tone styles
DEFAULT_TONE_INDICATORS: Dict[str, Dict[str, List[str]]] = {
    "formal": {
        "positive": [
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "thus",
            "hence",
        ],
        "negative": [
            "yo",
            "hey",
            "cool",
            "awesome",
            "btw",
            "gonna",
            "wanna",
        ],
    },
    "informal": {
        "positive": [
            "hey",
            "hi",
            "cool",
            "great",
            "awesome",
            "nice",
            "yeah",
        ],
        "negative": [
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "thus",
            "hence",
        ],
    },
    "neutral": {
        "positive": [
            "the",
            "is",
            "are",
            "this",
            "that",
            "these",
            "those",
        ],
        "negative": [
            "!",
            "!!",
            "???",
            "omg",
            "wow",
            "awesome",
            "terrible",
        ],
    },
}

DEFAULT_THRESHOLD: float = 0.7


class ToneConfig(BaseModel):
    """Configuration for tone consistency validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_tone: str = Field(..., description="The expected tone style")
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of positive indicators to pass validation",
    )
    tone_indicators: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=lambda: DEFAULT_TONE_INDICATORS,
        description="Dictionary of tone indicators for different styles",
    )

    @field_validator("expected_tone")
    @classmethod
    def validate_expected_tone(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate that the expected tone exists in tone indicators."""
        tone_indicators = values.get("tone_indicators", {})
        if v not in tone_indicators:
            raise ConfigurationError(f"Expected tone '{v}' not found in tone indicators")
        return v


class ToneAnalyzer(BaseModel):
    """Analyze text for tone consistency."""

    model_config = ConfigDict(frozen=True)

    config: ToneConfig

    def analyze(self, text: str) -> Tuple[float, List[str], List[str]]:
        """
        Analyze text for tone consistency.

        Args:
            text: The text to analyze

        Returns:
            Tuple containing:
            - Ratio of positive to total indicators
            - List of found positive indicators
            - List of found negative indicators
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        indicators = self.config.tone_indicators[self.config.expected_tone]

        positive_indicators = [
            word for word in indicators["positive"] if word.lower() in text_lower
        ]
        negative_indicators = [
            word for word in indicators["negative"] if word.lower() in text_lower
        ]

        total_indicators = len(positive_indicators) + len(negative_indicators)
        ratio = len(positive_indicators) / total_indicators if total_indicators > 0 else 1.0

        return ratio, positive_indicators, negative_indicators


class ToneValidator(BaseValidator[str]):
    """Validator that checks tone consistency."""

    def __init__(self, config: RuleConfig) -> None:
        """Initialize the validator."""
        self._config = config
        self._tone_config = ToneConfig(**config.params)
        self._analyzer = ToneAnalyzer(config=self._tone_config)

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate that the text matches the expected tone."""
        try:
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            ratio, positive_indicators, negative_indicators = self._analyzer.analyze(text)
            passed = ratio >= self._tone_config.threshold

            return RuleResult(
                passed=passed,
                message=(
                    f"Tone consistency ratio ({ratio:.2f}) meets threshold ({self._tone_config.threshold})"
                    if passed
                    else f"Tone consistency ratio ({ratio:.2f}) below threshold ({self._tone_config.threshold})"
                ),
                metadata={
                    "ratio": ratio,
                    "positive_indicators": positive_indicators,
                    "negative_indicators": negative_indicators,
                    "expected_tone": self._tone_config.expected_tone,
                    "threshold": self._tone_config.threshold,
                },
            )

        except Exception as e:
            raise ValidationError(f"Tone validation failed: {str(e)}") from e


class ToneConsistencyRule(Rule[str, RuleResult, ToneValidator, RuleResultHandler[RuleResult]]):
    """Rule that checks for tone consistency in text."""

    def __init__(
        self,
        name: str = "tone_consistency_rule",
        description: str = "Checks for tone consistency",
        config: Optional[RuleConfig] = None,
        validator: Optional[ToneValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the tone consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> ToneValidator:
        """Create a default validator from config."""
        return ToneValidator(self.config)


def create_tone_consistency_validator(
    expected_tone: str,
    threshold: float = DEFAULT_THRESHOLD,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    **kwargs,
) -> ToneValidator:
    """
    Create a tone consistency validator with the specified configuration.

    This factory function creates a configured tone consistency validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        expected_tone: The expected tone style
        threshold: Minimum ratio of positive indicators to pass validation
        tone_indicators: Dictionary of tone indicators for different styles
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured tone consistency validator
    """
    config_dict = {
        "expected_tone": expected_tone,
        "threshold": threshold,
        "tone_indicators": tone_indicators or DEFAULT_TONE_INDICATORS,
        **kwargs,
    }

    rule_config = RuleConfig(params=config_dict)
    return ToneValidator(rule_config)


def create_tone_consistency_rule(
    name: str = "tone_consistency_rule",
    description: str = "Validates text tone consistency",
    expected_tone: str = "formal",
    threshold: float = DEFAULT_THRESHOLD,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    **kwargs,
) -> ToneConsistencyRule:
    """
    Create a tone consistency rule with configuration.

    This factory function creates a configured ToneConsistencyRule instance.
    It uses create_tone_consistency_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        expected_tone: The expected tone style
        threshold: Minimum ratio of positive indicators to pass validation
        tone_indicators: Dictionary of tone indicators for different styles
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured ToneConsistencyRule instance
    """
    # Create validator using the validator factory
    validator = create_tone_consistency_validator(
        expected_tone=expected_tone,
        threshold=threshold,
        tone_indicators=tone_indicators,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return ToneConsistencyRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


# Export public classes and functions
__all__ = [
    # Helper classes
    "ToneConfig",
    "ToneAnalyzer",
    # Validator classes
    "ToneValidator",
    # Rule classes
    "ToneConsistencyRule",
    # Validator factory functions
    "create_tone_consistency_validator",
    # Rule factory functions
    "create_tone_consistency_rule",
]
