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

# Standard library
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional

# Third-party
from pydantic import BaseModel, Field, PrivateAttr

# Sifaka
from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import ContentValidator, ToneAnalyzer, DefaultToneAnalyzer


@dataclass(frozen=True)
class ToneIndicators:
    """Immutable container for tone indicators."""

    positive: frozenset[str]
    negative: frozenset[str]

    def __post_init__(self) -> None:
        if not self.positive and not self.negative:
            raise ConfigurationError("At least one indicator set must be non-empty")


class ToneConsistencyValidator(ContentValidator):
    """Validator that checks tone consistency using _ToneIndicatorAnalyzer."""

    # Default tone indicators for different tone styles
    DEFAULT_TONE_INDICATORS: Final[Dict[str, Dict[str, List[str]]]] = {
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

    def __init__(self, config: RuleConfig, analyzer: Optional[ToneAnalyzer] = None) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultToneAnalyzer())
        self._config: Final[RuleConfig] = config
        self._tone_analyzer = self._validate_tone_analyzer(analyzer or DefaultToneAnalyzer())
        self._indicator_analyzer = _ToneIndicatorAnalyzer(
            indicators=self.tone_indicators, expected_tone=self.expected_tone
        )

    def _validate_tone_analyzer(self, analyzer: Any) -> ToneAnalyzer:
        """Validate and return a tone analyzer."""
        if not isinstance(analyzer, ToneAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ToneAnalyzer protocol, got {type(analyzer)}"
            )
        return analyzer

    @property
    def expected_tone(self) -> str:
        """Get expected tone from config."""
        return self._config.params.get("expected_tone", "neutral")

    @property
    def tone_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """Get tone indicators from config."""
        return self._config.params.get("tone_indicators", self.DEFAULT_TONE_INDICATORS)

    @property
    def threshold(self) -> float:
        """Get threshold from config."""
        return self._config.params.get("threshold", 0.7)

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output maintains a consistent tone."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            if self.expected_tone not in self.tone_indicators:
                return RuleResult(
                    passed=False,
                    message=f"Unknown tone: {self.expected_tone}",
                    metadata={"available_tones": self._tone_analyzer.get_supported_tones()},
                )

            # Analyze tone
            tone_scores = self._tone_analyzer.analyze_tone(output)

            score, found_positive, found_negative = self._indicator_analyzer.analyze(output)
            passed = score >= self.threshold

            return RuleResult(
                passed=passed,
                message=f"Output {'maintains' if passed else 'does not maintain'} {self.expected_tone} tone",
                metadata={
                    "tone_scores": tone_scores,
                    "positive_indicators": found_positive,
                    "negative_indicators": found_negative,
                    "consistency_score": score,
                    "threshold": self.threshold,
                },
            )

        except Exception as e:
            raise ValidationError(f"Tone validation failed: {str(e)}") from e


class DefaultToneValidator(BaseValidator[str]):
    """Default implementation of tone validation."""

    # Default tone indicators for different tone styles
    DEFAULT_TONE_INDICATORS: Final[Dict[str, Dict[str, List[str]]]] = {
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

    DEFAULT_THRESHOLD: Final[float] = 0.7

    def __init__(self, config: RuleConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._analyzer = _ToneIndicatorAnalyzer(
            indicators=self.config.params.get("tone_indicators", self.DEFAULT_TONE_INDICATORS),
            expected_tone=self.config.params.get("expected_tone", "neutral"),
        )

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Delegate validation to _ToneIndicatorAnalyzer and evaluate threshold."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        threshold = self.config.params.get("threshold", self.DEFAULT_THRESHOLD)

        score, pos_ind, neg_ind = self._analyzer.analyze(text)
        passed = score >= threshold

        return RuleResult(
            passed=passed,
            message=(
                f"Text {'maintains' if passed else 'does not maintain'} expected tone"
            ),
            metadata={
                "consistency_score": score,
                "threshold": threshold,
                "positive_indicators": pos_ind,
                "negative_indicators": neg_ind,
            },
        )


class ToneConsistencyRule(
    Rule[str, RuleResult, DefaultToneValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for tone consistency."""

    def __init__(
        self,
        name: str = "tone_consistency_rule",
        description: str = "Checks for tone consistency",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultToneValidator] = None,
    ) -> None:
        """
        Initialize the tone consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Ensure we have a valid configuration
        if config is None:
            config = RuleConfig(
                params={
                    "expected_tone": "neutral",
                    "tone_indicators": DefaultToneValidator.DEFAULT_TONE_INDICATORS,
                    "threshold": DefaultToneValidator.DEFAULT_THRESHOLD,
                    "cache_size": 100,
                    "priority": 1,
                    "cost": 1.0,
                }
            )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
        )

    def _create_default_validator(self) -> DefaultToneValidator:
        """Create a default validator from config."""
        return DefaultToneValidator(self.config)


def create_tone_consistency_validator(
    expected_tone: Optional[str] = None,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> DefaultToneValidator:
    """
    Create a tone consistency validator with the specified configuration.

    This factory function creates a configured tone consistency validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        expected_tone: The expected tone style (formal, informal, neutral)
        tone_indicators: Dictionary mapping tone styles to positive/negative indicators
        threshold: Minimum tone consistency score required (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured tone consistency validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if expected_tone is not None:
        config_params["expected_tone"] = expected_tone
    if tone_indicators is not None:
        config_params["tone_indicators"] = tone_indicators
    if threshold is not None:
        config_params["threshold"] = threshold

    # Set defaults if not provided
    if "expected_tone" not in config_params:
        config_params["expected_tone"] = "neutral"
    if "tone_indicators" not in config_params:
        config_params["tone_indicators"] = DefaultToneValidator.DEFAULT_TONE_INDICATORS
    if "threshold" not in config_params:
        config_params["threshold"] = DefaultToneValidator.DEFAULT_THRESHOLD

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    rule_config = RuleConfig(params=config_params)

    # Return configured validator
    return DefaultToneValidator(rule_config)


def create_tone_consistency_rule(
    name: str = "tone_consistency_rule",
    description: str = "Validates text tone consistency",
    expected_tone: Optional[str] = None,
    tone_indicators: Optional[Dict[str, Dict[str, List[str]]]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> ToneConsistencyRule:
    """
    Create a tone consistency rule with configuration.

    This factory function creates a configured ToneConsistencyRule instance.
    It uses create_tone_consistency_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        expected_tone: The expected tone style (formal, informal, neutral)
        tone_indicators: Dictionary mapping tone styles to positive/negative indicators
        threshold: Minimum tone consistency score required (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured ToneConsistencyRule instance
    """
    # Create validator using the validator factory
    validator = create_tone_consistency_validator(
        expected_tone=expected_tone,
        tone_indicators=tone_indicators,
        threshold=threshold,
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
    "ToneIndicators",
    # Validator classes
    "ToneConsistencyValidator",
    "DefaultToneValidator",
    # Rule classes
    "ToneConsistencyRule",
    # Validator factory functions
    "create_tone_consistency_validator",
    # Rule factory functions
    "create_tone_consistency_rule",
    # Internal helpers
    "_ToneIndicatorAnalyzer",
]

# ---------------------------------------------------------------------------
# Analyzer helper
# ---------------------------------------------------------------------------


class _ToneIndicatorAnalyzer(BaseModel):
    """Compute positive/negative indicator ratios for a given text and tone."""

    indicators: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    expected_tone: str

    def analyze(self, text: str) -> tuple[float, List[str], List[str]]:
        tone_ind = self.indicators[self.expected_tone]
        pos_list = tone_ind["positive"]
        neg_list = tone_ind["negative"]

        lower = text.lower()
        found_pos = [w for w in pos_list if w.lower() in lower]
        found_neg = [w for w in neg_list if w.lower() in lower]

        pos_ratio = len(found_pos) / len(pos_list) if pos_list else 0
        neg_ratio = len(found_neg) / len(neg_list) if neg_list else 0
        score = pos_ratio - neg_ratio
        return score, found_pos, found_neg
