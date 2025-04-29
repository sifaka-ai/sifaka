"""
Tone consistency validation rules for Sifaka.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional

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
    """Validator that checks tone consistency."""

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

    def __init__(
        self,
        config: RuleConfig,
        analyzer: Optional[ToneAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultToneAnalyzer())
        self._config: Final[RuleConfig] = config
        self._tone_analyzer = self._validate_tone_analyzer(analyzer or DefaultToneAnalyzer())

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
            indicators = self.tone_indicators[self.expected_tone]

            # Check indicators
            output_lower = output.lower()
            found_positive = [ind for ind in indicators["positive"] if ind.lower() in output_lower]
            found_negative = [ind for ind in indicators["negative"] if ind.lower() in output_lower]

            # Calculate consistency score
            total_positive = len(indicators["positive"])
            total_negative = len(indicators["negative"])

            positive_ratio = len(found_positive) / total_positive if total_positive > 0 else 0
            negative_ratio = len(found_negative) / total_negative if total_negative > 0 else 0
            consistency_score = positive_ratio - negative_ratio

            passed = consistency_score >= self.threshold

            return RuleResult(
                passed=passed,
                message=f"Output {'maintains' if passed else 'does not maintain'} {self.expected_tone} tone",
                metadata={
                    "tone_scores": tone_scores,
                    "positive_indicators": found_positive,
                    "negative_indicators": found_negative,
                    "consistency_score": consistency_score,
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

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for tone consistency."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        tone_indicators = self.config.params.get("tone_indicators", self.DEFAULT_TONE_INDICATORS)
        expected_tone = self.config.params.get("expected_tone", "neutral")
        threshold = self.config.params.get("threshold", self.DEFAULT_THRESHOLD)

        text_lower = text.lower()
        tone_scores: Dict[str, Dict[str, float]] = {}

        # Calculate tone scores for each tone type
        for tone, indicators in tone_indicators.items():
            positive_matches = sum(1 for term in indicators["positive"] if term in text_lower)
            negative_matches = sum(1 for term in indicators["negative"] if term in text_lower)

            total_indicators = len(indicators["positive"]) + len(indicators["negative"])
            positive_score = (
                positive_matches / len(indicators["positive"]) if indicators["positive"] else 0
            )
            negative_score = (
                negative_matches / len(indicators["negative"]) if indicators["negative"] else 0
            )

            tone_scores[tone] = {
                "positive": positive_score,
                "negative": negative_score,
                "overall": (positive_score - negative_score + 1) / 2,  # Normalize to [0,1]
            }

        # Check if the expected tone meets the threshold
        expected_score = tone_scores[expected_tone]["overall"]
        meets_threshold = expected_score >= threshold

        if not meets_threshold:
            return RuleResult(
                passed=False,
                message=f"Text does not maintain expected {expected_tone} tone (score: {expected_score:.2f})",
                metadata={
                    "tone_scores": tone_scores,
                    "expected_tone": expected_tone,
                    "threshold": threshold,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Text maintains expected {expected_tone} tone",
            metadata={
                "tone_scores": tone_scores,
                "expected_tone": expected_tone,
                "threshold": threshold,
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


def create_tone_consistency_rule(
    name: str = "tone_consistency_rule",
    description: str = "Validates text tone consistency",
    config: Optional[Dict[str, Any]] = None,
) -> ToneConsistencyRule:
    """
    Create a tone consistency rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured ToneConsistencyRule instance
    """
    if config is None:
        config = {
            "expected_tone": "neutral",
            "tone_indicators": DefaultToneValidator.DEFAULT_TONE_INDICATORS,
            "threshold": DefaultToneValidator.DEFAULT_THRESHOLD,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config)

    return ToneConsistencyRule(
        name=name,
        description=description,
        config=rule_config,
    )
