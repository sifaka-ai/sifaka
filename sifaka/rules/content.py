"""
Content validation rules for Sifaka.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing_extensions import TypeGuard

from sifaka.rules.base import (
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    RuleValidator,
    ValidationError,
)

@runtime_checkable
class ContentAnalyzer(Protocol):
    """Protocol for content analysis components."""

    def analyze(self, text: str) -> Dict[str, Any]: ...
    def can_analyze(self, text: str) -> bool: ...

@runtime_checkable
class ToneAnalyzer(Protocol):
    """Protocol for tone analysis components."""

    def analyze_tone(self, text: str) -> Dict[str, float]: ...
    def get_supported_tones(self) -> List[str]: ...

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

@dataclass(frozen=True)
class ToneIndicators:
    """Immutable container for tone indicators."""

    positive: frozenset[str]
    negative: frozenset[str]

    def __post_init__(self) -> None:
        if not self.positive and not self.negative:
            raise ConfigurationError("At least one indicator set must be non-empty")

@dataclass(frozen=True)
class ProhibitedContentConfig(RuleConfig):
    """Configuration for prohibited content validation."""

    terms: List[str] = field(
        default_factory=lambda: [
            "profanity",
            "obscenity",
            "hate speech",
            "explicit content",
            "adult content",
            "nsfw",
            "inappropriate",
        ]
    )
    case_sensitive: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.terms:
            raise ValueError("Must provide at least one prohibited term")

@dataclass(frozen=True)
class ToneConfig(RuleConfig):
    """Configuration for tone consistency validation."""

    expected_tone: str = "neutral"
    tone_indicators: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: {
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
    )
    threshold: float = 0.7
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.tone_indicators:
            raise ValueError("Must provide at least one tone indicator")
        if self.expected_tone not in self.tone_indicators:
            raise ValueError(f"Expected tone {self.expected_tone} not found in indicators")
        for tone, indicators in self.tone_indicators.items():
            if "positive" not in indicators or "negative" not in indicators:
                raise ValueError(f"Tone {tone} must have both positive and negative indicators")
            if not indicators["positive"] or not indicators["negative"]:
                raise ValueError(
                    f"Tone {tone} must have non-empty positive and negative indicators"
                )

class ContentValidator(RuleValidator[str]):
    """Base validator for content-based rules."""

    def __init__(self, analyzer: ContentAnalyzer) -> None:
        """Initialize with content analyzer."""
        self._validate_analyzer(analyzer)
        self._analyzer: Final[ContentAnalyzer] = analyzer

    def _validate_analyzer(self, analyzer: Any) -> TypeGuard[ContentAnalyzer]:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, ContentAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ContentAnalyzer protocol, got {type(analyzer)}"
            )
        return True

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator accepts."""
        return str

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str) and self._analyzer.can_analyze(output)

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

class ToneConsistencyValidator(ContentValidator):
    """Validator that checks tone consistency."""

    def __init__(
        self,
        config: ToneConfig,
        analyzer: Optional[ToneAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultToneAnalyzer())
        self._config: Final[ToneConfig] = config
        self._tone_analyzer = self._validate_tone_analyzer(analyzer or DefaultToneAnalyzer())

    def _validate_tone_analyzer(self, analyzer: Any) -> ToneAnalyzer:
        """Validate and return a tone analyzer."""
        if not isinstance(analyzer, ToneAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ToneAnalyzer protocol, got {type(analyzer)}"
            )
        return analyzer

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output maintains a consistent tone."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            if self._config.expected_tone not in self._config.tone_indicators:
                return RuleResult(
                    passed=False,
                    message=f"Unknown tone: {self._config.expected_tone}",
                    metadata={"available_tones": self._tone_analyzer.get_supported_tones()},
                )

            # Analyze tone
            tone_scores = self._tone_analyzer.analyze_tone(output)
            indicators = self._config.tone_indicators[self._config.expected_tone]

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

            passed = consistency_score >= self._config.threshold

            return RuleResult(
                passed=passed,
                message=f"Output {'maintains' if passed else 'does not maintain'} {self._config.expected_tone} tone",
                metadata={
                    "tone_scores": tone_scores,
                    "positive_indicators": found_positive,
                    "negative_indicators": found_negative,
                    "consistency_score": consistency_score,
                    "threshold": self._config.threshold,
                },
            )

        except Exception as e:
            raise ValidationError(f"Tone validation failed: {str(e)}") from e

class DefaultContentAnalyzer:
    """Default implementation of ContentAnalyzer."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """Basic content analysis."""
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "has_content": bool(text.strip()),
        }

    def can_analyze(self, text: str) -> bool:
        """Check if text can be analyzed."""
        return isinstance(text, str)

class DefaultToneAnalyzer:
    """Default implementation of ToneAnalyzer."""

    def analyze_tone(self, text: str) -> Dict[str, float]:
        """Basic tone analysis."""
        text_lower = text.lower()
        words = text_lower.split()
        return {
            "formality": sum(1 for w in words if len(w) > 6) / len(words) if words else 0,
            "complexity": len(set(words)) / len(words) if words else 0,
        }

    def get_supported_tones(self) -> List[str]:
        """Get list of supported tones."""
        return ["formal", "informal", "technical", "casual"]

class DefaultProhibitedContentValidator(RuleValidator[str]):
    """Default implementation of prohibited content validation."""

    def __init__(self, config: ProhibitedContentConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> ProhibitedContentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for prohibited content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        check_text = text if self.config.case_sensitive else text.lower()
        found_terms = []

        for term in self.config.terms:
            check_term = term if self.config.case_sensitive else term.lower()
            if check_term in check_text:
                found_terms.append(term)

        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Found prohibited terms: {', '.join(found_terms)}",
                metadata={
                    "found_terms": found_terms,
                    "case_sensitive": self.config.case_sensitive,
                },
            )

        return RuleResult(
            passed=True,
            message="No prohibited terms found",
            metadata={
                "found_terms": [],
                "case_sensitive": self.config.case_sensitive,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class DefaultToneValidator(RuleValidator[str]):
    """Default implementation of tone validation."""

    def __init__(self, config: ToneConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> ToneConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for tone consistency."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        tone_scores: Dict[str, Dict[str, float]] = {}

        # Calculate tone scores for each tone type
        for tone, indicators in self.config.tone_indicators.items():
            positive_matches = sum(1 for term in indicators["positive"] if term in text_lower)
            negative_matches = sum(1 for term in indicators["negative"] if term in text_lower)

            len(indicators["positive"]) + len(indicators["negative"])
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
        expected_score = tone_scores[self.config.expected_tone]["overall"]
        meets_threshold = expected_score >= self.config.threshold

        if not meets_threshold:
            return RuleResult(
                passed=False,
                message=f"Text does not maintain expected {self.config.expected_tone} tone (score: {expected_score:.2f})",
                metadata={
                    "tone_scores": tone_scores,
                    "expected_tone": self.config.expected_tone,
                    "threshold": self.config.threshold,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Text maintains expected {self.config.expected_tone} tone",
            metadata={
                "tone_scores": tone_scores,
                "expected_tone": self.config.expected_tone,
                "threshold": self.config.threshold,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class ProhibitedContentRule(
    Rule[str, RuleResult, ProhibitedContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for prohibited content in the output."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the prohibited content rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        prohibited_config = ProhibitedContentConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultProhibitedContentValidator(prohibited_config)

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            validator=validator,
            config=prohibited_config,
            result_handler=None,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate output for prohibited content."""
        return self._validator.validate(output)

class ToneConsistencyRule(
    Rule[str, RuleResult, ToneConsistencyValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for tone consistency."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the tone consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        tone_config = ToneConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultToneValidator(tone_config)

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            validator=validator,
            config=tone_config,
            result_handler=None,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate output tone consistency."""
        return self._validator.validate(output)

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
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return ProhibitedContentRule(
        name=name,
        description=description,
        config=config,
    )

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
            "tone_indicators": {
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
            },
            "threshold": 0.7,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return ToneConsistencyRule(
        name=name,
        description=description,
        config=config,
    )

# Export public classes and functions
__all__ = [
    "ProhibitedContentRule",
    "ProhibitedContentConfig",
    "DefaultProhibitedContentValidator",
    "ToneConsistencyRule",
    "ToneConfig",
    "DefaultToneValidator",
    "create_prohibited_content_rule",
    "create_tone_consistency_rule",
]
