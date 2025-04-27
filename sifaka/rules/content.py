"""
Content validation rules for Sifaka.
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, TypeVar, Final, Sequence
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import (
    Rule,
    RuleResult,
    RuleConfig,
    RuleValidator,
    RuleResultHandler,
    ValidationError,
    ConfigurationError,
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
class ToneConfig(RuleConfig):
    """Immutable configuration for tone consistency."""

    expected_tone: str
    indicators: Dict[str, ToneIndicators]
    threshold: float = 0.7

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.threshold <= 1:
            raise ConfigurationError("Threshold must be between 0 and 1")
        if not self.expected_tone:
            raise ConfigurationError("Expected tone cannot be empty")
        if not self.indicators:
            raise ConfigurationError("Tone indicators cannot be empty")

    def with_tone(self, tone: str) -> "ToneConfig":
        """Create new config with updated tone."""
        return self.with_options(expected_tone=tone)

    def with_threshold(self, threshold: float) -> "ToneConfig":
        """Create new config with updated threshold."""
        return self.with_options(threshold=threshold)


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

            if self._config.expected_tone not in self._config.indicators:
                return RuleResult(
                    passed=False,
                    message=f"Unknown tone: {self._config.expected_tone}",
                    metadata={"available_tones": self._tone_analyzer.get_supported_tones()},
                )

            # Analyze tone
            tone_scores = self._tone_analyzer.analyze_tone(output)
            indicators = self._config.indicators[self._config.expected_tone]

            # Check indicators
            output_lower = output.lower()
            found_positive = [ind for ind in indicators.positive if ind.lower() in output_lower]
            found_negative = [ind for ind in indicators.negative if ind.lower() in output_lower]

            # Calculate consistency score
            total_positive = len(indicators.positive)
            total_negative = len(indicators.negative)

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


class ProhibitedContentRule(
    Rule[str, RuleResult, ProhibitedContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for prohibited content in the output."""

    def __init__(
        self,
        name: str,
        description: str,
        terms: Sequence[str],
        case_sensitive: bool = False,
        analyzer: Optional[ContentAnalyzer] = None,
        config: Optional[RuleConfig] = None,
        result_handler: Optional[RuleResultHandler[RuleResult]] = None,
    ) -> None:
        """Initialize the prohibited content rule."""
        prohibited_terms = ProhibitedTerms(
            terms=frozenset(terms),
            case_sensitive=case_sensitive,
        )
        validator = ProhibitedContentValidator(
            terms=prohibited_terms,
            analyzer=analyzer,
        )
        super().__init__(
            name=name,
            description=description,
            validator=validator,
            config=config or RuleConfig(),
            result_handler=result_handler,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output does not contain prohibited terms."""
        return self._validator.validate(output, **kwargs)


class ToneConsistencyRule(
    Rule[str, RuleResult, ToneConsistencyValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks if the output maintains a consistent tone."""

    def __init__(
        self,
        name: str,
        description: str,
        expected_tone: str,
        indicators: Dict[str, Dict[str, List[str]]],
        threshold: float = 0.7,
        analyzer: Optional[ToneAnalyzer] = None,
        config: Optional[RuleConfig] = None,
        result_handler: Optional[RuleResultHandler[RuleResult]] = None,
    ) -> None:
        """Initialize the tone consistency rule."""
        # Convert indicators to immutable format
        tone_indicators = {
            tone: ToneIndicators(
                positive=frozenset(inds.get("positive", [])),
                negative=frozenset(inds.get("negative", [])),
            )
            for tone, inds in indicators.items()
        }

        tone_config = ToneConfig(
            expected_tone=expected_tone,
            indicators=tone_indicators,
            threshold=threshold,
            priority=config.priority if config else RuleConfig().priority,
            cache_size=config.cache_size if config else RuleConfig().cache_size,
            cost=config.cost if config else RuleConfig().cost,
            metadata=config.metadata if config else RuleConfig().metadata,
        )

        validator = ToneConsistencyValidator(
            config=tone_config,
            analyzer=analyzer,
        )

        super().__init__(
            name=name,
            description=description,
            validator=validator,
            config=tone_config,
            result_handler=result_handler,
        )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output maintains a consistent tone."""
        return self._validator.validate(output, **kwargs)
