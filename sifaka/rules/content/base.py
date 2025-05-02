"""
Base classes and protocols for content validation.

This module provides the foundation for content-based validation rules in Sifaka.
It defines protocols for analyzers and base classes for validators and analyzers.
"""

from typing import Any, Dict, List, Optional, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult


class ContentAnalysis(BaseModel):
    """Model for content analysis results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    length: int = Field(description="Length of the content in characters")
    word_count: int = Field(description="Number of words in the content")
    has_content: bool = Field(description="Whether the content has any meaningful text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis",
    )


class ToneAnalysis(BaseModel):
    """Model for tone analysis results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tone: str = Field(description="The detected tone of the content")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the tone detection",
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="List of tone indicators found in the content",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the analysis",
    )


@runtime_checkable
class ContentAnalyzer(Protocol):
    """Protocol for content analysis components."""

    def analyze(self, text: str) -> ContentAnalysis:
        """Analyze text and return structured results."""
        ...

    def can_analyze(self, text: str) -> bool:
        """Check if text can be analyzed."""
        ...


@runtime_checkable
class ToneAnalyzer(Protocol):
    """Protocol for tone analysis components."""

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """Analyze text tone and return structured results."""
        ...

    def get_supported_tones(self) -> List[str]:
        """Get list of supported tone categories."""
        ...


class ContentValidator(BaseValidator[str]):
    """Base validator for content-based rules."""

    def __init__(self, analyzer: ContentAnalyzer) -> None:
        """Initialize with content analyzer."""
        self._validate_analyzer(analyzer)
        self._analyzer = analyzer

    def _validate_analyzer(self, analyzer: Any) -> None:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, ContentAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ContentAnalyzer protocol, got {type(analyzer)}"
            )


class DefaultContentAnalyzer(BaseModel):
    """Default implementation of ContentAnalyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def analyze(self, text: str) -> ContentAnalysis:
        """Basic content analysis."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = text.split()
        return {
            "length": len(text),
            "word_count": len(words),
            "has_content": bool(text.strip()),
            "metadata": {
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "unique_words": len(set(words)) if words else 0,
            },
        }

    def can_analyze(self, text: str) -> bool:
        """Check if text can be analyzed."""
        return isinstance(text, str)


class DefaultToneAnalyzer(BaseModel):
    """Default implementation of ToneAnalyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def analyze_tone(self, text: str) -> ToneAnalysis:
        """Basic tone analysis."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = text.lower().split()
        unique_words = set(words)
        return {
            "formality": sum(1 for w in words if len(w) > 6) / len(words) if words else 0,
            "complexity": len(unique_words) / len(words) if words else 0,
            "metadata": {
                "unique_words": len(unique_words),
                "total_words": len(words),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            },
        }

    def get_supported_tones(self) -> List[str]:
        """Get list of supported tones."""
        return ["formal", "informal", "technical", "casual"]


class IndicatorAnalyzer(BaseModel):
    """Base analyzer for indicator-based rules.

    This can be used for toxicity, sentiment, etc. analyzers that
    check for indicator terms in text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicators: List[str] = Field(default_factory=list)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    higher_is_better: bool = Field(default=True)
    case_sensitive: bool = Field(default=False)

    def analyze(self, text: str) -> RuleResult:
        """Find indicators in text and compute score relative to threshold."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_to_check = text if self.case_sensitive else text.lower()
        indicators_to_check = (
            self.indicators if self.case_sensitive else [i.lower() for i in self.indicators]
        )

        found = [ind for ind in indicators_to_check if ind in text_to_check]
        score = len(found) / len(indicators_to_check) if indicators_to_check else 0.0
        passed = score >= self.threshold if self.higher_is_better else score <= self.threshold

        return RuleResult(
            passed=passed,
            message=(
                f"Score ({score:.2f}) {'meets' if passed else 'does not meet'} "
                f"threshold ({self.threshold})"
            ),
            metadata={
                "score": score,
                "found_indicators": found,
                "total_indicators": len(indicators_to_check),
                "threshold": self.threshold,
                "higher_is_better": self.higher_is_better,
            },
        )


class CategoryAnalyzer(BaseModel):
    """Base analyzer for category-based rules.

    This can be used for bias, emotion, etc. analyzers that
    check for multiple categories of indicators.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    categories: Dict[str, List[str]] = Field(default_factory=dict)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    fail_if_any: bool = Field(default=False)
    higher_is_better: bool = Field(default=False)
    case_sensitive: bool = Field(default=False)

    def analyze(self, text: str) -> RuleResult:
        """Find category indicators in text and compute scores."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_to_check = text if self.case_sensitive else text.lower()
        scores = {}
        found_indicators = {}

        for category, indicators in self.categories.items():
            indicators_to_check = (
                indicators if self.case_sensitive else [i.lower() for i in indicators]
            )
            found = [ind for ind in indicators_to_check if ind in text_to_check]
            score = len(found) / len(indicators_to_check) if indicators_to_check else 0.0
            scores[category] = score
            found_indicators[category] = found

        if self.fail_if_any:
            passed = all(
                score <= self.threshold if self.higher_is_better else score >= self.threshold
                for score in scores.values()
            )
        else:
            passed = any(
                score <= self.threshold if self.higher_is_better else score >= self.threshold
                for score in scores.values()
            )

        return RuleResult(
            passed=passed,
            message=(
                f"Category scores {'all' if self.fail_if_any else 'any'} "
                f"{'meet' if passed else 'do not meet'} threshold ({self.threshold})"
            ),
            metadata={
                "scores": scores,
                "found_indicators": found_indicators,
                "threshold": self.threshold,
                "fail_if_any": self.fail_if_any,
                "higher_is_better": self.higher_is_better,
            },
        )
