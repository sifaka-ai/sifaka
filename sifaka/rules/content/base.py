"""
Base classes and protocols for content validation.
"""

from typing import Any, Dict, List, Protocol, Optional, runtime_checkable

from pydantic import BaseModel, Field

from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult


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


class ContentValidator(BaseValidator[str]):
    """Base validator for content-based rules."""

    def __init__(self, analyzer: ContentAnalyzer) -> None:
        """Initialize with content analyzer."""
        self._validate_analyzer(analyzer)
        self._analyzer = analyzer

    def _validate_analyzer(self, analyzer: Any) -> bool:
        """Validate that an analyzer implements the required protocol."""
        if not isinstance(analyzer, ContentAnalyzer):
            raise ConfigurationError(
                f"Analyzer must implement ContentAnalyzer protocol, got {type(analyzer)}"
            )
        return True


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


# ---------------------------------------------------------------------------
# Shared analyzer base classes for consistent content rule implementations
# ---------------------------------------------------------------------------


class IndicatorAnalyzerBase(BaseModel):
    """Base analyzer for simple indicator-based rules.

    This can be subclassed for toxicity, sentiment, etc. analyzers that
    check for indicator terms in text.
    """

    indicators: List[str] = Field(default_factory=list)
    threshold: float = 0.5
    higher_is_better: bool = True  # If True, higher score = passing; else lower score = passing

    def analyze(self, text: str) -> RuleResult:
        """Find indicators in text and compute score relative to threshold."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        lower = text.lower()
        found = [ind for ind in self.indicators if ind.lower() in lower]
        score = len(found) / len(self.indicators) if self.indicators else 0.0

        # Determine if passing based on higher_is_better flag
        passed = score >= self.threshold if self.higher_is_better else score <= self.threshold

        message_type = "positive" if self.higher_is_better else "problematic"
        threshold_relation = "above" if self.higher_is_better else "below"

        return RuleResult(
            passed=passed,
            message=(
                f"{message_type.capitalize()} indicators {threshold_relation} threshold"
                if passed
                else f"{message_type.capitalize()} indicators not {threshold_relation} threshold ({score:.2f})"
            ),
            metadata={"found": found, "score": score, "threshold": self.threshold},
        )


class CategoryAnalyzerBase(BaseModel):
    """Base analyzer for category-based rules.

    This can be subclassed for bias, emotion, etc. analyzers that
    check for multiple categories of indicators.
    """

    categories: Dict[str, List[str]] = Field(default_factory=dict)
    threshold: float = 0.3
    fail_if_any: bool = False
    higher_is_better: bool = False  # For bias/harmful content, lower is better

    def analyze(self, text: str) -> RuleResult:
        """Analyze text against multiple indicator categories."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        lower = text.lower()
        scores: Dict[str, float] = {}
        found_categories: Dict[str, List[str]] = {}

        for cat, indicators in self.categories.items():
            found = [ind for ind in indicators if ind.lower() in lower]
            score = len(found) / len(indicators) if indicators else 0.0
            scores[cat] = score

            # In fail_if_any mode, any indicator is a failure
            # Otherwise compare score to threshold
            if self.higher_is_better:
                threshold_test = score < self.threshold
            else:
                threshold_test = score > self.threshold

            should_fail = (self.fail_if_any and found) or threshold_test

            if should_fail:
                found_categories[cat] = found

        passed = not found_categories
        message_type = "acceptable" if not self.higher_is_better else "detected"

        return RuleResult(
            passed=passed,
            message=("Content " + message_type if passed else "Problematic content detected"),
            metadata={
                "scores": scores,
                "found_categories": found_categories,
                "threshold": self.threshold,
            },
        )
