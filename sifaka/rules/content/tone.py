"""
Tone validation rules.

This module provides validation rules for analyzing and enforcing tone requirements
in text content, including formal/informal tone detection and sentiment analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sifaka.rules.base import RuleConfig, RuleResult
from .base import BaseContentValidator
from .analyzers import ToneAnalyzer


@dataclass
class ToneConfig(RuleConfig):
    """Configuration for tone validation."""

    # Core settings
    required_tone: str = "neutral"  # One of: positive, negative, neutral
    min_confidence: float = 0.5
    tone_threshold: float = 0.3

    # Customization
    boost_words: Set[str] = field(default_factory=set)
    dampen_words: Set[str] = field(default_factory=set)

    # Performance settings
    cache_size: int = 1000
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        super().__post_init__()
        if self.required_tone not in {"positive", "negative", "neutral"}:
            raise ValueError("required_tone must be one of: positive, negative, neutral")

        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        if not 0 <= self.tone_threshold <= 1:
            raise ValueError("tone_threshold must be between 0 and 1")

        # For consistency, copy configuration values to params
        if not self.params:
            object.__setattr__(
                self,
                "params",
                {
                    "required_tone": self.required_tone,
                    "min_confidence": self.min_confidence,
                    "tone_threshold": self.tone_threshold,
                    "boost_words": self.boost_words,
                    "dampen_words": self.dampen_words,
                    "cache_size": self.cache_size,
                    "priority": self.priority,
                    "cost": self.cost,
                },
            )

    def add_boost_words(self, words: Set[str]) -> None:
        """Add words that boost tone scores."""
        self.boost_words.update(words)
        if "boost_words" in self.params:
            self.params["boost_words"].update(words)

    def add_dampen_words(self, words: Set[str]) -> None:
        """Add words that dampen tone scores."""
        self.dampen_words.update(words)
        if "dampen_words" in self.params:
            self.params["dampen_words"].update(words)


class ToneValidator(BaseContentValidator):
    """Validator for text tone."""

    def __init__(self, config: Optional[ToneConfig] = None) -> None:
        """Initialize validator with configuration."""
        super().__init__(config or ToneConfig())
        self._analyzer = ToneAnalyzer(config=self.config)

    def validate(self, content: str, **kwargs) -> RuleResult:
        """Validate content tone."""
        if not self.can_validate(content):
            return RuleResult(
                passed=False,
                message="Invalid content",
                metadata={"error": "Content must be a non-empty string"},
            )

        # Analyze tone
        scores = self._analyzer.analyze(content)
        required_tone = self.config.params["required_tone"]
        min_confidence = self.config.params["min_confidence"]
        tone_threshold = self.config.params["tone_threshold"]

        # Determine dominant tone
        if abs(scores["compound"]) < tone_threshold:
            dominant_tone = "neutral"
        else:
            dominant_tone = "positive" if scores["compound"] > 0 else "negative"

        # Calculate confidence
        if dominant_tone == "neutral":
            confidence = 1.0 - abs(scores["compound"])
        else:
            confidence = abs(scores["compound"])

        # Check if tone matches requirement
        matches_required = dominant_tone == required_tone and confidence >= min_confidence

        return RuleResult(
            passed=matches_required,
            message=(
                f"Content tone '{dominant_tone}' "
                f"{'matches' if matches_required else 'does not match'} "
                f"required tone '{required_tone}' "
                f"(confidence: {confidence:.2f})"
            ),
            metadata={
                "analysis": scores,
                "dominant_tone": dominant_tone,
                "confidence": confidence,
            },
        )

    def get_validation_errors(self, content: str) -> List[str]:
        """Get list of validation errors."""
        if not self.can_validate(content):
            return ["Content must be a non-empty string"]

        result = self.validate(content)
        if result.passed:
            return []

        return [result.message]
