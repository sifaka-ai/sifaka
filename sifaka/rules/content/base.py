"""
Base classes and protocols for content validation.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable

from sifaka.rules.base import BaseValidator, ConfigurationError


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
