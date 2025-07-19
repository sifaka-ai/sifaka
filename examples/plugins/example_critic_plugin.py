"""Example critic plugin for Sifaka.

This is a complete, working example of how to create a critic plugin.
Copy this file and modify it to create your own critic plugin.
"""

import logging
from typing import Any, Dict

from sifaka.core.models import CritiqueResult, SifakaResult
from sifaka.core.plugin_interfaces import CriticPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class ExampleCriticPlugin(CriticPlugin):
    """Example critic plugin that checks text readability.

    This critic analyzes text for basic readability issues like:
    - Text length
    - Sentence complexity
    - Word repetition
    """

    def __init__(self) -> None:
        """Initialize the example critic plugin."""
        super().__init__()
        self._min_words = 20
        self._max_sentence_length = 30

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="example_critic",
            version="1.0.0",
            author="Sifaka Team",
            description="Example critic plugin for readability analysis",
            plugin_type=PluginType.CRITIC,
            dependencies=[],
            sifaka_version=">=0.1.0",
            python_version=">=3.10",
            license="MIT",
            keywords=["example", "readability", "critic", "sifaka"],
            default_config={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000,
                "min_words": 20,
                "max_sentence_length": 30,
            },
        )

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Analyze text for readability issues.

        Args:
            text: The text to analyze
            result: The complete SifakaResult with history

        Returns:
            CritiqueResult with readability feedback
        """
        try:
            issues = []
            suggestions = []

            # Check text length
            words = text.split()
            if len(words) < self._min_words:
                issues.append(
                    f"Text has only {len(words)} words (minimum: {self._min_words})"
                )
                suggestions.append(
                    "Consider adding more detail and examples to expand the text."
                )

            # Check sentence complexity
            sentences = text.split(".")
            long_sentences = [
                s for s in sentences if len(s.split()) > self._max_sentence_length
            ]
            if long_sentences:
                issues.append(f"Found {len(long_sentences)} overly long sentences")
                suggestions.append(
                    "Break up long sentences into shorter, clearer ones."
                )

            # Check for repeated words
            word_counts = {}
            for word in words:
                word_lower = word.lower().strip(".,!?;:")
                if len(word_lower) > 3:  # Only check longer words
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

            repeated_words = [word for word, count in word_counts.items() if count > 3]
            if repeated_words:
                issues.append(f"Repeated words: {', '.join(repeated_words[:3])}")
                suggestions.append("Vary your vocabulary to avoid repetition.")

            # Determine if improvement is needed
            needs_improvement = len(issues) > 0
            confidence = (
                0.9 if len(issues) == 0 else max(0.5, 1.0 - (len(issues) * 0.2))
            )

            # Create feedback
            if needs_improvement:
                feedback = (
                    f"Found {len(issues)} readability issues: {'; '.join(issues)}"
                )
            else:
                feedback = "Text meets readability standards."

            return CritiqueResult(
                critic=self.name,
                feedback=feedback,
                suggestions=suggestions,
                needs_improvement=needs_improvement,
                confidence=confidence,
                metadata={
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "issues_found": len(issues),
                    "analysis_type": "readability",
                },
            )

        except Exception as e:
            logger.error(f"Error in example critic: {e}")
            return CritiqueResult(
                critic=self.name,
                feedback=f"Error during readability analysis: {str(e)}",
                suggestions=["Please check the input text and try again."],
                needs_improvement=False,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        if "min_words" in config:
            if not isinstance(config["min_words"], int) or config["min_words"] < 1:
                raise ValueError("min_words must be a positive integer")

        if "max_sentence_length" in config:
            if (
                not isinstance(config["max_sentence_length"], int)
                or config["max_sentence_length"] < 5
            ):
                raise ValueError("max_sentence_length must be at least 5")

        return True

    def _on_initialize(self) -> None:
        """Initialize plugin-specific settings."""
        self._min_words = self.config.get("min_words", 20)
        self._max_sentence_length = self.config.get("max_sentence_length", 30)

        logger.info(
            f"ExampleCriticPlugin initialized: min_words={self._min_words}, max_sentence_length={self._max_sentence_length}"
        )


# Example usage (for testing - remove in production)
if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    async def test_plugin():
        """Test the example critic plugin."""
        plugin = ExampleCriticPlugin()
        plugin.initialize()
        plugin.activate()

        # Test with short text
        short_text = "This is too short."
        result = SifakaResult(
            id="test",
            original_text=short_text,
            final_text=short_text,
            iteration=1,
            processing_time=0.1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            generations=[],
            critiques=[],
            validations=[],
        )

        critique = await plugin.critique(short_text, result)
        print(f"Critique: {critique.feedback}")
        print(f"Suggestions: {critique.suggestions}")
        print(f"Needs improvement: {critique.needs_improvement}")
        print(f"Confidence: {critique.confidence}")

    asyncio.run(test_plugin())
