"""Base node class for Sifaka graph nodes.

This module provides a common base class for all Sifaka graph nodes
with shared utilities and helper methods.
"""

from typing import Any

from pydantic_graph import BaseNode

from sifaka.core.thought import SifakaThought


class SifakaNode(BaseNode[SifakaThought, Any, SifakaThought]):
    """Base class for all Sifaka graph nodes.
    
    This class provides a common base for all nodes in the Sifaka workflow
    and includes shared utility methods for context building and feedback processing.
    """
    
    def _get_prominence_level(self, weight: float) -> str:
        """Get prominence indicator based on weight.

        Args:
            weight: Weight value between 0.0 and 1.0

        Returns:
            Prominence indicator string
        """
        if weight >= 0.7:
            return "🔥 CRITICAL"
        elif weight >= 0.5:
            return "⚠️ IMPORTANT"
        elif weight >= 0.3:
            return "📝 MODERATE"
        else:
            return "💡 MINOR"
    
    def _extract_suggestions(self, feedback: str) -> list[str]:
        """Extract actionable suggestions from critic feedback.

        Args:
            feedback: The raw feedback text from the critic

        Returns:
            List of extracted suggestions
        """
        if "NO_IMPROVEMENTS_NEEDED" in feedback:
            return []

        # Simple extraction logic - look for bullet points or numbered lists
        suggestions = []
        lines = feedback.split("\n")

        for line in lines:
            line = line.strip()
            # Look for bullet points, numbers, or suggestion keywords
            if (
                line.startswith("-")
                or line.startswith("*")
                or line.startswith("•")
                or any(line.startswith(f"{i}.") for i in range(1, 10))
                or "suggest" in line.lower()
                or "recommend" in line.lower()
                or "improve" in line.lower()
            ):

                # Clean up the suggestion text
                cleaned = line.lstrip("-*•0123456789. ").strip()
                if cleaned and len(cleaned) > 10:  # Ignore very short suggestions
                    suggestions.append(cleaned)

        # Limit to 3 suggestions per critic
        return suggestions[:3]
