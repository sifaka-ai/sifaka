"""
Protocol for toxicity detection models.
"""

from typing import Dict, List, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class ToxicityModel(Protocol):
    """Protocol for toxicity detection models."""

    def predict(self, text: str | List[str]) -> Dict[str, np.ndarray | float]:
        """
        Predict toxicity scores for the given text(s).

        Args:
            text: Single text string or list of text strings to analyze

        Returns:
            Dictionary mapping toxicity categories to scores
        """
        ...