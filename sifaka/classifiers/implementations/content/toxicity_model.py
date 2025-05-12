"""
Protocol for toxicity detection models.
"""

from typing import Dict, List, Protocol, runtime_checkable, Union
import numpy as np


@runtime_checkable
class ToxicityModel(Protocol):
    """Protocol for toxicity detection models."""

    def predict(self, text: Union[str, List[str]]) -> Dict[str, np.Union[ndarray, float]]:
        """
        Predict toxicity scores for the given text(s).

        Args:
            text: Single text string or list of text strings to analyze

        Returns:
            Dictionary containing toxicity scores with the following keys:
            - 'toxicity': Overall toxicity score (float or array)
            - 'severe_toxicity': Score for severe toxicity (float or array)
            - 'obscene': Score for obscene content (float or array)
            - 'threat': Score for threatening content (float or array)
            - 'insult': Score for insulting content (float or array)
            - 'identity_attack': Score for identity-based attacks (float or array)

            If input is a single string, values are floats.
            If input is a list of strings, values are numpy arrays.

        Raises:
            ValueError: If input text is empty or invalid
            RuntimeError: If model fails to make predictions
        """
        ...
