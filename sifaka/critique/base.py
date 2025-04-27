"""
Base class for critique functionality.
"""

from typing import Dict, Any
from pydantic import BaseModel, ConfigDict


class Critique(BaseModel):
    """Base class for critiquing and validating prompts."""

    name: str
    description: str
    config: Dict[str, Any] = {}
    min_confidence: float = 0.7

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def validate(self, prompt: str) -> bool:
        """
        Validate if a prompt meets quality standards.

        Args:
            prompt: The prompt to validate

        Returns:
            True if prompt meets standards, False otherwise
        """
        raise NotImplementedError()

    def critique(self, prompt: str) -> Dict[str, Any]:
        """
        Critique a prompt and provide feedback.

        Args:
            prompt: The prompt to critique

        Returns:
            Dictionary containing critique results
        """
        raise NotImplementedError()
