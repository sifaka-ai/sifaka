"""
Text improvement module for Sifaka.

This module provides the Improver class which is responsible for
improving text using model providers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from sifaka.models.base import ModelProvider
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)
OutputType = TypeVar("OutputType")


@dataclass
class ImprovementResult(Generic[OutputType]):
    """
    Result of an improvement operation.

    This class represents the result of an improvement operation, including
    the improved output, confidence score, and additional metadata.
    """

    output: OutputType
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Improver(Generic[OutputType]):
    """
    Handles text improvement using model providers.

    This class is responsible for improving text using model providers.
    It provides a consistent interface for text improvement across different
    model providers.
    """

    def __init__(self, model: ModelProvider) -> None:
        """
        Initialize an Improver instance.

        Args:
            model: The model provider to use for text improvement
        """
        self._model = model

    def improve(self, text: str, feedback: str) -> Any:
        """
        Improve text using the model.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            TypeError: If text or feedback is not a string
            ValueError: If text or feedback is empty
            RuntimeError: If improvement fails
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text cannot be empty")
        if not isinstance(feedback, str):
            raise TypeError("feedback must be a string")
        if not feedback.strip():
            raise ValueError("feedback cannot be empty")
        try:
            prompt = f"Original text: {text}\n\nFeedback: {feedback}\n\nImproved text:"
            improved_text = self._model.generate(prompt)
            return ImprovementResult[str](
                output=improved_text,
                confidence=0.9,
                metadata={
                    "model": self._model.name if hasattr(self._model, "name") else str(self._model)
                },
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error improving text: {error_msg}")
            raise RuntimeError(f"Error improving text: {error_msg}") from e
