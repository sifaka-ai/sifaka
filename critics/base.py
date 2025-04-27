from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Annotated
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from pydantic.functional_validators import BeforeValidator
from copy import deepcopy


def validate_dict(v: Any) -> Dict[str, Any]:
    """Validate that the value is a dictionary."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValidationError(
            [
                {
                    "loc": ("config",),
                    "msg": f"Config must be a dictionary, got {type(v).__name__}",
                    "type": "type_error.dict",
                }
            ],
            validate_dict,
        )
    return v


class Critic(BaseModel, ABC):
    """Base class for all critics."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    config: Annotated[Dict[str, Any], BeforeValidator(validate_dict)] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return string representation of the critic."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Return detailed string representation of the critic."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}', min_confidence={self.min_confidence}, config={self.config})"

    @abstractmethod
    def validate(self, text: str) -> bool:
        """Validate the given text.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text is valid, False otherwise
        """
        pass

    @abstractmethod
    def critique(self, text: str) -> Dict[str, Any]:
        """Critique the given text.

        Args:
            text: The text to critique

        Returns:
            Dict[str, Any]: Dictionary containing critique results with keys:
                - score (float): Confidence score
                - feedback (str): Feedback message
                - issues (List[str]): List of identified issues
                - suggestions (List[str]): List of improvement suggestions
        """
        pass

    @abstractmethod
    def improve(self, text: str, violations: Optional[List[Dict[str, Any]]] = None) -> str:
        """Improve the given text based on violations.

        Args:
            text: The text to improve
            violations: List of violation dictionaries

        Returns:
            str: The improved text
        """
        pass
