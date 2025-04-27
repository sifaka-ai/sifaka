from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict as PydanticConfigDict
from pydantic.functional_validators import field_validator
from pydantic_core import PydanticCustomError
from .protocols import TextValidator, TextCritic, TextImprover, ConfigDict, is_config_dict


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


class Critic(BaseModel):
    """Base class for all critics.

    This class provides the core functionality for text validation,
    critique, and improvement. It uses protocols for flexibility
    and dependency injection.
    """

    model_config = PydanticConfigDict(strict=True, validate_assignment=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    config: ConfigDict = Field(default_factory=dict)

    validator: Optional[TextValidator] = None
    critic: Optional[TextCritic] = None
    improver: Optional[TextImprover] = None

    @field_validator("config")
    def validate_config(cls, v: Any) -> ConfigDict:
        """Validate configuration dictionary."""
        if not is_config_dict(v):
            raise PydanticCustomError(
                "invalid_config", "Config must be a dictionary with valid types"
            )
        return v

    def __str__(self) -> str:
        """Return string representation of the critic."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Return detailed string representation of the critic."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"min_confidence={self.min_confidence}, "
            f"config={self.config})"
        )

    def validate(self, text: str) -> bool:
        """Validate the given text.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text is valid, False otherwise

        Raises:
            RuntimeError: If no validator is configured
        """
        if self.validator is None:
            raise RuntimeError("No validator configured")
        return self.validator.validate(text)

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

        Raises:
            RuntimeError: If no critic is configured
        """
        if self.critic is None:
            raise RuntimeError("No critic configured")
        return self.critic.critique(text)

    def improve(self, text: str, feedback: str) -> str:
        """Improve the given text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide improvement

        Returns:
            str: The improved text

        Raises:
            RuntimeError: If no improver is configured
        """
        if self.improver is None:
            raise RuntimeError("No improver configured")
        return self.improver.improve(text, feedback)

    async def avalidate(self, text: str) -> bool:
        """Async version of validate."""
        if self.validator is None:
            raise RuntimeError("No validator configured")
        return await self.validator.validate(text)

    async def acritique(self, text: str) -> Dict[str, Any]:
        """Async version of critique."""
        if self.critic is None:
            raise RuntimeError("No critic configured")
        return await self.critic.critique(text)

    async def aimprove(self, text: str, feedback: str) -> str:
        """Async version of improve."""
        if self.improver is None:
            raise RuntimeError("No improver configured")
        return await self.improver.improve(text, feedback)

    def __enter__(self) -> "Critic":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        # Cleanup any resources if needed
        pass
