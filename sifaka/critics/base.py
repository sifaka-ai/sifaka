"""
Base class for critics that provide feedback and validation on prompts.
"""

from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, StrictFloat


class Critic(BaseModel, ABC):
    """
    Base class for critics that provide feedback and validation.

    A critic is responsible for analyzing prompts and providing feedback on their quality,
    as well as validating whether they meet certain standards.

    Attributes:
        name: A descriptive name for the critic.
        description: A detailed description of what this critic evaluates.
        config: Configuration parameters for the critic.
        min_confidence: The minimum confidence score required for validation.
    """

    name: str = Field(..., description="Name of the critic")
    description: str = Field(..., description="Description of what this critic evaluates")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
    min_confidence: StrictFloat = Field(
        default=0.7, description="Minimum confidence score required for validation", ge=0.0, le=1.0
    )

    model_config = {"strict": True, "validate_assignment": True}

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty or whitespace")
        return v

    @field_validator("description")
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("description cannot be empty or whitespace")
        return v

    @field_validator("min_confidence")
    def validate_min_confidence(cls, v: float) -> float:
        """Validate min_confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        return v

    @field_validator("config")
    def validate_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config is a dictionary and make a deep copy."""
        if not isinstance(v, dict):
            raise ValueError("config must be a dictionary")
        # Return a deep copy to ensure immutability
        from copy import deepcopy

        return deepcopy(v)

    def __str__(self) -> str:
        """Return string representation of the critic."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Return detailed string representation of the critic."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}', min_confidence={self.min_confidence}, config={self.config})"

    @abstractmethod
    def validate(self, prompt: str) -> bool:
        """
        Check if a prompt meets quality standards.

        Args:
            prompt: The prompt to validate.

        Returns:
            bool: True if the prompt meets quality standards, False otherwise.
        """
        pass

    @abstractmethod
    def critique(self, prompt: str) -> Dict[str, Any]:
        """
        Provide feedback on a prompt.

        Args:
            prompt: The prompt to critique.

        Returns:
            Dict containing critique results with keys:
                - score: float between 0 and 1
                - feedback: str with general feedback
                - issues: list of identified issues
                - suggestions: list of improvement suggestions
        """
        pass

    @abstractmethod
    def improve(self, output: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve an output based on rule violations.

        Args:
            output: The output to improve
            violations: List of rule violations

        Returns:
            str: The improved output
        """
        pass
