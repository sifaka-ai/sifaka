"""
Improvement module for Sifaka.

This module provides components for improving outputs based on validation results.
"""

from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, Dict, Any, Union

from .critics import PromptCritic
from .critics.prompt import CriticMetadata
from .validation import ValidationResult

OutputType = TypeVar("OutputType")


@dataclass
class ImprovementResult(Generic[OutputType]):
    """Result from an improvement attempt, including the improved output and details."""

    output: OutputType
    critique_details: Optional[Dict[str, Any]] = None
    improved: bool = False


class Improver(Generic[OutputType]):
    """
    Improver class that handles improving outputs based on validation results.

    This class is responsible for using critics to improve outputs that fail validation.
    """

    def __init__(self, critic: PromptCritic):
        """
        Initialize an Improver instance.

        Args:
            critic: The critic to use for improvement
        """
        self.critic = critic

    def improve(self,
                output: OutputType,
                validation_result: ValidationResult[OutputType]) -> ImprovementResult[OutputType]:
        """
        Improve the output based on validation results.

        Args:
            output: The original output to improve
            validation_result: Validation results containing rule failures

        Returns:
            ImprovementResult containing the improved output and critique details
        """
        if validation_result.all_passed or not self.critic:
            return ImprovementResult(output=output, improved=False)

        # Get critique from the critic
        critique = self.critic.critique(output)

        # Process critique details based on type
        critique_details = None
        if isinstance(critique, CriticMetadata):
            critique_details = critique.__dict__
        elif isinstance(critique, dict):
            critique_details = critique

        # Output didn't need improvement or critic didn't provide feedback
        if not critique_details:
            return ImprovementResult(output=output, improved=False)

        return ImprovementResult(
            output=output,
            critique_details=critique_details,
            improved=True
        )

    def get_feedback(self, critique_details: Dict[str, Any]) -> str:
        """
        Extract feedback from critique details.

        Args:
            critique_details: The critique details to extract feedback from

        Returns:
            Feedback string from the critique
        """
        if isinstance(critique_details, dict) and "feedback" in critique_details:
            return critique_details["feedback"]
        return ""