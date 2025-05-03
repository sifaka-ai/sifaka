"""
Improvement module for Sifaka.

This module provides components for improving outputs based on validation results.
"""

from dataclasses import dataclass, asdict
from typing import Optional, TypeVar, Generic, Dict, Any

from .critics.base import BaseCritic, CriticMetadata
from .validation import ValidationResult
from .utils.logging import get_logger

logger = get_logger(__name__)

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

    def __init__(self, critic: BaseCritic):
        """
        Initialize an Improver instance.

        Args:
            critic: The critic to use for improvement
        """
        self.critic = critic

    def improve(
        self, output: OutputType, validation_result: ValidationResult[OutputType]
    ) -> ImprovementResult[OutputType]:
        """
        Improve the output based on validation results.

        Args:
            output: The original output to improve
            validation_result: Validation results containing rule failures

        Returns:
            ImprovementResult containing the improved output and critique details
        """
        if validation_result.all_passed or not self.critic:
            logger.debug("No improvement needed: validation passed or no critic")
            return ImprovementResult(output=output, improved=False)

        # Get critique from the critic
        critique = self.critic.critique(output)
        logger.debug(f"Got critique: {critique}")
        logger.debug(f"Critique type: {type(critique)}")

        # If critique is None, return unimproved result
        if critique is None:
            logger.debug("No critique provided, returning unimproved result")
            return ImprovementResult(output=output, improved=False)

        # Process critique details based on type
        critique_details = None
        if isinstance(critique, CriticMetadata):
            logger.debug("Processing CriticMetadata")
            # Convert CriticMetadata to dict using asdict
            critique_details = asdict(critique)
            logger.debug(f"CriticMetadata fields: {critique_details}")

            logger.debug(f"Converted CriticMetadata to dict: {critique_details}")
        elif isinstance(critique, dict):
            logger.debug("Processing dict feedback")
            critique_details = critique.copy()
            logger.debug(f"Dict feedback: {critique_details}")

        # If we have critique details, mark as improved
        if critique_details:
            logger.debug("Returning improved result with critique details")
            return ImprovementResult(
                output=output, critique_details=critique_details, improved=True
            )

        logger.debug("No critique details, returning unimproved result")
        return ImprovementResult(output=output, improved=False)

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
