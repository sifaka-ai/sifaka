"""
Improvement module for Sifaka.

This module provides components for improving outputs based on validation results. It includes:
- ImprovementResult: A generic container for improvement results
- Improver: A class that uses critics to improve outputs

The improvement process follows these steps:
1. Initialize an Improver with a critic
2. Pass an output and its validation results to improve()
3. Receive an ImprovementResult containing:
   - The improved output
   - Critique details
   - Improvement status
4. Extract feedback if needed using get_feedback()

Example:
    ```python
    from sifaka.improvement import Improver
    from sifaka.critics import BaseCritic
    from sifaka.validation import ValidationResult

    # Create critic
    critic = BaseCritic()

    # Create improver
    improver = Improver(critic)

    # Improve output
    result = improver.improve("Original text", validation_result)
    if result.improved:
        feedback = improver.get_feedback(result.critique_details)
        print(f"Improvement feedback: {feedback}")
    ```
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
    """
    Result from an improvement attempt, including the improved output and details.

    This class serves as a container for improvement results, providing:
    - The output (original or improved)
    - Details from the critic's critique
    - Status of the improvement attempt

    Attributes:
        output: The output (original or improved)
        critique_details: Optional dictionary containing details from the critic
        improved: Boolean indicating if improvement was successful

    Example:
        ```python
        result = ImprovementResult(
            output="Improved text",
            critique_details={"feedback": "Made text more concise"},
            improved=True
        )
        ```
    """

    output: OutputType
    critique_details: Optional[Dict[str, Any]] = None
    improved: bool = False


class Improver(Generic[OutputType]):
    """
    Improver class that handles improving outputs based on validation results.

    This class is responsible for:
    1. Using critics to analyze outputs
    2. Processing critique results
    3. Determining if improvement was successful
    4. Providing access to feedback

    The improver follows a simple workflow:
    1. Initialize with a critic
    2. Pass output and validation results to improve()
    3. Extract feedback if needed using get_feedback()

    Example:
        ```python
        improver = Improver(BaseCritic())
        result = improver.improve("Original text", validation_result)
        if result.improved:
            feedback = improver.get_feedback(result.critique_details)
        ```
    """

    def __init__(self, critic: BaseCritic):
        """
        Initialize an Improver instance.

        Args:
            critic: The critic to use for improvement. Must implement the
                   BaseCritic protocol and provide a critique() method.

        Raises:
            ValueError: If critic is None
            TypeError: If critic does not implement BaseCritic
        """
        if critic is None:
            raise ValueError("Critic cannot be None")
        self.critic = critic

    def improve(
        self, output: OutputType, validation_result: ValidationResult[OutputType]
    ) -> ImprovementResult[OutputType]:
        """
        Improve the output based on validation results.

        This method:
        1. Checks if improvement is needed
        2. Gets critique from the critic
        3. Processes critique details
        4. Returns an ImprovementResult

        Args:
            output: The original output to improve
            validation_result: Validation results containing rule failures.
                             Used to determine if improvement is needed.

        Returns:
            ImprovementResult containing:
            - The output (original or improved)
            - Critique details from the critic
            - Improvement status

        Raises:
            ValueError: If output is None or validation_result is invalid
            RuntimeError: If critic fails during critique
        """
        if output is None:
            raise ValueError("Output cannot be None")
        if validation_result is None:
            raise ValueError("Validation result cannot be None")

        if validation_result.all_passed or not self.critic:
            logger.debug("No improvement needed: validation passed or no critic")
            return ImprovementResult(output=output, improved=False)

        try:
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
                critique_details = asdict(critique)
                logger.debug(f"CriticMetadata fields: {critique_details}")
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
        except Exception as e:
            raise RuntimeError(f"Critic failed during critique: {str(e)}")

    def get_feedback(self, critique_details: Dict[str, Any]) -> str:
        """
        Extract feedback from critique details.

        This method extracts the feedback message from critique details,
        handling different formats of critique data.

        Args:
            critique_details: The critique details to extract feedback from.
                            Can be a dictionary or CriticMetadata object.

        Returns:
            Feedback string from the critique. Returns empty string if no
            feedback is found.

        Example:
            ```python
            details = {"feedback": "Text needs more detail"}
            feedback = improver.get_feedback(details)
            # feedback = "Text needs more detail"
            ```
        """
        if isinstance(critique_details, dict) and "feedback" in critique_details:
            return critique_details["feedback"]
        return ""
