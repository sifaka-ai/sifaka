"""
Chain executor module for Sifaka.

This module provides the ChainExecutor class which is responsible for executing
a single generation-validation-improvement cycle.
"""

from typing import Generic, Optional, TypeVar, Dict, Any, Tuple

from ..generation import Generator
from ..improvement import Improver
from ..validation import Validator, ValidationResult
from .feedback import FeedbackFormatter

OutputType = TypeVar("OutputType")


class ChainExecutor(Generic[OutputType]):
    """
    Executes a single generation-validation-improvement cycle.
    
    This class is responsible for running a single attempt of generation,
    validation, and improvement, and returning the result.
    """
    
    def __init__(
        self,
        generator: Generator[OutputType],
        validator: Validator[OutputType],
        improver: Optional[Improver[OutputType]] = None,
        feedback_formatter: Optional[FeedbackFormatter] = None,
    ):
        """
        Initialize a ChainExecutor instance.
        
        Args:
            generator: The generator to use for output generation
            validator: The validator to use for output validation
            improver: Optional improver to use for output improvement
            feedback_formatter: Optional formatter for feedback
        """
        self.generator = generator
        self.validator = validator
        self.improver = improver
        self.feedback_formatter = feedback_formatter or FeedbackFormatter()
        
    def execute(
        self, prompt: str
    ) -> Tuple[OutputType, ValidationResult[OutputType], Optional[Dict[str, Any]]]:
        """
        Execute a single cycle.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            Tuple of (output, validation_result, critique_details)
            
        Raises:
            ValueError: If validation fails and no improver is available
        """
        # Generate output
        output = self.generator.generate(prompt)
        
        # Validate output
        validation_result = self.validator.validate(output)
        
        # If validation passed, return result
        if validation_result.all_passed:
            return output, validation_result, None
            
        # If validation failed but we have no improver, raise error
        if not self.improver:
            error_messages = self.validator.get_error_messages(validation_result)
            raise ValueError(f"Validation failed. Errors:\n" + "\n".join(error_messages))
            
        # If we have an improver and validation failed, try to improve
        improvement_result = self.improver.improve(output, validation_result)
        
        return output, validation_result, improvement_result.critique_details
