"""
Chain orchestrator module for Sifaka.

This module provides the ChainOrchestrator class which is responsible for
coordinating the overall flow between components.
"""

from typing import Generic, Optional, TypeVar

from ..generation import Generator
from ..improvement import Improver
from ..validation import Validator
from .executor import ChainExecutor
from .feedback import FeedbackFormatter
from .result import ChainResult
from .retry import RetryManager

OutputType = TypeVar("OutputType")


class ChainOrchestrator(Generic[OutputType]):
    """
    Orchestrates the flow between generation, validation, and improvement components.
    
    This class is responsible for coordinating the overall flow, initializing
    and configuring components, and returning final results.
    """
    
    def __init__(
        self,
        generator: Generator[OutputType],
        validator: Validator[OutputType],
        improver: Optional[Improver[OutputType]] = None,
        retry_manager: Optional[RetryManager] = None,
        feedback_formatter: Optional[FeedbackFormatter] = None,
    ):
        """
        Initialize a ChainOrchestrator instance.
        
        Args:
            generator: The generator to use for output generation
            validator: The validator to use for output validation
            improver: Optional improver to use for output improvement
            retry_manager: Optional manager for retries
            feedback_formatter: Optional formatter for feedback
        """
        self.generator = generator
        self.validator = validator
        self.improver = improver
        self.retry_manager = retry_manager or RetryManager()
        self.feedback_formatter = feedback_formatter or FeedbackFormatter()
        
    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the orchestration flow.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            ChainResult containing the output and validation details
            
        Raises:
            ValueError: If validation fails after max attempts
        """
        executor = ChainExecutor(
            generator=self.generator,
            validator=self.validator,
            improver=self.improver,
            feedback_formatter=self.feedback_formatter,
        )
        
        return self.retry_manager.run_with_retries(executor, prompt)
