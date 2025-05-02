"""
Base chain module for Sifaka.

This module provides the Chain class which is the main user-facing class for
orchestrating the validation and improvement flow.
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import PromptCritic
from ..generation import Generator
from ..improvement import Improver
from ..models.base import ModelProvider
from ..rules import Rule
from ..validation import Validator
from .feedback import FeedbackFormatter
from .orchestrator import ChainOrchestrator
from .result import ChainResult
from .retry import RetryManager

OutputType = TypeVar("OutputType")


class Chain(Generic[OutputType]):
    """
    Chain class that orchestrates the validation and improvement flow.

    This class combines generation, validation, and improvement components to
    create a complete pipeline for generating, validating, and improving outputs.
    
    It delegates to specialized components for each responsibility:
    - ChainOrchestrator: Coordinates the overall flow
    - ChainExecutor: Executes a single cycle
    - RetryManager: Manages retry logic
    - FeedbackFormatter: Formats feedback
    """

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[PromptCritic] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a Chain instance.

        Args:
            model: The model provider to use for generation
            rules: List of validation rules to apply
            critic: Optional critic for improving outputs
            max_attempts: Maximum number of improvement attempts
        """
        # Create core components
        self.generator = Generator[OutputType](model)
        self.validator = Validator[OutputType](rules)
        self.improver = Improver[OutputType](critic) if critic else None
        
        # Create supporting components
        self.feedback_formatter = FeedbackFormatter()
        self.retry_manager = RetryManager(max_attempts=max_attempts)
        
        # Create orchestrator
        self.orchestrator = ChainOrchestrator(
            generator=self.generator,
            validator=self.validator,
            improver=self.improver,
            retry_manager=self.retry_manager,
            feedback_formatter=self.feedback_formatter,
        )

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the prompt through the chain.

        Args:
            prompt: The input prompt to process

        Returns:
            ChainResult containing the output and validation details

        Raises:
            ValueError: If validation fails after max attempts
        """
        return self.orchestrator.run(prompt)
