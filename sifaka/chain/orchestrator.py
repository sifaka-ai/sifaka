"""
Chain orchestrator module for Sifaka.

This module provides the ChainOrchestrator class which implements the
standard chain orchestration pattern.
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..generation import Generator
from ..models.base import ModelProvider
from ..rules import Rule
from ..validation import Validator
from .core import ChainCore
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import SimpleRetryStrategy

OutputType = TypeVar("OutputType")


class ChainOrchestrator(Generic[OutputType]):
    """
    Orchestrates the execution of a validation and improvement flow.

    This class provides a standardized implementation of the chain
    orchestration pattern using specialized components.
    """

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a ChainOrchestrator instance.

        Args:
            model: The model provider to use
            rules: The rules to validate outputs against
            critic: Optional critic for improving outputs
            max_attempts: Maximum number of attempts
        """
        # Create components
        validation_manager = ValidationManager[OutputType](rules)
        prompt_manager = PromptManager()
        retry_strategy = SimpleRetryStrategy[OutputType](max_attempts=max_attempts)
        result_formatter = ResultFormatter[OutputType]()

        # Create core chain
        self._core = ChainCore[OutputType](
            model=model,
            validation_manager=validation_manager,
            prompt_manager=prompt_manager,
            retry_strategy=retry_strategy,
            result_formatter=result_formatter,
            critic=critic,
        )

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the prompt through the orchestration flow.

        Args:
            prompt: The prompt to process

        Returns:
            ChainResult containing the output and validation details

        Raises:
            ValueError: If validation fails after max attempts
        """
        return self._core.run(prompt)
