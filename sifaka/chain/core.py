"""
Core chain module for Sifaka.

This module provides the ChainCore class which is the main interface for
chains, delegating to specialized components.
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..generation import Generator
from ..models.base import ModelProvider
from ..rules import Rule
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import RetryStrategy, SimpleRetryStrategy
from ..utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ChainCore(Generic[OutputType]):
    """
    Core chain implementation that delegates to specialized components.
    
    This class implements the Chain interface but delegates most of its
    functionality to specialized components for better separation of concerns.
    """
    
    def __init__(
        self,
        model: ModelProvider,
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        retry_strategy: RetryStrategy[OutputType],
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ):
        """
        Initialize a ChainCore instance.
        
        Args:
            model: The model provider to use
            validation_manager: The validation manager to use
            prompt_manager: The prompt manager to use
            retry_strategy: The retry strategy to use
            result_formatter: The result formatter to use
            critic: Optional critic to use
        """
        self._model = model
        self._validation_manager = validation_manager
        self._prompt_manager = prompt_manager
        self._retry_strategy = retry_strategy
        self._result_formatter = result_formatter
        self._critic = critic
        
        # Create generator
        self._generator = Generator[OutputType](model)
        
    @property
    def model(self) -> ModelProvider:
        """
        Get the model provider.
        
        Returns:
            The model provider
        """
        return self._model
        
    @property
    def validation_manager(self) -> ValidationManager[OutputType]:
        """
        Get the validation manager.
        
        Returns:
            The validation manager
        """
        return self._validation_manager
        
    @property
    def prompt_manager(self) -> PromptManager:
        """
        Get the prompt manager.
        
        Returns:
            The prompt manager
        """
        return self._prompt_manager
        
    @property
    def retry_strategy(self) -> RetryStrategy[OutputType]:
        """
        Get the retry strategy.
        
        Returns:
            The retry strategy
        """
        return self._retry_strategy
        
    @property
    def result_formatter(self) -> ResultFormatter[OutputType]:
        """
        Get the result formatter.
        
        Returns:
            The result formatter
        """
        return self._result_formatter
        
    @property
    def critic(self) -> Optional[CriticCore]:
        """
        Get the critic.
        
        Returns:
            The critic, or None if not set
        """
        return self._critic
        
    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain with the given prompt.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            The chain result
            
        Raises:
            ValueError: If validation fails after max attempts
        """
        return self._retry_strategy.run(
            prompt=prompt,
            generator=self._generator,
            validation_manager=self._validation_manager,
            prompt_manager=self._prompt_manager,
            result_formatter=self._result_formatter,
            critic=self._critic,
        )
