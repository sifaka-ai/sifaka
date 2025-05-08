"""
Backoff chain implementation module for Sifaka.

This module provides the BackoffChainImplementation class which implements
the ChainImplementation protocol for a chain with exponential backoff retry strategy.
"""

from typing import Generic, List, Optional, TypeVar, Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ...critics import CriticCore
from ...generation import Generator
from ...models.base import ModelProvider
from ...rules import Rule
from ..config import ChainConfig
from ..formatters.result import ResultFormatter
from ..managers.prompt import PromptManager
from ..managers.validation import ValidationManager
from ..result import ChainResult
from ..strategies.retry import BackoffRetryStrategy
from ...utils.logging import get_logger
from ...utils.state import create_chain_state, ChainState

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class BackoffChainImplementation(BaseModel, Generic[OutputType]):
    """
    Backoff chain implementation that uses exponential backoff for retries.

    This class implements the ChainImplementation protocol for a chain with
    exponential backoff retry strategy. It follows the composition over inheritance
    pattern, allowing it to be composed with the Chain class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize a BackoffChainImplementation instance.

        Args:
            model: The model provider to use
            rules: The rules to validate against
            critic: Optional critic to use
            max_attempts: Maximum number of attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Factor to multiply backoff by each attempt
            max_backoff: Maximum backoff time in seconds
            **kwargs: Additional keyword arguments
        """
        # Initialize the base class
        super().__init__(**kwargs)

        # Create components
        validation_manager = ValidationManager[OutputType](rules)
        prompt_manager = PromptManager()
        retry_strategy = BackoffRetryStrategy[OutputType](
            max_attempts=max_attempts,
            initial_backoff=initial_backoff,
            backoff_factor=backoff_factor,
            max_backoff=max_backoff,
        )
        result_formatter = ResultFormatter[OutputType]()

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store components in state
        state.model = model
        state.validation_manager = validation_manager
        state.prompt_manager = prompt_manager
        state.retry_strategy = retry_strategy
        state.result_formatter = result_formatter
        state.critic = critic

        # Create generator
        state.generator = Generator[OutputType](model)

        # Mark as initialized
        state.initialized = True

    def run_impl(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain implementation with the given prompt.

        Args:
            prompt: The prompt to process

        Returns:
            The chain result with output, validation results, and critique details

        Raises:
            ValueError: If validation fails after max attempts
        """
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("BackoffChainImplementation not properly initialized")

        # Delegate to retry strategy
        return state.retry_strategy.run(
            prompt=prompt,
            generator=state.generator,
            validation_manager=state.validation_manager,
            prompt_manager=state.prompt_manager,
            result_formatter=state.result_formatter,
            critic=state.critic,
        )

    def warm_up_impl(self) -> None:
        """
        Warm up the chain implementation.

        This method initializes any resources needed by the chain implementation.
        """
        # Currently, there's nothing to warm up
        pass
