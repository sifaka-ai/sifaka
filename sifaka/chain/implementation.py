"""
Chain implementation module for Sifaka.

This module provides the ChainImplementation protocol and Chain class which
implement the composition over inheritance pattern for chains.
"""

from typing import Dict, Generic, List, Optional, Protocol, TypeVar, Any, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..critics import CriticCore
from ..generation import Generator
from ..models.base import ModelProvider
from ..rules import Rule
from .config import ChainConfig
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .result import ChainResult
from .strategies.retry import RetryStrategy
from ..utils.logging import get_logger
from ..utils.state import create_chain_state, ChainState, StateManager

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


@runtime_checkable
class ChainImplementation(Protocol[OutputType]):
    """
    Protocol for chain implementations.

    This protocol defines the core chain logic that can be composed with
    the Chain class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.
    """

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
        ...

    def warm_up_impl(self) -> None:
        """
        Warm up the chain implementation.

        This method initializes any resources needed by the chain implementation.
        """
        ...


class Chain(BaseModel, Generic[OutputType]):
    """
    Chain that uses composition over inheritance.

    This class delegates chain execution to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the chain", min_length=1)
    description: str = Field(description="Description of the chain", min_length=1)
    config: ChainConfig = Field(description="Configuration for the chain")
    _implementation: ChainImplementation[OutputType] = PrivateAttr()
    _state_manager: StateManager[ChainState] = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        name: str,
        description: str,
        config: ChainConfig,
        implementation: ChainImplementation[OutputType],
        **kwargs: Any,
    ):
        """
        Initialize the chain.

        Args:
            name: The name of the chain
            description: The description of the chain
            config: The configuration for the chain
            implementation: The implementation to use
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain with the given prompt.

        This method delegates to the implementation's run_impl method.

        Args:
            prompt: The prompt to process

        Returns:
            The chain result with output, validation results, and critique details

        Raises:
            ValueError: If validation fails after max attempts
        """
        return self._implementation.run_impl(prompt)

    def warm_up(self) -> None:
        """
        Warm up the chain.

        This method delegates to the implementation's warm_up_impl method.
        """
        self._implementation.warm_up_impl()
