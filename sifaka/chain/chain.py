"""
Chain Module

A module providing the main Chain class for the Sifaka chain system.

## Overview
This module provides the main Chain class for the Sifaka chain system,
serving as the primary user-facing interface for running chains. The Chain
class orchestrates the validation and improvement flow between models, rules,
and critics, providing a standardized way to generate, validate, and improve text.

## Components
- Chain: Main user-facing class for running chains
- Model: Interface for text generation models
- Validator: Interface for output validators
- Improver: Interface for output improvers
- Formatter: Interface for result formatters

## Usage Examples
```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]
improver = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain
chain = Chain(
    model=model,
    validators=validators,
    improver=improver,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")

# Run chain asynchronously
import asyncio

async def run_async():
    result = await chain.run_async("Write a short story")
    print(f"Output: {result.output}")

asyncio.run(run_async())
```

## Error Handling
The module handles various error conditions:
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improver refinement fails
- ModelError: Raised when model generation fails
- FormatterError: Raised when formatter formatting fails

## Configuration
Chain behavior can be configured with:
- max_attempts: Maximum number of generation attempts
- cache_enabled: Whether to enable result caching
- trace_enabled: Whether to enable execution tracing
- async_enabled: Whether to enable async execution
- timeout: Timeout for chain operations in seconds
"""

from typing import Any, Dict, List, Optional
import time
import asyncio

from sifaka.interfaces.chain.components import Model, Validator, Improver
from sifaka.interfaces.chain.components.formatter import ChainFormatter as Formatter
from .engine import Engine
from ..utils.state import create_chain_state
from ..utils.common import update_statistics
from ..utils.logging import get_logger
from ..core.results import ChainResult
from ..utils.config.chain import ChainConfig
from ..utils.errors.component import ChainError

# Configure logger
logger = get_logger(__name__)


class Chain:
    """
    Main user-facing class for running chains.

    This class implements the Chain interface from sifaka.interfaces.chain.
    It provides a standardized way to run chains with proper lifecycle management
    and state tracking, orchestrating the validation and improvement flow
    between models, rules, and critics.

    ## Architecture
    The Chain class follows a component-based architecture:
    - Uses Engine for core execution logic
    - Manages components (Model, Validator, Improver, Formatter)
    - Uses StateManager for state management
    - Provides synchronous and asynchronous execution
    - Implements proper error handling and statistics tracking

    ## Lifecycle
    1. **Initialization**: Set up chain resources and configuration
    2. **Execution**: Run inputs through the flow
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage chain configuration
    5. **State Management**: Manage chain state
    6. **Error Handling**: Handle and track errors
    7. **Execution Tracking**: Track execution statistics
    8. **Cleanup**: Release resources when no longer needed

    ## Examples
    ```python
    # Create a chain
    chain = Chain(
        model=OpenAIProvider("gpt-3.5-turbo"),
        validators=[create_length_rule(min_chars=10, max_chars=1000)],
        improver=create_prompt_critic(
            llm_provider=model,
            system_prompt="You are an expert editor that improves text."
        ),
        max_attempts=3
    )

    # Run the chain
    result = chain.run("Write a short story")

    # Access the result
    print(f"Output: {result.output}")
    print(f"All validations passed: {result.all_passed}")
    print(f"Execution time: {result.execution_time:.2f}s")
    ```
    """

    def __init__(
        self,
        model: Model,
        validators: List[Validator] = None,
        improver: Optional[Improver] = None,
        formatter: Optional[Formatter] = None,
        max_attempts: int = 3,
        config: Optional[ChainConfig] = None,
        name: str = "chain",
        description: str = "Sifaka chain for text generation and validation",
    ):
        """
        Initialize the chain.

        Args:
            model: The model to use for generation
            validators: The validators to use for validation
            improver: Optional improver for output improvement
            formatter: Optional formatter for result formatting
            max_attempts: Maximum number of generation attempts
            config: Chain configuration
            name: Chain name
            description: Chain description
        """
        self._name = name
        self._description = description
        self._model = model
        self._validators = validators or []
        self._improver = improver
        self._formatter = formatter
        self._config = config or ChainConfig(max_attempts=max_attempts)

        # Create state manager using the standardized state management
        self.__state_manager = create_chain_state()

        # Create engine with EngineConfig
        from sifaka.utils.config.chain import EngineConfig

        engine_config = EngineConfig(
            max_attempts=self._config.max_attempts, params=self._config.params
        )

        self._engine = Engine(
            state_manager=self._state_manager,
            config=engine_config,
        )

        # Initialize the chain
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the chain.

        This method initializes the chain state and prepares it for execution.
        It should be called after the chain is created to set up any resources
        or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize state
            self._state_manager.update("name", self._name)
            self._state_manager.update("description", self._description)
            self._state_manager.update("model", self._model)
            self._state_manager.update("validators", self._validators)
            self._state_manager.update("improver", self._improver)
            self._state_manager.update("formatter", self._formatter)
            self._state_manager.update("config", self._config)
            self._state_manager.update("initialized", True)
            self._state_manager.update("execution_count", 0)
            self._state_manager.update("result_cache", {})

            # Set metadata
            self._state_manager.set_metadata("component_type", "chain")
            self._state_manager.set_metadata("creation_time", time.time())

            logger.debug(f"Chain '{self._name}' initialized")
        except Exception as e:
            logger.error(f"Chain initialization failed: {str(e)}")
            raise RuntimeError(f"Chain initialization failed: {str(e)}")

    def cleanup(self) -> None:
        """
        Clean up the chain.

        This method cleans up any resources used by the chain.
        It should be called when the chain is no longer needed.

        Raises:
            RuntimeError: If cleanup fails
        """
        try:
            # Clean up resources
            self._state_manager.update("initialized", False)
            logger.debug(f"Chain '{self._name}' cleaned up")
        except Exception as e:
            logger.error(f"Chain cleanup failed: {str(e)}")
            raise RuntimeError(f"Chain cleanup failed: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            The current state as a dictionary
        """
        return {
            "name": self._state_manager.get("name"),
            "description": self._state_manager.get("description"),
            "initialized": self._state_manager.get("initialized", False),
            "execution_count": self._state_manager.get("execution_count", 0),
            "result_cache": self._state_manager.get("result_cache", {}),
            "metadata": self._state_manager._state.metadata,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state.

        Args:
            state: The new state

        Raises:
            ValueError: If the state is invalid
        """
        try:
            # Update state
            for key, value in state.items():
                if key != "metadata":
                    self._state_manager.update(key, value)

            # Update metadata if provided
            if "metadata" in state:
                for key, value in state["metadata"].items():
                    self._state_manager.set_metadata(key, value)

            logger.debug(f"Chain '{self._name}' state updated")
        except Exception as e:
            logger.error(f"Chain state update failed: {str(e)}")
            raise ValueError(f"Invalid state: {str(e)}")

    @property
    def name(self) -> str:
        """
        Get chain name.

        Returns:
            str: The name of the chain
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Get chain description.

        Returns:
            str: The description of the chain
        """
        return self._description

    @property
    def config(self) -> ChainConfig:
        """
        Get chain configuration.

        Returns:
            ChainConfig: The configuration of the chain
        """
        return self._config

    @property
    def _state_manager(self) -> Any:
        """
        Get the state manager.

        This property provides access to the state manager used by the chain
        for state management and tracking.

        Returns:
            StateManager: The state manager instance
        """
        return self.__state_manager

    def update_config(self, config: ChainConfig) -> None:
        """
        Update chain configuration.

        Args:
            config: New chain configuration

        Raises:
            ValueError: If the configuration is invalid
        """
        try:
            self._config = config
            self._state_manager.update("config", config)
            logger.debug(f"Chain '{self._name}' configuration updated")
        except Exception as e:
            logger.error(f"Chain configuration update failed: {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")

    def run(self, prompt: str, **kwargs: Any) -> ChainResult:
        """
        Run the chain on the given prompt.

        Args:
            prompt: The prompt to process
            **kwargs: Additional run parameters

        Returns:
            The chain result

        Raises:
            ChainError: If chain execution fails
            ValueError: If prompt is empty or not a string
        """
        # Ensure the chain is initialized
        if not self._state_manager.get("initialized", False):
            self.initialize()

        # Validate prompt
        from sifaka.utils.text import is_empty_text

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        if is_empty_text(prompt):
            raise ValueError("Prompt must be a non-empty string")

        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()
        self._state_manager.set_metadata("execution_start_time", start_time)

        try:
            # Run engine
            result = self._engine.run(
                prompt=prompt,
                model=self._model,
                validators=self._validators,
                improver=self._improver,
                formatter=self._formatter,
                **kwargs,
            )

            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=True)

            return result

        except Exception as e:
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time

            # Update statistics
            self._update_statistics(execution_time, success=False, error=e)

            # Raise as chain error
            if isinstance(e, ChainError):
                raise e
            raise ChainError(f"Chain execution failed: {str(e)}")

    async def run_async(self, prompt: str, **kwargs: Any) -> ChainResult:
        """
        Run the chain asynchronously.

        Args:
            prompt: The prompt to process
            **kwargs: Additional run parameters

        Returns:
            The chain result

        Raises:
            ChainError: If chain execution fails
            ValueError: If prompt is empty or not a string
        """
        # Ensure the chain is initialized
        if not self._state_manager.get("initialized", False):
            self.initialize()

        # Validate prompt
        from sifaka.utils.text import is_empty_text

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        if is_empty_text(prompt):
            raise ValueError("Prompt must be a non-empty string")

        # Check if async is enabled
        if not self._config.async_enabled:
            raise ChainError("Async execution is not enabled in the configuration")

        try:
            # Check if engine has async methods
            if hasattr(self._engine, "run_async"):
                return await self._engine.run_async(
                    prompt=prompt,
                    model=self._model,
                    validators=self._validators,
                    improver=self._improver,
                    formatter=self._formatter,
                    **kwargs,
                )

            # Fall back to running synchronous method in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.run(prompt, **kwargs))
        except Exception as e:
            # Raise as chain error
            if isinstance(e, ChainError):
                raise e
            raise ChainError(f"Async chain execution failed: {str(e)}")

    def _update_statistics(
        self,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Update chain statistics.

        Args:
            execution_time: Execution time in seconds
            success: Whether execution was successful
            error: Optional error that occurred
        """
        # Use the standardized utility function
        update_statistics(
            state_manager=self._state_manager,
            execution_time=execution_time,
            success=success,
            error=error,
        )

        # Update additional chain-specific statistics
        self._state_manager.set_metadata("last_execution_time", execution_time)

        max_time = self._state_manager.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self._state_manager.set_metadata("max_execution_time", execution_time)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chain statistics.

        Returns:
            Dictionary with chain statistics
        """
        return {
            "name": self._name,
            "execution_count": self._state_manager.get("execution_count", 0),
            "success_count": self._state_manager.get_metadata("success_count", 0),
            "failure_count": self._state_manager.get_metadata("failure_count", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "last_execution_time": self._state_manager.get_metadata("last_execution_time", 0),
            "last_error": self._state_manager.get_metadata("last_error", None),
            "last_error_time": self._state_manager.get_metadata("last_error_time", None),
            "cache_size": len(self._state_manager.get("result_cache", {})),
        }

    def clear_cache(self) -> None:
        """Clear the chain result cache."""
        self._state_manager.update("result_cache", {})
        logger.debug("Chain cache cleared")

    def reset_state(self) -> None:
        """
        Reset the state to its initial values.

        This method resets the chain state to its initial values.
        It should be called when the chain needs to be reset to its initial state.

        Raises:
            RuntimeError: If state reset fails
        """
        try:
            # Reset state manager
            self._state_manager.reset()

            # Re-initialize state
            self.initialize()

            logger.debug(f"Chain '{self._name}' state reset")
        except Exception as e:
            logger.error(f"Chain state reset failed: {str(e)}")
            raise RuntimeError(f"Chain state reset failed: {str(e)}")
