"""
Chain Module

This module provides the main Chain class for the Sifaka chain system.
It serves as the primary user-facing interface for running chains.

## Components
1. **Chain**: Main user-facing class for running chains

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
```
"""

from typing import Any, Dict, List, Optional
import time
import asyncio

from .interfaces import Model, Validator, Improver, Formatter
from .engine import Engine
from ..utils.state import StateManager, create_chain_state
from ..utils.common import update_statistics, record_error
from ..utils.logging import get_logger
from .result import ChainResult
from .config import ChainConfig
from ..utils.errors import ChainError

# Configure logger
logger = get_logger(__name__)


class Chain:
    """Main user-facing class for running chains."""

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
        self._state_manager = create_chain_state()

        # Create engine
        self._engine = Engine(
            state_manager=self._state_manager,
            config=self._config,
        )

        # Initialize state
        self._state_manager.update("name", name)
        self._state_manager.update("description", description)
        self._state_manager.update("model", model)
        self._state_manager.update("validators", validators or [])
        self._state_manager.update("improver", improver)
        self._state_manager.update("formatter", formatter)
        self._state_manager.update("config", self._config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "chain")
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get chain name."""
        return self._name

    @property
    def description(self) -> str:
        """Get chain description."""
        return self._description

    @property
    def config(self) -> ChainConfig:
        """Get chain configuration."""
        return self._config

    def update_config(self, config: ChainConfig) -> None:
        """
        Update chain configuration.

        Args:
            config: New chain configuration
        """
        self._config = config
        self._state_manager.update("config", config)

    def run(self, prompt: str) -> ChainResult:
        """
        Run the chain on the given prompt.

        Args:
            prompt: The prompt to process

        Returns:
            The chain result

        Raises:
            ChainError: If chain execution fails
        """
        try:
            # Track execution count
            execution_count = self._state_manager.get("execution_count", 0)
            self._state_manager.update("execution_count", execution_count + 1)

            # Record start time
            start_time = time.time()
            self._state_manager.set_metadata("execution_start_time", start_time)

            # Run engine
            result = self._engine.run(
                prompt=prompt,
                model=self._model,
                validators=self._validators,
                improver=self._improver,
                formatter=self._formatter,
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

    async def run_async(self, prompt: str) -> ChainResult:
        """
        Run the chain asynchronously.

        Args:
            prompt: The prompt to process

        Returns:
            The chain result

        Raises:
            ChainError: If chain execution fails
        """
        # Check if async is enabled
        if not self._config.async_enabled:
            raise ChainError("Async execution is not enabled in the configuration")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, prompt)

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
        """Reset chain state."""
        self._state_manager.reset()

        # Re-initialize state
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

        logger.debug("Chain state reset")
