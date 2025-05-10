"""
Chain Core Module

This module provides the core chain implementation for Sifaka,
enabling text generation, validation, and improvement.
"""

from typing import Any, Dict, Optional, TypeVar
import time

from pydantic import PrivateAttr

from sifaka.core.base import BaseComponent, BaseResult
from sifaka.models.core import ModelProviderCore
from sifaka.critics.core import CriticCore
from sifaka.chain.managers.validation import ValidationManager
from sifaka.core.managers.prompt import PromptManager
from sifaka.chain.strategies.retry import RetryStrategy
from sifaka.chain.formatters.result import ResultFormatter
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ChainCore(BaseComponent):
    """
    Core chain implementation for Sifaka.

    This class provides a unified interface for text generation, validation,
    and improvement, coordinating between model providers, validators,
    and critics.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        model: ModelProviderCore,
        validation_manager: ValidationManager,
        prompt_manager: PromptManager,
        retry_strategy: RetryStrategy,
        result_formatter: ResultFormatter,
        critic: Optional[CriticCore] = None,
        name: str = "chain",
        description: str = "Core chain implementation for Sifaka",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the chain core.

        Args:
            model: The model provider for text generation
            validation_manager: Manager for rule validation
            prompt_manager: Manager for prompt handling
            retry_strategy: Strategy for retry logic
            result_formatter: Formatter for results
            critic: Optional critic for text improvement
            name: Name of the chain
            description: Description of the chain
            config: Additional configuration
        """
        super().__init__()

        self._state.update("model", model)
        self._state.update("validation_manager", validation_manager)
        self._state.update("prompt_manager", prompt_manager)
        self._state.update("retry_strategy", retry_strategy)
        self._state.update("result_formatter", result_formatter)
        self._state.update("critic", critic)
        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "chain")
        self._state.set_metadata("creation_time", time.time())

    def run(self, prompt: str) -> BaseResult:
        """
        Run the chain on the given prompt.

        Args:
            prompt: The prompt to process

        Returns:
            Result containing the output and validation results

        Raises:
            ChainError: If chain execution fails
            ValidationError: If validation fails
            CriticError: If critic refinement fails
            ModelError: If model generation fails
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state.get("result_cache", {})
        if prompt in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[prompt]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get components from state
            model = self._state.get("model")
            validation_manager = self._state.get("validation_manager")
            prompt_manager = self._state.get("prompt_manager")
            retry_strategy = self._state.get("retry_strategy")
            result_formatter = self._state.get("result_formatter")
            critic = self._state.get("critic")

            # Process prompt
            formatted_prompt = prompt_manager.format_prompt(prompt)

            # Generate output
            output = model.invoke(formatted_prompt)

            # Validate output
            validation_results = validation_manager.validate(output)

            # Improve output if needed
            if critic and not all(r.passed for r in validation_results):
                output = critic.improve(output, validation_results)

            # Format result
            result = result_formatter.format_result(output, validation_results)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[prompt] = result
            self._state.update("result_cache", cache)

            return result

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Chain execution error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about chain usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state.get("execution_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "model_name": self._state.get("model").name,
        }

    def clear_cache(self) -> None:
        """Clear the chain result cache."""
        self._state.update("result_cache", {})
        logger.debug("Chain cache cleared")
