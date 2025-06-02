"""Base critic implementation for Sifaka.

This module provides the base class for all critics, implementing common
functionality and defining the standard interface that all critics must follow.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin
from sifaka.critics.mixins.validation_aware import ValidationAwareMixin

logger = get_logger(__name__)


class BaseCritic(ContextAwareMixin, ValidationAwareMixin, ABC):
    """Base class for all critics.

    This class provides common functionality for critics including:
    - Standard critique/improve interface
    - Context handling via ContextAwareMixin
    - Validation awareness via ValidationAwareMixin
    - Async-under-the-hood implementation
    - Consistent error handling
    - Standard return schema

    All critics must inherit from this class and implement the abstract methods.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the base critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__()

        # Initialize model
        if model is None:
            if model_name is None:
                raise ValueError("Either model or model_name must be provided")
            self.model = create_model(model_name, **model_kwargs)
        else:
            # Handle case where model is actually a string (model name)
            if isinstance(model, str):
                self.model = create_model(model, **model_kwargs)
            else:
                self.model = model

    async def critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Critique text asynchronously.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary containing critique results.
        """
        start_time = time.time()

        with critic_context(
            critic_name=self.__class__.__name__,
            operation="critique_async",
            message_prefix=f"Failed to critique text with {self.__class__.__name__}",
        ):
            # Check if text is available
            if not thought.text:
                return self._create_error_result(
                    "No text available for critique",
                    issues=["Text is empty or None"],
                    suggestions=["Provide text to critique"],
                    start_time=start_time,
                )

            # Delegate to subclass implementation
            try:
                # All critics must now be async-only
                result = await self._perform_critique_async(thought)

                # Ensure processing time is included
                processing_time = (time.time() - start_time) * 1000
                result["processing_time_ms"] = processing_time

                # Validate result schema
                self._validate_result_schema(result)

                return result

            except Exception as e:
                logger.error(f"{self.__class__.__name__} async critique failed: {e}")
                return self._create_error_result(
                    f"Async critique failed: {str(e)}",
                    issues=[f"Internal error: {str(e)}"],
                    suggestions=["Please try again or check the critic configuration"],
                    start_time=start_time,
                )

    @abstractmethod
    async def improve_async(self, thought: Thought) -> str:
        """Improve text based on critique asynchronously.

        Each critic must implement its own async improvement strategy.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the improvement fails.
        """

    @abstractmethod
    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic asynchronously (implemented by subclasses).

        All critics must now be async-only to work with PydanticAI architecture.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """

    def _create_error_result(
        self, message: str, issues: List[str], suggestions: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Create a standard error result.

        Args:
            message: The error message.
            issues: List of issues found.
            suggestions: List of suggestions.
            start_time: When the operation started.

        Returns:
            A standard error result dictionary.
        """
        processing_time = (time.time() - start_time) * 1000
        return {
            "needs_improvement": True,
            "message": message,
            "issues": issues,
            "suggestions": suggestions,
            "processing_time_ms": processing_time,
            "confidence": 0.0,
            "metadata": {"error": True},
        }

    def _validate_result_schema(self, result: Dict[str, Any]) -> None:
        """Validate that the result follows the standard schema.

        Args:
            result: The result dictionary to validate.

        Raises:
            ValueError: If the result doesn't follow the standard schema.
        """
        required_fields = ["needs_improvement", "message", "issues", "suggestions"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field '{field}' in critic result")

        # Type validation
        if not isinstance(result["needs_improvement"], bool):
            raise ValueError("'needs_improvement' must be a boolean")
        if not isinstance(result["message"], str):
            raise ValueError("'message' must be a string")
        if not isinstance(result["issues"], list):
            raise ValueError("'issues' must be a list")
        if not isinstance(result["suggestions"], list):
            raise ValueError("'suggestions' must be a list")
