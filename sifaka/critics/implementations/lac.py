"""
LAC (LLM-Based Actor-Critic) critic module for Sifaka.

This module implements the LAC approach for critics, which combines language feedback
and value scoring to improve language model-based decision making.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692

## Component Lifecycle

### LAC Critic Lifecycle

1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Factory initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Text improvement
   - Feedback processing

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery

Example:
    ```python
    from sifaka.critics.implementations.lac import create_lac_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a LAC critic
    critic = create_lac_critic(llm_provider=provider)

    # Use the critic to improve text
    task = "Summarize the causes of World War I in 3 bullet points."
    response = provider.generate(f"Task:\n{task}")
    results = critic.critique(response, {"task": task})

    print("Feedback:", results["feedback"])
    print("Value Score:", results["value"])
    ```
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field, PrivateAttr, ConfigDict

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...core.base import BaseResult as CriticResult
from ..config import FeedbackCriticConfig, ValueCriticConfig, LACCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Default prompt templates
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides detailed feedback and evaluation."
)

DEFAULT_FEEDBACK_PROMPT_TEMPLATE = (
    "Please provide detailed feedback on the following response to a task.\n\n"
    "Task:\n{task}\n\n"
    "Response:\n{response}\n\n"
    "Feedback:"
)

DEFAULT_VALUE_PROMPT_TEMPLATE = (
    "Please evaluate the following response to a task on a scale from 0 to 10, "
    "where 0 is completely incorrect or unhelpful and 10 is perfect.\n\n"
    "Task:\n{task}\n\n"
    "Response:\n{response}\n\n"
    "Score (0-10):"
)


class FeedbackCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that produces natural language feedback for a model's response to a task.

    This critic analyzes text and provides detailed feedback on what could be
    improved or what was done well.

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - initialized: Initialization status
      - cache: Temporary data storage
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: FeedbackCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[FeedbackCriticConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the feedback critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: Language model provider to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_FEEDBACK_CONFIG

            config = DEFAULT_FEEDBACK_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            self._state_manager.update("model", llm_provider)

            # Store configuration in cache
            cache = {
                "feedback_prompt_template": config.feedback_prompt_template,
                "system_prompt": config.system_prompt,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize FeedbackCritic: {str(e)}") from e

    def _check_input(self, text: str) -> None:
        """
        Validate input text and initialization state.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state_manager.get("initialized", False):
            raise RuntimeError("FeedbackCritic not properly initialized")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If metadata is None or missing task key
        """
        if metadata is None or "task" not in metadata:
            raise ValueError("metadata must contain a 'task' key")
        return metadata["task"]

    def process(self, input: str) -> CriticResult:
        """
        Process the input text and return a result.

        This is the main method required by BaseComponent.

        Args:
            input: The text to process

        Returns:
            CriticResult: The result of processing the text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Validate input
            if not isinstance(input, str) or not input.strip():
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("FeedbackCritic not properly initialized")

            # Create a default task if none provided
            task = "Provide feedback on the following text"

            # Generate feedback
            feedback = self.run(task, input)

            # Create result
            result = CriticResult(
                passed=True,  # Feedback critics always pass
                message=feedback,
                metadata={"operation": "process"},
                score=0.5,  # Default score
                issues=[],
                suggestions=[],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            self.update_statistics(result)

            return result

        except Exception as e:
            self.record_error(e)
            processing_time = (time.time() - start_time) * 1000
            return CriticResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Processing error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=processing_time,
            )

    def run(self, task: str, response: str) -> str:
        """
        Generate natural language feedback for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback on

        Returns:
            Natural language feedback

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(response)

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create feedback prompt
            prompt = cache.get("feedback_prompt_template", DEFAULT_FEEDBACK_PROMPT_TEMPLATE).format(
                task=task,
                response=response,
            )

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Generate feedback
            feedback = model.generate(
                prompt,
                system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            ).strip()

            # Update statistics
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            feedback_count = self._state_manager.get_metadata("feedback_count", 0)
            self._state_manager.set_metadata("feedback_count", feedback_count + 1)

            return feedback

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to generate feedback: {str(e)}") from e

    async def arun(self, task: str, response: str) -> str:
        """
        Asynchronously generate natural language feedback for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback on

        Returns:
            Natural language feedback

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(response)

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create feedback prompt
            prompt = cache.get("feedback_prompt_template", DEFAULT_FEEDBACK_PROMPT_TEMPLATE).format(
                task=task,
                response=response,
            )

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Check if model supports async
            if hasattr(model, "agenerate"):
                feedback = await model.agenerate(
                    prompt,
                    system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    temperature=cache.get("temperature", 0.7),
                    max_tokens=cache.get("max_tokens", 1000),
                )
                feedback = feedback.strip()
            else:
                # Fallback to sync method in async context
                import asyncio

                feedback = await asyncio.to_thread(
                    model.generate,
                    prompt,
                    system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    temperature=cache.get("temperature", 0.7),
                    max_tokens=cache.get("max_tokens", 1000),
                )
                feedback = feedback.strip()

            # Update statistics
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            feedback_count = self._state_manager.get_metadata("feedback_count", 0)
            self._state_manager.set_metadata("feedback_count", feedback_count + 1)

            return feedback

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously generate feedback: {str(e)}") from e

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Feedback critics always return True for validation
            # as they focus on providing feedback rather than validation

            # Update statistics
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return True

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Generate feedback
            feedback = self.run(task, text)

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create improvement prompt
            prompt = (
                f"Task:\n{task}\n\n"
                f"Original response:\n{text}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Improved response:"
            )

            # Generate improved response
            improved_text = model.generate(
                prompt,
                system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            ).strip()

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            self._state_manager.set_metadata("last_improvement_time", time.time())

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create improvement prompt
            prompt = f"Original text:\n{text}\n\n" f"Feedback:\n{feedback}\n\n" f"Improved text:"

            # Generate improved response
            improved_text = model.generate(
                prompt,
                system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            ).strip()

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            self._state_manager.set_metadata("last_improvement_time", time.time())

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing feedback

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Generate feedback
            feedback = self.run(task, text)

            # Create critique result
            result = {
                "score": 0.5,  # Default score since feedback critics don't provide scores
                "feedback": feedback,
                "issues": [],
                "suggestions": [],
            }

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    async def avalidate(self, text: str) -> bool:
        """
        Asynchronously check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Feedback critics always return True for validation
            # as they focus on providing feedback rather than validation

            # Update statistics
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return True

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously validate text: {str(e)}") from e

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Asynchronously improve text based on feedback.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Generate feedback
            feedback = await self.arun(task, text)

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create improvement prompt
            prompt = (
                f"Task:\n{task}\n\n"
                f"Original response:\n{text}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Improved response:"
            )

            # Check if model supports async
            if hasattr(model, "agenerate"):
                improved_text = await model.agenerate(
                    prompt,
                    system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    temperature=cache.get("temperature", 0.7),
                    max_tokens=cache.get("max_tokens", 1000),
                )
                improved_text = improved_text.strip()
            else:
                # Fallback to sync method in async context
                import asyncio

                improved_text = await asyncio.to_thread(
                    model.generate,
                    prompt,
                    system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    temperature=cache.get("temperature", 0.7),
                    max_tokens=cache.get("max_tokens", 1000),
                )
                improved_text = improved_text.strip()

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            self._state_manager.set_metadata("last_improvement_time", time.time())

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously improve text: {str(e)}") from e

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing feedback

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Generate feedback
            feedback = await self.arun(task, text)

            # Create critique result
            result = {
                "score": 0.5,  # Default score since feedback critics don't provide scores
                "feedback": feedback,
                "issues": [],
                "suggestions": [],
            }

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously critique text: {str(e)}") from e

    def warm_up(self) -> None:
        """
        Prepare the critic for use.

        This method ensures that the critic is properly initialized and ready to use.
        It can be called before using the critic to ensure that all resources are
        properly initialized.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Check if already initialized
            if self._state_manager.get("initialized", False):
                return

            # Initialize components if needed
            if not self._state_manager.get("model"):
                raise RuntimeError("Model provider not initialized")

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("warm_up_time", time.time())

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to warm up critic: {str(e)}") from e

    def cleanup(self) -> None:
        """
        Clean up resources used by the critic.

        This method releases any resources held by the critic, such as
        connections to external services or cached data.

        Raises:
            RuntimeError: If cleanup fails
        """
        try:
            # Clear cache
            self._state_manager.update("cache", {})

            # Mark as not initialized
            self._state_manager.update("initialized", False)
            self._state_manager.set_metadata("cleanup_time", time.time())

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to clean up critic: {str(e)}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the critic's performance.

        Returns:
            Dict[str, Any]: A dictionary containing statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()

        # Add critic-specific statistics
        stats.update(
            {
                "critique_count": self._state_manager.get_metadata("critique_count", 0),
                "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
                "feedback_count": self._state_manager.get_metadata("feedback_count", 0),
                "last_improvement_time": self._state_manager.get_metadata("last_improvement_time"),
                "model_provider": (
                    str(self._state_manager.get("model").__class__.__name__)
                    if self._state_manager.get("model")
                    else None
                ),
            }
        )

        return stats


class ValueCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that estimates a numeric value for a model's response to a task.

    This critic analyzes text and provides a numeric score (e.g., probability of success)
    for the response.

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - initialized: Initialization status
      - cache: Temporary data storage
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: ValueCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[ValueCriticConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the value critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: Language model provider to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_VALUE_CONFIG

            config = DEFAULT_VALUE_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            self._state_manager.update("model", llm_provider)

            # Store configuration in cache
            cache = {
                "value_prompt_template": config.value_prompt_template,
                "system_prompt": config.system_prompt,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "min_score": config.min_score,
                "max_score": config.max_score,
            }
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize ValueCritic: {str(e)}") from e

    def _check_input(self, text: str) -> None:
        """
        Validate input text and initialization state.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state_manager.get("initialized", False):
            raise RuntimeError("ValueCritic not properly initialized")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If metadata is None or missing task key
        """
        if metadata is None or "task" not in metadata:
            raise ValueError("metadata must contain a 'task' key")
        return metadata["task"]

    def run(self, task: str, response: str) -> float:
        """
        Generate a numeric value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide a value for

        Returns:
            Numeric value between min_score and max_score

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Create value prompt
        cache = self._state_manager.get("cache", {})
        prompt = cache.get("value_prompt_template", "").format(
            task=task,
            response=response,
        )

        # Generate value
        model = self._state_manager.get("model")
        value_text = model.generate(
            prompt,
            system_prompt=cache.get("system_prompt", ""),
            temperature=cache.get("temperature", 0.3),
            max_tokens=cache.get("max_tokens", 100),
        ).strip()

        # Parse value
        try:
            value = float(value_text)
            # Clamp value to range
            min_score = cache.get("min_score", 0.0)
            max_score = cache.get("max_score", 1.0)
            value = max(min_score, min(max_score, value))
        except ValueError:
            # Default value if parsing fails
            value = 0.5

        return value

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        self._check_input(text)
        # Value critics always return True for validation
        # as they focus on providing values rather than validation
        return True

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on value.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate value
        value = self.run(task, text)

        # Create improvement prompt
        prompt = (
            f"Task:\n{task}\n\n"
            f"Original response:\n{text}\n\n"
            f"Quality score: {value:.2f} (on a scale of 0 to 1)\n\n"
            f"Improved response:"
        )

        # Generate improved response
        improved_text = (
            self._state_manager.get("model")
            .generate(
                prompt,
                system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
            )
            .strip()
        )

        return improved_text

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide a value score.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate value
        value = self.run(task, text)

        # Create critique result
        return {
            "score": value,
            "feedback": f"Quality score: {value:.2f}",
            "issues": [],
            "suggestions": [],
        }


class LACCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the LLM-Based Actor-Critic (LAC) approach.

    This critic combines language feedback and value scoring to improve
    language model-based decision making.

    Based on: Language Feedback Improves Language Model-based Decision Making
    https://arxiv.org/abs/2403.03692

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - initialized: Initialization status
      - cache: Temporary data storage containing feedback and value critics
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: LACCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[LACCriticConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LAC critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: Language model provider to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_LAC_CONFIG

            config = DEFAULT_LAC_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store model in state
            self._state_manager.update("model", llm_provider)

            # Create feedback critic config
            feedback_config = FeedbackCriticConfig(
                name=f"{config.name}_feedback",
                description=f"Feedback component for {config.name}",
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                min_confidence=config.min_confidence,
                max_attempts=config.max_attempts,
                cache_size=config.cache_size,
                priority=config.priority,
                cost=config.cost,
                feedback_prompt_template=config.feedback_prompt_template,
                feedback_dimensions=config.feedback_dimensions,
                track_performance=config.track_performance,
                track_errors=config.track_errors,
            )

            # Create value critic config
            value_config = ValueCriticConfig(
                name=f"{config.name}_value",
                description=f"Value component for {config.name}",
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                min_confidence=config.min_confidence,
                max_attempts=config.max_attempts,
                cache_size=config.cache_size,
                priority=config.priority,
                cost=config.cost,
                value_prompt_template=config.value_prompt_template,
                value_dimensions=config.value_dimensions,
                min_score=config.min_score,
                max_score=config.max_score,
                track_performance=config.track_performance,
                track_errors=config.track_errors,
            )

            # Create feedback and value critics
            feedback_critic = create_feedback_critic(
                name=f"{config.name}_feedback",
                description=f"Feedback component for {config.name}",
                llm_provider=llm_provider,
                config=feedback_config,
            )

            value_critic = create_value_critic(
                name=f"{config.name}_value",
                description=f"Value component for {config.name}",
                llm_provider=llm_provider,
                config=value_config,
            )

            # Store components in cache
            cache = {
                "feedback_critic": feedback_critic,
                "value_critic": value_critic,
                "system_prompt": config.system_prompt,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize LACCritic: {str(e)}") from e

    def _check_input(self, text: str) -> None:
        """
        Validate input text and initialization state.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state_manager.get("initialized", False):
            raise RuntimeError("LACCritic not properly initialized")

    def _get_task_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        Extract task from metadata.

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Task string

        Raises:
            ValueError: If metadata is None or missing task key
        """
        if metadata is None or "task" not in metadata:
            raise ValueError("metadata must contain a 'task' key")
        return metadata["task"]

    def run(self, task: str, response: str) -> Dict[str, Any]:
        """
        Generate feedback and value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to provide feedback and value for

        Returns:
            Dictionary containing feedback and value

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Get feedback and value critics
        cache = self._state_manager.get("cache", {})
        feedback_critic = cache.get("feedback_critic")
        value_critic = cache.get("value_critic")

        # Generate feedback and value
        feedback = feedback_critic.run(task, response)
        value = value_critic.run(task, response)

        # Return results
        return {
            "feedback": feedback,
            "value": value,
        }

    def validate(self, text: str) -> bool:
        """
        Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        self._check_input(text)
        # LAC critics always return True for validation
        # as they focus on providing feedback and values rather than validation
        return True

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve text based on feedback and value.

        Args:
            text: The text to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback and value
        result = self.run(task, text)
        feedback = result["feedback"]
        value = result["value"]

        # Create improvement prompt
        prompt = (
            f"Task:\n{task}\n\n"
            f"Original response:\n{text}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Quality score: {value:.2f} (on a scale of 0 to 1)\n\n"
            f"Improved response:"
        )

        # Generate improved response
        model = self._state_manager.get("model")
        cache = self._state_manager.get("cache", {})
        improved_text = model.generate(
            prompt,
            system_prompt=cache.get("system_prompt", ""),
            temperature=cache.get("temperature", 0.7),
            max_tokens=cache.get("max_tokens", 1000),
        ).strip()

        return improved_text

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text and provide feedback and value.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing feedback and value

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get task from metadata
        task = self._get_task_from_metadata(metadata)

        # Generate feedback and value
        result = self.run(task, text)
        feedback = result["feedback"]
        value = result["value"]

        # Create critique result
        return {
            "score": value,
            "feedback": feedback,
            "value": value,
            "issues": [],
            "suggestions": [],
        }


def create_feedback_critic(
    llm_provider: Any,
    name: str = "feedback_critic",
    description: str = "Provides natural language feedback for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic:
    """
    Create a feedback critic with the given parameters.

    This factory function creates a configured feedback critic instance
    that provides natural language feedback for text.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        cache_size: The size of the cache
        priority: The priority of the critic
        cost: The cost of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        feedback_prompt_template: The template for feedback prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional configuration parameters

    Returns:
        A configured FeedbackCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_FEEDBACK_CONFIG

            config = DEFAULT_FEEDBACK_CONFIG.model_copy()

            # Update config with provided values
            updates = {}
            if name is not None:
                updates["name"] = name
            if description is not None:
                updates["description"] = description
            if system_prompt is not None:
                updates["system_prompt"] = system_prompt
            if temperature is not None:
                updates["temperature"] = temperature
            if max_tokens is not None:
                updates["max_tokens"] = max_tokens
            if min_confidence is not None:
                updates["min_confidence"] = min_confidence
            if max_attempts is not None:
                updates["max_attempts"] = max_attempts
            if cache_size is not None:
                updates["cache_size"] = cache_size
            if priority is not None:
                updates["priority"] = priority
            if cost is not None:
                updates["cost"] = cost
            if feedback_prompt_template is not None:
                updates["feedback_prompt_template"] = feedback_prompt_template

            # Add any additional kwargs
            updates.update(kwargs)

            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from ..config import FeedbackCriticConfig

            config = FeedbackCriticConfig(**config)

        # Create and return the critic
        return FeedbackCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            config=config,
        )
    except Exception as e:
        raise ValueError(f"Failed to create feedback critic: {str(e)}") from e


def create_value_critic(
    llm_provider: Any,
    name: str = "value_critic",
    description: str = "Provides numeric value scoring for text",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    value_prompt_template: str = None,
    min_score: float = None,
    max_score: float = None,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic:
    """
    Create a value critic with the given parameters.

    This factory function creates a configured value critic instance
    that provides numeric value scoring for text.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        cache_size: The size of the cache
        priority: The priority of the critic
        cost: The cost of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        value_prompt_template: The template for value prompts
        min_score: The minimum score value
        max_score: The maximum score value
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional configuration parameters

    Returns:
        A configured ValueCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_VALUE_CONFIG

            config = DEFAULT_VALUE_CONFIG.model_copy()

            # Update config with provided values
            updates = {}
            if name is not None:
                updates["name"] = name
            if description is not None:
                updates["description"] = description
            if system_prompt is not None:
                updates["system_prompt"] = system_prompt
            if temperature is not None:
                updates["temperature"] = temperature
            if max_tokens is not None:
                updates["max_tokens"] = max_tokens
            if min_confidence is not None:
                updates["min_confidence"] = min_confidence
            if max_attempts is not None:
                updates["max_attempts"] = max_attempts
            if cache_size is not None:
                updates["cache_size"] = cache_size
            if priority is not None:
                updates["priority"] = priority
            if cost is not None:
                updates["cost"] = cost
            if value_prompt_template is not None:
                updates["value_prompt_template"] = value_prompt_template
            if min_score is not None:
                updates["min_score"] = min_score
            if max_score is not None:
                updates["max_score"] = max_score

            # Add any additional kwargs
            updates.update(kwargs)

            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from ..config import ValueCriticConfig

            config = ValueCriticConfig(**config)

        # Create and return the critic
        return ValueCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            config=config,
        )
    except Exception as e:
        raise ValueError(f"Failed to create value critic: {str(e)}") from e


def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    feedback_prompt_template: str = None,
    value_prompt_template: str = None,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic:
    """
    Create a LAC critic with the given parameters.

    This factory function creates a configured LAC critic instance
    that combines language feedback and value scoring.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        cache_size: The size of the cache
        priority: The priority of the critic
        cost: The cost of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        feedback_prompt_template: The template for feedback prompts
        value_prompt_template: The template for value prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional configuration parameters

    Returns:
        A configured LACCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_LAC_CONFIG

            config = DEFAULT_LAC_CONFIG.model_copy()

            # Update config with provided values
            updates = {}
            if name is not None:
                updates["name"] = name
            if description is not None:
                updates["description"] = description
            if system_prompt is not None:
                updates["system_prompt"] = system_prompt
            if temperature is not None:
                updates["temperature"] = temperature
            if max_tokens is not None:
                updates["max_tokens"] = max_tokens
            if min_confidence is not None:
                updates["min_confidence"] = min_confidence
            if max_attempts is not None:
                updates["max_attempts"] = max_attempts
            if cache_size is not None:
                updates["cache_size"] = cache_size
            if priority is not None:
                updates["priority"] = priority
            if cost is not None:
                updates["cost"] = cost
            if feedback_prompt_template is not None:
                updates["feedback_prompt_template"] = feedback_prompt_template
            if value_prompt_template is not None:
                updates["value_prompt_template"] = value_prompt_template

            # Add any additional kwargs
            updates.update(kwargs)

            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from ..config import LACCriticConfig

            config = LACCriticConfig(**config)

        # Create and return the critic
        return LACCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            config=config,
        )
    except Exception as e:
        raise ValueError(f"Failed to create LAC critic: {str(e)}") from e
