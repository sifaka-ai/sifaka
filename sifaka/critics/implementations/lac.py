"""
LAC (LLM-Based Actor-Critic) Critic Module

A module implementing the LAC approach for critics, which combines language feedback
and value scoring to improve language model-based decision making.

## Overview
This module provides a comprehensive implementation of the LLM-Based Actor-Critic (LAC)
approach for text evaluation and improvement. It combines natural language feedback
with numeric value scoring to enhance language model outputs.

Based on: Language Feedback Improves Language Model-based Decision Making
https://arxiv.org/abs/2403.03692

## Components
- FeedbackCritic: Provides detailed natural language feedback on text
- ValueCritic: Assigns numeric quality scores to text
- LACCritic: Combines feedback and value critics for comprehensive evaluation
- Factory functions for creating configured critic instances

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

## Usage Examples
```python
from sifaka.critics.implementations.lac import create_lac_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a LAC critic
critic = create_lac_critic(llm_provider=provider)

# Use the critic to improve text
task = "Summarize the causes of World War I in 3 bullet points."
response = provider.generate(f"Task:\n{task}") if provider else ""
results = critic.critique(response, {"task": task}) if critic else ""

print("Feedback:", results["feedback"])
print("Value Score:", results["value"])
```

## Error Handling
The module implements comprehensive error handling with specific exceptions for:
- Configuration errors (ValueError)
- Initialization failures (RuntimeError)
- Provider compatibility issues (TypeError)
- Input validation errors (ValueError)

## Configuration
All critics support configuration through Pydantic models with options for:
- Language model provider settings
- Prompt templates
- Temperature and token limits
- Performance tracking
- Caching behavior
"""

import time
from typing import Any, Dict, Optional, Union, List, cast

from pydantic import Field, PrivateAttr, ConfigDict

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...core.base import BaseResult
from sifaka.utils.config.critics import FeedbackCriticConfig, ValueCriticConfig, LACCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult

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


class FeedbackCritic(BaseComponent[str, BaseResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that produces natural language feedback for a model's response to a task.

    This critic analyzes text and provides detailed feedback on what could be
    improved or what was done well. It focuses on qualitative assessment rather
    than numeric scoring.

    ## Architecture
    The FeedbackCritic implements multiple interfaces:
    - BaseComponent: Core component functionality
    - TextValidator: Text quality validation
    - TextImprover: Text improvement capabilities
    - TextCritic: Text critique and analysis

    ## Lifecycle
    1. **Initialization**: Configure with LLM provider and settings
    2. **Operation**: Process text through validation, critique, or improvement
    3. **Cleanup**: Release resources and reset state

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

    ## Error Handling
    - ValueError: For invalid inputs or configuration
    - RuntimeError: For initialization or processing failures
    - All errors are tracked if track_errors is enabled

    ## Examples
    ```python
    from sifaka.critics.implementations.lac import create_feedback_critic
    from sifaka.models.providers import OpenAIProvider

    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_feedback_critic(llm_provider=provider)

    task = "Write a concise summary of quantum computing."
    response = "Quantum computers use qubits."

    feedback = critic.run(task, response) if critic else ""
    print(feedback)
    ```

    Attributes:
        config (FeedbackCriticConfig): Configuration settings for the critic
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
            from sifaka.utils.config.critics import DEFAULT_FEEDBACK_CRITIC_CONFIG
            from copy import deepcopy

            # Create a copy of the default config and update it
            config_dict = deepcopy(DEFAULT_FEEDBACK_CRITIC_CONFIG)
            config_dict.update({"name": name, "description": description, **kwargs})

            # Create a new FeedbackCriticConfig from the updated dict
            from sifaka.utils.config.critics import FeedbackCriticConfig

            config = FeedbackCriticConfig(**config_dict)

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            if hasattr(self, "_state_manager"):
                self._state_manager.update("model", llm_provider)

                # Store configuration in cache
                # Ensure config is a FeedbackCriticConfig
                feedback_config = cast(FeedbackCriticConfig, config)

                # Check if feedback_prompt_template exists, otherwise use a default
                feedback_prompt_template = getattr(
                    feedback_config,
                    "feedback_prompt_template",
                    getattr(feedback_config, "feedback_prompt", DEFAULT_FEEDBACK_PROMPT_TEMPLATE),
                )

                cache = {
                    "feedback_prompt_template": feedback_prompt_template,
                    "system_prompt": feedback_config.system_prompt,
                    "temperature": feedback_config.temperature,
                    "max_tokens": feedback_config.max_tokens,
                }
                self._state_manager.update("cache", cache)

                # Mark as initialized
                self._state_manager.update("initialized", True)
                self._state_manager.set_metadata("component_type", self.__class__.__name__)
                self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            if hasattr(self, "record_error"):
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
        if not isinstance(text, str) or not (text.strip() if text else ""):
            raise ValueError("text must be a non-empty string")

        if not hasattr(self, "_state_manager") or not self._state_manager.get("initialized", False):
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

    def process(self, input: str) -> BaseResult:
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
            if not isinstance(input, str) or not (input.strip() if input else ""):
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not hasattr(self, "_state_manager") or not self._state_manager.get(
                "initialized", False
            ):
                raise RuntimeError("FeedbackCritic not properly initialized")

            # Create a default task if none provided
            task = "Provide feedback on the following text"

            # Generate feedback
            feedback = self.run(task, input) if hasattr(self, "run") else ""

            # Create result
            result = BaseResult(
                passed=True,  # Feedback critics always pass
                message=feedback,
                metadata={"operation": "process"},
                score=0.5,  # Default score
                issues=[],
                suggestions=[],
                processing_time_ms=((time.time() - start_time) * 1000),
            )

            # Update statistics
            if hasattr(self, "update_statistics"):
                self.update_statistics(result)

            return result

        except Exception as e:
            if hasattr(self, "record_error"):
                self.record_error(e)
            processing_time = (time.time() - start_time) * 1000
            return BaseResult(
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
            if hasattr(self, "_check_input"):
                self._check_input(response)

            # Get cache from state
            cache = self._state_manager.get("cache", {}) if hasattr(self, "_state_manager") else {}

            # Create feedback prompt
            template = (
                cache.get("feedback_prompt_template", DEFAULT_FEEDBACK_PROMPT_TEMPLATE)
                if cache
                else DEFAULT_FEEDBACK_PROMPT_TEMPLATE
            )
            prompt = template.format(
                task=task,
                response=response,
            )

            # Get model from state
            model = self._state_manager.get("model") if hasattr(self, "_state_manager") else None
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
            if (
                hasattr(self, "config")
                and hasattr(self.config, "track_performance")
                and self.config.track_performance
            ):
                if hasattr(self, "_state_manager"):
                    total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                    self._state_manager.set_metadata(
                        "total_processing_time_ms", total_time + ((time.time() - start_time) * 1000)
                    )

            if hasattr(self, "_state_manager"):
                feedback_count = self._state_manager.get_metadata("feedback_count", 0)
                self._state_manager.set_metadata("feedback_count", feedback_count + 1)

            return feedback

        except Exception as e:
            if hasattr(self, "record_error"):
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
            if hasattr(self, "_check_input"):
                self._check_input(response)

            # Get cache from state
            cache = self._state_manager.get("cache", {}) if hasattr(self, "_state_manager") else {}

            # Create feedback prompt
            template = (
                cache.get("feedback_prompt_template", DEFAULT_FEEDBACK_PROMPT_TEMPLATE)
                if cache
                else DEFAULT_FEEDBACK_PROMPT_TEMPLATE
            )
            prompt = template.format(
                task=task,
                response=response,
            )

            # Get model from state
            model = self._state_manager.get("model") if hasattr(self, "_state_manager") else None
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
                feedback = feedback.strip() if feedback else ""
            else:
                # Fallback to sync method in async context
                import asyncio

                # Create a wrapper function for asyncio.run_in_executor
                loop = asyncio.get_event_loop()
                feedback = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        prompt,
                        system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                        temperature=cache.get("temperature", 0.7),
                        max_tokens=cache.get("max_tokens", 1000),
                    ),
                )
                feedback = feedback.strip() if feedback else ""

            # Update statistics
            if hasattr(self, "config") and self.config.track_performance:
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

            if hasattr(self, "config") and self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return True

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def improve(self, text: str, feedback: str = "") -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (not used directly, metadata is used instead)

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # This implementation uses metadata instead of direct feedback parameter
        metadata = {"task": "Improve the text"} if not feedback else {"task": feedback}
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
            if not isinstance(feedback, str) or not (feedback.strip() if feedback else ""):
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

    def critique(self, text: str) -> CritiqueResult:
        """
        Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult containing feedback

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Use a default task since the interface doesn't allow for metadata
        metadata = {"task": "Provide feedback on the text"}
        start_time = time.time()

        try:
            self._check_input(text)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Generate feedback
            feedback = self.run(task, text)

            # Create critique result
            result: CritiqueResult = {
                "score": 0.5,  # Default score since feedback critics don't provide scores
                "feedback": feedback,
                "issues": [],
                "suggestions": [],
            }

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            if hasattr(self, "config") and self.config.track_performance:
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

            if hasattr(self, "config") and self.config.track_performance:
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

                # Create a wrapper function for asyncio.run_in_executor
                loop = asyncio.get_event_loop()
                improved_text = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        prompt,
                        system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                        temperature=cache.get("temperature", 0.7),
                        max_tokens=cache.get("max_tokens", 1000),
                    ),
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
    ) -> CritiqueResult:
        """
        Asynchronously analyze text and provide detailed feedback.

        Args:
            text: The text to critique
            metadata: Optional metadata containing the task

        Returns:
            CritiqueResult containing feedback

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
            result: CritiqueResult = {
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
        if stats:
            stats.update(
                {
                    "critique_count": self._state_manager.get_metadata("critique_count", 0),
                    "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
                    "feedback_count": self._state_manager.get_metadata("feedback_count", 0),
                    "last_improvement_time": self._state_manager.get_metadata(
                        "last_improvement_time"
                    ),
                    "model_provider": (
                        str(self._state_manager.get("model").__class__.__name__)
                        if self._state_manager.get("model")
                        else None
                    ),
                }
            )

        return stats


class ValueCritic(BaseComponent[str, BaseResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that estimates a numeric value for a model's response to a task.

    This critic analyzes text and provides a numeric score (e.g., probability of success)
    for the response. It focuses on quantitative assessment rather than qualitative feedback.

    ## Architecture
    The ValueCritic implements multiple interfaces:
    - BaseComponent: Core component functionality
    - TextValidator: Text quality validation
    - TextImprover: Text improvement capabilities
    - TextCritic: Text critique and analysis

    ## Lifecycle
    1. **Initialization**: Configure with LLM provider and scoring settings
    2. **Operation**: Process text to generate numeric scores
    3. **Cleanup**: Release resources and reset state

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

    ## Error Handling
    - ValueError: For invalid inputs or configuration
    - RuntimeError: For initialization or processing failures
    - All errors are tracked if track_errors is enabled

    ## Examples
    ```python
    from sifaka.critics.implementations.lac import create_value_critic
    from sifaka.models.providers import OpenAIProvider

    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_value_critic(llm_provider=provider)

    task = "Write a concise summary of quantum computing."
    response = "Quantum computers use qubits."

    score = critic.run(task, response) if critic else ""
    print(f"Quality score: {score:.2f}")
    ```

    Attributes:
        config (ValueCriticConfig): Configuration settings for the critic
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
            from sifaka.utils.config.critics import DEFAULT_VALUE_CRITIC_CONFIG
            from copy import deepcopy

            # Create a copy of the default config and update it
            config_dict = deepcopy(DEFAULT_VALUE_CRITIC_CONFIG)
            config_dict.update({"name": name, "description": description, **kwargs})

            # Create a new ValueCriticConfig from the updated dict
            from sifaka.utils.config.critics import ValueCriticConfig

            config = ValueCriticConfig(**config_dict)

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store components in state
            self._state_manager.update("model", llm_provider)

            # Store configuration in cache
            # Ensure config is a ValueCriticConfig
            value_config = cast(ValueCriticConfig, config)

            # Check if value_prompt_template exists, otherwise use a default
            value_prompt_template = getattr(
                value_config,
                "value_prompt_template",
                getattr(value_config, "value_prompt", DEFAULT_VALUE_PROMPT_TEMPLATE),
            )

            cache = {
                "value_prompt_template": value_prompt_template,
                "system_prompt": value_config.system_prompt,
                "temperature": value_config.temperature,
                "max_tokens": value_config.max_tokens,
                "min_score": value_config.min_score,
                "max_score": value_config.max_score,
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
        if not isinstance(text, str) or not (text.strip() if text else ""):
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

    def improve(self, text: str, feedback: str = "") -> str:
        """
        Improve text based on value.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (not used directly, metadata is used instead)

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # This implementation uses metadata instead of direct feedback parameter
        metadata = {"task": "Improve the text"} if not feedback else {"task": feedback}
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

    def critique(self, text: str) -> CritiqueResult:
        """
        Analyze text and provide a value score.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult containing score

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Use a default task since the interface doesn't allow for metadata
        metadata = {"task": "Evaluate the text"}
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


class LACCritic(BaseComponent[str, BaseResult], TextValidator, TextImprover, TextCritic):
    """
    A critic that implements the LLM-Based Actor-Critic (LAC) approach.

    This critic combines language feedback and value scoring to improve
    language model-based decision making. It integrates both qualitative feedback
    and quantitative assessment for comprehensive text evaluation.

    Based on: Language Feedback Improves Language Model-based Decision Making
    https://arxiv.org/abs/2403.03692

    ## Architecture
    The LACCritic implements multiple interfaces:
    - BaseComponent: Core component functionality
    - TextValidator: Text quality validation
    - TextImprover: Text improvement capabilities
    - TextCritic: Text critique and analysis

    It composes two specialized critics:
    - FeedbackCritic: For qualitative natural language feedback
    - ValueCritic: For quantitative scoring

    ## Lifecycle
    1. **Initialization**: Configure with LLM provider and create sub-critics
    2. **Operation**: Process text through both feedback and value components
    3. **Cleanup**: Release resources and reset state for all components

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

    ## Error Handling
    - ValueError: For invalid inputs or configuration
    - RuntimeError: For initialization or processing failures
    - All errors are tracked if track_errors is enabled

    ## Examples
    ```python
    from sifaka.critics.implementations.lac import create_lac_critic
    from sifaka.models.providers import OpenAIProvider

    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_lac_critic(llm_provider=provider)

    task = "Write a concise summary of quantum computing."
    response = "Quantum computers use qubits."

    result = critic.critique(response, {"task": task}) if critic else ""
    print(f"Feedback: {result['feedback']}")
    print(f"Value: {result['value']}")
    ```

    Attributes:
        config (LACCriticConfig): Configuration settings for the critic
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
        config: Optional[Optional[LACCriticConfig]] = None,
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
            from sifaka.utils.config.critics import DEFAULT_LAC_CRITIC_CONFIG

            config = DEFAULT_LAC_CRITIC_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Store model in state
            self._state_manager.update("model", llm_provider)

            # Create feedback critic config
            feedback_config = FeedbackCriticConfig(
                name=f"{config.name}_feedback" if hasattr(config, "name") else "feedback_component",
                description=(
                    f"Feedback component for {config.name}"
                    if hasattr(config, "name")
                    else "Feedback component"
                ),
                system_prompt=config.system_prompt if config else None,
                temperature=config.temperature if config else None,
                max_tokens=config.max_tokens if config else None,
                min_confidence=config.min_confidence if config else None,
                max_attempts=config.max_attempts if config else None,
                cache_size=config.cache_size if config else None,
                priority=config.priority if config else None,
                cost=config.cost if config else None,
                feedback_prompt_template=config.feedback_prompt_template if config else None,
                feedback_dimensions=config.feedback_dimensions if config else None,
                track_performance=config.track_performance if config else None,
                track_errors=config.track_errors if config else None,
            )

            # Create value critic config
            value_config = ValueCriticConfig(
                name=f"{config.name}_value" if hasattr(config, "name") else "value_component",
                description=(
                    f"Value component for {config.name}"
                    if hasattr(config, "name")
                    else "Value component"
                ),
                system_prompt=config.system_prompt if config else None,
                temperature=config.temperature if config else None,
                max_tokens=config.max_tokens if config else None,
                min_confidence=config.min_confidence if config else None,
                max_attempts=config.max_attempts if config else None,
                cache_size=config.cache_size if config else None,
                priority=config.priority if config else None,
                cost=config.cost if config else None,
                value_prompt_template=config.value_prompt_template if config else None,
                value_dimensions=config.value_dimensions if config else None,
                min_score=config.min_score if config else None,
                max_score=config.max_score if config else None,
                track_performance=config.track_performance if config else None,
                track_errors=config.track_errors if config else None,
            )

            # Create feedback and value critics
            feedback_critic = create_feedback_critic(
                name=f"{config.name}_feedback" if hasattr(config, "name") else "feedback_component",
                description=(
                    f"Feedback component for {config.name}"
                    if hasattr(config, "name")
                    else "Feedback component"
                ),
                llm_provider=llm_provider,
                config=feedback_config,
            )

            value_critic = create_value_critic(
                name=f"{config.name}_value" if hasattr(config, "name") else "value_component",
                description=(
                    f"Value component for {config.name}"
                    if hasattr(config, "name")
                    else "Value component"
                ),
                llm_provider=llm_provider,
                config=value_config,
            )

            # Store components in cache
            cache = {
                "feedback_critic": feedback_critic,
                "value_critic": value_critic,
                "system_prompt": config.system_prompt if config else None,
                "temperature": config.temperature if config else None,
                "max_tokens": config.max_tokens if config else None,
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
        if not isinstance(text, str) or not (text.strip() if text else ""):
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
        feedback = feedback_critic.run(task, response) if feedback_critic else ""
        value = value_critic.run(task, response) if value_critic else 0.0

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

    def improve(self, text: str, feedback: str = "") -> str:
        """
        Improve text based on feedback and value.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (not used directly, metadata is used instead)

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # This implementation uses metadata instead of direct feedback parameter
        metadata = {"task": "Improve the text"} if not feedback else {"task": feedback}
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

    def critique(self, text: str) -> CritiqueResult:
        """
        Analyze text and provide feedback and value.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult containing feedback and value

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        # Use a default task since the interface doesn't allow for metadata
        metadata = {"task": "Evaluate the text"}
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
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    feedback_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic:
    """
    Create a feedback critic with the given parameters.

    This factory function creates a configured feedback critic instance
    that provides natural language feedback for text. It handles configuration
    creation, validation, and critic instantiation.

    Detailed description of what the method does:
    - Creates a default configuration if none is provided
    - Updates configuration with any provided parameters
    - Validates the configuration
    - Instantiates and returns a FeedbackCritic

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
        FeedbackCritic: A configured FeedbackCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider

    Example:
        ```python
        from sifaka.critics.implementations.lac import create_feedback_critic
        from sifaka.models.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="your-api-key")

        # Create with default settings
        critic = create_feedback_critic(llm_provider=provider)

        # Create with custom settings
        custom_critic = create_feedback_critic(
            llm_provider=provider,
            name="custom_feedback",
            temperature=0.5,
            max_tokens=500
        )
        ```
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_FEEDBACK_CRITIC_CONFIG

            config = DEFAULT_FEEDBACK_CRITIC_CONFIG.model_copy()

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
            from sifaka.utils.config.critics import FeedbackCriticConfig

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
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    value_prompt_template: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic:
    """
    Create a value critic with the given parameters.

    This factory function creates a configured value critic instance
    that provides numeric value scoring for text. It handles configuration
    creation, validation, and critic instantiation.

    Detailed description of what the method does:
    - Creates a default configuration if none is provided
    - Updates configuration with any provided parameters
    - Validates the configuration
    - Instantiates and returns a ValueCritic

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
        ValueCritic: A configured ValueCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider

    Example:
        ```python
        from sifaka.critics.implementations.lac import create_value_critic
        from sifaka.models.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="your-api-key")

        # Create with default settings
        critic = create_value_critic(llm_provider=provider)

        # Create with custom settings
        custom_critic = create_value_critic(
            llm_provider=provider,
            name="custom_value",
            temperature=0.3,
            min_score=0.0,
            max_score=10.0
        )
        ```
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_VALUE_CRITIC_CONFIG

            config = DEFAULT_VALUE_CRITIC_CONFIG.model_copy()

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
            from sifaka.utils.config.critics import ValueCriticConfig

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
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    feedback_prompt_template: Optional[str] = None,
    value_prompt_template: Optional[str] = None,
    config: Any = None,
    **kwargs: Any,
) -> LACCritic:
    """
    Create a LAC critic with the given parameters.

    This factory function creates a configured LAC critic instance
    that combines language feedback and value scoring. It implements
    the LLM-Based Actor-Critic approach for comprehensive text evaluation.

    Detailed description of what the method does:
    - Creates a default configuration if none is provided
    - Updates configuration with any provided parameters
    - Validates the configuration
    - Creates and configures feedback and value sub-critics
    - Instantiates and returns a LACCritic

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
        LACCritic: A configured LACCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider

    Example:
        ```python
        from sifaka.critics.implementations.lac import create_lac_critic
        from sifaka.models.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="your-api-key")

        # Create with default settings
        critic = create_lac_critic(llm_provider=provider)

        # Create with custom settings
        custom_critic = create_lac_critic(
            llm_provider=provider,
            name="custom_lac",
            temperature=0.5,
            feedback_prompt_template="Provide detailed feedback on this text: {response}"
        )

        # Use the critic
        result = critic.critique("This is a test", {"task": "Evaluate this text"})
        ```
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config.critics import DEFAULT_LAC_CRITIC_CONFIG

            config = DEFAULT_LAC_CRITIC_CONFIG.model_copy()

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
            from sifaka.utils.config.critics import LACCriticConfig

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
