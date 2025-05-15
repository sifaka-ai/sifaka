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
from typing import Any, Dict, Optional, Union, List, Set, cast, TypedDict

from pydantic import Field, PrivateAttr, ConfigDict

from ...core.base import BaseComponent, BaseConfig
from ...utils.state import create_critic_state, StateManager
from ...core.base import BaseResult
from sifaka.utils.config.critics import FeedbackCriticConfig, ValueCriticConfig, LACCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult

# Add type ignores for imports used at runtime
if False:  # This will never execute, just for mypy
    from ..implementations.factories import create_feedback_critic, create_value_critic

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
    A critic that provides natural language feedback on text quality.

    This critic uses a language model to generate detailed feedback on text,
    focusing on aspects like clarity, coherence, and effectiveness. It can
    be used to improve text quality through iterative feedback loops.

    Features:
    - Natural language feedback generation
    - Text validation and improvement
    - Performance tracking
    - Error handling and recovery
    - State management

    Usage:
        critic = FeedbackCritic()
        feedback = critic.run(task, response)
        improved_text = critic.improve(response, feedback)
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration as a private attribute that won't conflict with BaseComponent's config
    _critic_config: FeedbackCriticConfig = PrivateAttr()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feedback critic.

        Args:
            config: Optional configuration dictionary
        """
        name = "feedback_critic"
        description = "Provides natural language feedback for text"
        base_config = BaseConfig(name=name, description=description)
        super().__init__(name=name, description=description, config=base_config)

        # Create a proper config object if it's a dict, otherwise use default
        if isinstance(config, dict):
            self._critic_config = FeedbackCriticConfig(**config)
        else:
            self._critic_config = FeedbackCriticConfig()

        self._initialize_components()

    # Property that accesses the private critic config instead of overriding BaseComponent's config
    def get_critic_config(self) -> FeedbackCriticConfig:
        """Get the critic configuration."""
        return self._critic_config

    def process(self, input_text: str) -> BaseResult:
        """Process input text and return a result.

        This method processes the input text and returns a basic result.
        It implements the abstract method from BaseComponent.

        Args:
            input_text: The text to process

        Returns:
            BaseResult: The processing result
        """
        result = self.critique(input_text)
        return BaseResult(
            passed=True,
            message="Feedback generated successfully",
            metadata={"feedback": result.get("feedback", ""), "score": result.get("score", 0.0)},
        )

    def _initialize_components(self) -> None:
        """Initialize critic components."""
        # Initialize cache
        self._state_manager.update(
            "cache",
            {
                "feedback_prompt_template": DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        # Mark as initialized
        self._state_manager.update("initialized", True)

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
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Generate feedback
            feedback = model.generate(
                prompt,
                system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            )

            # Ensure we return a string type
            result = ""
            if feedback is not None:
                result = str(feedback).strip()

            # Update statistics
            if hasattr(self, "_critic_config") and self._critic_config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                total_time_ms = total_time + (time.time() - start_time) * 1000
                self._state_manager.set_metadata("total_processing_time_ms", total_time_ms)

            feedback_count = self._state_manager.get_metadata("feedback_count", 0)
            self._state_manager.set_metadata("feedback_count", feedback_count + 1)

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to generate feedback: {str(e)}") from e

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
                total_time_ms = total_time + (time.time() - start_time) * 1000
                self._state_manager.set_metadata("total_processing_time_ms", total_time_ms)

            return True

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def improve(self, text: str, feedback: str = "") -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get components
        cache = self._state_manager.get("cache", {})
        feedback_critic = cache.get("feedback_critic")
        value_critic = cache.get("value_critic")

        # Create components if not initialized
        if not feedback_critic:
            # Import here to avoid circular import issues
            from ..implementations.factories import create_feedback_critic as create_fb_critic  # type: ignore

            model = self._state_manager.get("model")
            feedback_critic = create_fb_critic(
                llm_provider=model,
                name=f"{self.name}_feedback",
                description=f"Feedback component for {self.name}",
            )
            cache["feedback_critic"] = feedback_critic
            self._state_manager.update("cache", cache)

        # Improve text
        if not feedback:
            # Generate feedback if not provided
            task = "Improve the following text"
            feedback = str(feedback_critic.run(task, text) if feedback_critic else "")

        # Get improved text using feedback critic
        if feedback_critic:
            improved_text = feedback_critic.improve(text, feedback)
            # Make sure we always return a string
            return str(improved_text if improved_text is not None else "")
        else:
            # If no feedback_critic is available, return the original text
            return str(text)

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
        start_time = time.time()

        try:
            self._check_input(text)

            # Generate feedback for a default task
            task = "Evaluate the following text"
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

            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                total_time_ms = total_time + (time.time() - start_time) * 1000
                self._state_manager.set_metadata("total_processing_time_ms", total_time_ms)

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    def _check_input(self, text: str) -> None:
        """Validate input text.

        Args:
            text: The text to validate

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

    def record_error(self, error: Exception) -> None:
        """Record an error in the state manager.

        Args:
            error: The error to record
        """
        if hasattr(self, "config") and self.config.track_errors:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(error))
            self._state_manager.set_metadata("last_error_time", time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self._state_manager.get_metadata("validation_count", 0),
            "critique_count": self._state_manager.get_metadata("critique_count", 0),
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "feedback_count": self._state_manager.get_metadata("feedback_count", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "total_processing_time_ms": self._state_manager.get_metadata(
                "total_processing_time_ms", 0
            ),
        }


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

    # Configuration as a private attribute
    _critic_config: ValueCriticConfig = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
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
        base_config = BaseConfig(name=name, description=description)
        super().__init__(name=name, description=description, config=base_config)

        # Initialize proper configuration
        if isinstance(config, dict):
            self._critic_config = ValueCriticConfig(**config)
        elif isinstance(config, ValueCriticConfig):
            self._critic_config = config
        else:
            self._critic_config = ValueCriticConfig()

        # Apply any additional kwargs to the config
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self._critic_config, key):
                    object.__setattr__(self._critic_config, key, value)

        # Store the LLM provider
        self._state_manager.update("model", llm_provider)
        self._state_manager.set_metadata("model_type", type(llm_provider).__name__)

        # Initialize components
        self._initialize_components()

    # Property that accesses the private critic config
    def get_critic_config(self) -> ValueCriticConfig:
        """Get the critic configuration."""
        return self._critic_config

    def process(self, input_text: str) -> BaseResult:
        """Process input text and return a result.

        This method processes the input text and returns a basic result.
        It implements the abstract method from BaseComponent.

        Args:
            input_text: The text to process

        Returns:
            BaseResult: The processing result
        """
        result = self.critique(input_text)
        return BaseResult(
            passed=True,
            message="Value score generated successfully",
            metadata={"score": result.get("score", 0.0)},
        )

    def _initialize_components(self) -> None:
        """Initialize critic components."""
        # Initialize cache
        self._state_manager.update(
            "cache",
            {
                "value_prompt_template": DEFAULT_VALUE_PROMPT_TEMPLATE,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        # Mark as initialized
        self._state_manager.update("initialized", True)

    def run(self, task: str, response: str) -> float:
        """
        Generate a value score for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to score

        Returns:
            Value score between 0 and 1

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(response)

            # Get cache from state
            cache = self._state_manager.get("cache", {})

            # Create value prompt
            template = cache.get("value_prompt_template", DEFAULT_VALUE_PROMPT_TEMPLATE)
            prompt = template.format(
                task=task,
                response=response,
            )

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Generate value
            value_text = model.generate(
                prompt,
                system_prompt=cache.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                temperature=cache.get("temperature", 0.7),
                max_tokens=cache.get("max_tokens", 1000),
            )

            # Extract numeric value
            import re

            value_text = str(value_text).strip()
            number_pattern = r"\d+(\.\d+)?"
            numbers = re.findall(number_pattern, value_text)

            # Use the first number found, or 5 as a fallback (middle of scale)
            try:
                value = float(re.findall(number_pattern, value_text)[0]) if numbers else 5.0
            except (ValueError, IndexError):
                value = 5.0

            # Scale to 0-1
            min_score = cache.get("min_score", 0.0)
            max_score = cache.get("max_score", 10.0)
            value = (value - min_score) / (max_score - min_score)
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]

            # Update statistics
            if hasattr(self, "_critic_config") and self._critic_config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                total_time_ms = total_time + (time.time() - start_time) * 1000
                self._state_manager.set_metadata("total_processing_time_ms", total_time_ms)

            # Return the value
            return value

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to generate value score: {str(e)}") from e

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
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            self._check_input(text)

            # Generate task and score
            task = "Improve the following text"
            score = 0.0
            if not feedback:
                try:
                    score_result = self.run(task, text)
                    score = float(score_result)
                except (ValueError, TypeError):
                    score = 0.5  # Default score if conversion fails

            # Create improvement prompt
            prompt = f"Task: {task}\n\n" f"Original text: {text}\n\n"

            if feedback:
                prompt += f"Feedback: {feedback}\n\n"
            else:
                prompt += f"Current quality score: {score:.1f}/10\n\n"

            prompt += "Please provide an improved version:"

            # Get model from state
            model = self._state_manager.get("model")
            if not model:
                raise RuntimeError("Model not initialized")

            # Generate improved text
            result_text = model.generate(
                prompt,
                system_prompt="You are a helpful assistant that improves text quality.",
                temperature=0.7,
                max_tokens=1000,
            )

            # Ensure we return a string
            if result_text is None:
                return text  # Return original if generation failed

            # Cast to string and return
            return str(result_text).strip()

        except Exception as e:
            self.record_error(e)
            # Return original text on error
            return text

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
        self._check_input(text)

        # Use a default task
        task = "Evaluate the text"

        # Generate value
        value = self.run(task, text)

        # Create critique result
        return {
            "score": value,
            "feedback": f"Quality score: {value:.2f}",
            "issues": [],
            "suggestions": [],
        }

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

    def record_error(self, error: Exception) -> None:
        """Record an error in the state manager.

        Args:
            error: The error to record
        """
        if hasattr(self, "config") and self.config.track_errors:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(error))
            self._state_manager.set_metadata("last_error_time", time.time())


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

    # Configuration as a private attribute
    _critic_config: LACCriticConfig = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
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
        base_config = BaseConfig(name=name, description=description)
        super().__init__(name=name, description=description, config=base_config)

        # Initialize proper configuration
        if isinstance(config, dict):
            self._critic_config = LACCriticConfig(**config)
        elif isinstance(config, LACCriticConfig):
            self._critic_config = config
        else:
            self._critic_config = LACCriticConfig()

        # Apply any additional kwargs to the config
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self._critic_config, key):
                    object.__setattr__(self._critic_config, key, value)

        # Store the LLM provider
        self._state_manager.update("model", llm_provider)
        self._state_manager.set_metadata("model_type", type(llm_provider).__name__)

        # Initialize components
        self._initialize_components()

    # Property that accesses the private critic config
    def get_critic_config(self) -> LACCriticConfig:
        """Get the critic configuration."""
        return self._critic_config

    def process(self, input_text: str) -> BaseResult:
        """Process input text and return a result.

        This method processes the input text and returns a basic result.
        It implements the abstract method from BaseComponent.

        Args:
            input_text: The text to process

        Returns:
            BaseResult: The processing result
        """
        result = self.critique(input_text)
        return BaseResult(
            passed=True,
            message="LAC critique generated successfully",
            metadata={"feedback": result.get("feedback", ""), "score": result.get("score", 0.0)},
        )

    def _initialize_components(self) -> None:
        """Initialize critic components."""
        # Initialize cache
        self._state_manager.update(
            "cache",
            {
                "feedback_critic": None,
                "value_critic": None,
                "system_prompt": self._critic_config.system_prompt,
                "temperature": self._critic_config.temperature,
                "max_tokens": self._critic_config.max_tokens,
            },
        )
        # Mark as initialized
        self._state_manager.update("initialized", True)

    def run(self, task: str, response: str) -> Dict[str, Any]:
        """
        Generate feedback and value for a response to a task.

        Args:
            task: The task that the response is addressing
            response: The response to evaluate

        Returns:
            Dictionary with feedback and value

        Raises:
            ValueError: If response is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(response)

        # Get components
        cache = self._state_manager.get("cache", {})
        feedback_critic = cache.get("feedback_critic")
        value_critic = cache.get("value_critic")

        # Create components if not initialized
        if not feedback_critic or not value_critic:
            # Import here to avoid circular import issues
            from ..implementations.factories import create_feedback_critic as create_fb_critic  # type: ignore
            from ..implementations.factories import create_value_critic as create_val_critic  # type: ignore

            model = self._state_manager.get("model")

            if not feedback_critic:
                feedback_critic = create_fb_critic(
                    llm_provider=model,
                    name=f"{self.name}_feedback",
                    description=f"Feedback component for {self.name}",
                )
                cache["feedback_critic"] = feedback_critic

            if not value_critic:
                value_critic = create_val_critic(
                    llm_provider=model,
                    name=f"{self.name}_value",
                    description=f"Value component for {self.name}",
                )
                cache["value_critic"] = value_critic

            self._state_manager.update("cache", cache)

        # Generate feedback and value
        feedback = str(feedback_critic.run(task, response) if feedback_critic else "")
        value = float(value_critic.run(task, response) if value_critic else 0.5)

        # Return combined results
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
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        self._check_input(text)

        # Get components
        cache = self._state_manager.get("cache", {})
        feedback_critic = cache.get("feedback_critic")
        value_critic = cache.get("value_critic")

        # Create components if not initialized
        if not feedback_critic:
            # Import here to avoid circular import issues
            from ..implementations.factories import create_feedback_critic as create_fb_critic  # type: ignore

            model = self._state_manager.get("model")
            feedback_critic = create_fb_critic(
                llm_provider=model,
                name=f"{self.name}_feedback",
                description=f"Feedback component for {self.name}",
            )
            cache["feedback_critic"] = feedback_critic
            self._state_manager.update("cache", cache)

        # Improve text
        if not feedback:
            # Generate feedback if not provided
            task = "Improve the following text"
            feedback = str(feedback_critic.run(task, text) if feedback_critic else "")

        # Get improved text using feedback critic
        if feedback_critic:
            improved_text = feedback_critic.improve(text, feedback)
            # Make sure we always return a string
            return str(improved_text if improved_text is not None else "")
        else:
            # If no feedback_critic is available, return the original text
            return str(text)

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
        self._check_input(text)

        # Use a default task
        task = "Evaluate the text"

        # Generate feedback and value
        result = self.run(task, text)
        feedback = result["feedback"]
        value = result["value"]

        # Create critique result
        critique_result: CritiqueResult = {
            "score": value,
            "feedback": feedback,
            "issues": [],
            "suggestions": [],
        }

        return critique_result

    def record_error(self, error: Exception) -> None:
        """Record an error in the state manager.

        Args:
            error: The error to record
        """
        if hasattr(self, "config") and self.config.track_errors:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(error))
            self._state_manager.set_metadata("last_error_time", time.time())

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
    feedback_prompt: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> FeedbackCritic:
    """
    Create a feedback critic.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the result cache
        priority: Priority of the critic
        cost: Computational cost of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
        feedback_prompt: Prompt for feedback
        config: Optional configuration for the critic
        **kwargs: Additional configuration parameters

    Returns:
        FeedbackCritic: The created critic

    Raises:
        ValueError: If configuration is invalid
    """
    # Merge configuration from all sources
    config_data: Dict[str, Any] = {}

    # Start with defaults
    if not config:
        config_data = {
            "name": name,
            "description": description,
            "min_confidence": 0.7,
            "max_attempts": 3,
            "cache_size": 100,
            "priority": 1,
            "track_performance": True,
            "system_prompt": "You are a helpful critic that provides detailed feedback.",
            "temperature": 0.7,
            "max_tokens": 1000,
            "feedback_prompt": "Provide feedback on the following text.",
        }

    # Update with provided config
    if isinstance(config, dict):
        config_data.update(config)
    elif config:
        # If it's a FeedbackCriticConfig, extract its attributes
        config_data = config.model_dump()

    # Update with individual parameters if provided
    if min_confidence is not None:
        config_data["min_confidence"] = min_confidence
    if max_attempts is not None:
        config_data["max_attempts"] = max_attempts
    if cache_size is not None:
        config_data["cache_size"] = cache_size
    if priority is not None:
        config_data["priority"] = priority
    if cost is not None:
        config_data["cost"] = cost
    if system_prompt is not None:
        config_data["system_prompt"] = system_prompt
    if temperature is not None:
        config_data["temperature"] = temperature
    if max_tokens is not None:
        config_data["max_tokens"] = max_tokens
    if feedback_prompt is not None:
        config_data["feedback_prompt"] = feedback_prompt

    # Create the critic with proper configuration
    critic = FeedbackCritic(config=config_data)
    critic._state_manager.update("model", llm_provider)
    critic._state_manager.set_metadata("model_type", type(llm_provider).__name__)

    return critic


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
    value_prompt: Optional[str] = None,
    min_score: float = 0.0,
    max_score: float = 10.0,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> ValueCritic:
    """
    Create a value critic.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the result cache
        priority: Priority of the critic
        cost: Computational cost of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
        value_prompt: Prompt for value scoring
        min_score: Minimum score value
        max_score: Maximum score value
        config: Optional configuration for the critic
        **kwargs: Additional configuration parameters

    Returns:
        ValueCritic: The created critic

    Raises:
        ValueError: If configuration is invalid
    """
    # Merge configuration from all sources
    config_data: Dict[str, Any] = {}

    # Start with defaults
    if not config:
        config_data = {
            "min_confidence": 0.7 if min_confidence is None else min_confidence,
            "max_attempts": 3 if max_attempts is None else max_attempts,
            "cache_size": 100 if cache_size is None else cache_size,
            "priority": 1 if priority is None else priority,
            "cost": cost,
            "track_performance": True,
            "system_prompt": system_prompt
            or "You are a helpful critic that evaluates text quality.",
            "temperature": 0.7 if temperature is None else temperature,
            "max_tokens": 1000 if max_tokens is None else max_tokens,
            "value_prompt": value_prompt
            or "Evaluate the quality of the following text on a scale from 0 to 10.",
        }

    # Update with provided config
    if isinstance(config, dict):
        config_data.update(config)
    elif config:
        # If it's a ValueCriticConfig, extract its attributes
        config_data = config.model_dump()

    # Add min_score and max_score to kwargs
    kwargs["min_score"] = min_score
    kwargs["max_score"] = max_score

    # Create the critic
    return ValueCritic(
        name=name, description=description, llm_provider=llm_provider, config=config_data, **kwargs
    )


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
    feedback_prompt: Optional[str] = None,
    value_prompt: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> LACCritic:
    """
    Create a LAC critic.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the result cache
        priority: Priority of the critic
        cost: Computational cost of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
        feedback_prompt: Prompt for feedback
        value_prompt: Prompt for value scoring
        config: Optional configuration for the critic
        **kwargs: Additional configuration parameters

    Returns:
        LACCritic: The created critic

    Raises:
        ValueError: If configuration is invalid
    """
    # Merge configuration from all sources
    config_data: Dict[str, Any] = {}

    # Start with defaults
    if not config:
        config_data = {
            "min_confidence": 0.7 if min_confidence is None else min_confidence,
            "max_attempts": 3 if max_attempts is None else max_attempts,
            "cache_size": 100 if cache_size is None else cache_size,
            "priority": 1 if priority is None else priority,
            "cost": cost,
            "track_performance": True,
            "system_prompt": system_prompt
            or "You are a helpful critic that evaluates text quality.",
            "temperature": 0.7 if temperature is None else temperature,
            "max_tokens": 1000 if max_tokens is None else max_tokens,
        }

    # Update with provided config
    if isinstance(config, dict):
        config_data.update(config)
    elif config:
        # If it's a LACCriticConfig, extract its attributes
        config_data = config.model_dump()

    # Create the feedback and value critic configurations
    feedback_config = {
        "system_prompt": system_prompt
        or "You are a helpful critic that provides detailed feedback.",
        "temperature": 0.7 if temperature is None else temperature,
        "max_tokens": 1000 if max_tokens is None else max_tokens,
        "feedback_prompt": feedback_prompt or "Provide feedback on the following text.",
    }

    value_config = {
        "system_prompt": system_prompt or "You are a helpful critic that evaluates text quality.",
        "temperature": 0.7 if temperature is None else temperature,
        "max_tokens": 1000 if max_tokens is None else max_tokens,
        "value_prompt": value_prompt
        or "Evaluate the quality of the following text on a scale from 0 to 10.",
    }

    # Create the critic
    critic = LACCritic(
        name=name, description=description, llm_provider=llm_provider, config=config_data
    )

    # Initialize the critic's components
    critic._state_manager.update("model", llm_provider)

    return critic
