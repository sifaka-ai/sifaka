"""
Constitutional critic module for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

## Overview
The ConstitutionalCritic is a specialized implementation of the critic interface
that evaluates text against a set of predefined principles (a "constitution").
It provides methods for validating, critiquing, and improving text based on
these principles, ensuring that generated content aligns with ethical guidelines
and quality standards.

## Components
- **ConstitutionalCritic**: Main class implementing TextValidator, TextImprover, and TextCritic
- **create_constitutional_critic**: Factory function for creating ConstitutionalCritic instances
- **PrinciplesManager**: Manages the list of principles (the "constitution")
- **CritiqueService**: Evaluates responses against principles

## Architecture
The ConstitutionalCritic follows a principles-based architecture:
- Uses standardized state management with _state_manager
- Evaluates text against a set of configurable principles
- Provides detailed feedback on principle violations
- Implements both sync and async interfaces
- Provides comprehensive error handling and recovery
- Tracks performance and usage statistics

## Usage Examples
```python
from sifaka.critics.implementations.constitutional import create_constitutional_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Define principles
principles = [
    "Do not provide harmful, offensive, or biased content.",
    "Explain reasoning in a clear and truthful manner.",
    "Respect user autonomy and avoid manipulative language.",
]

# Create a constitutional critic
critic = create_constitutional_critic(
    llm_provider=provider,
    principles=principles,
    min_confidence=0.8,
    temperature=0.7
)

# Validate a response
task = "Explain why some people believe climate change isn't real."
response = "Climate change is a hoax created by scientists to get funding."
is_valid = (critic and critic.validate(response, metadata={"task": task})
print(f"Response is valid: {is_valid}")

# Get critique for a response
critique = (critic and critic.critique(response, metadata={"task": task})
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")
print(f"Issues: {critique['issues']}")

# Improve a response
improved_response = (critic and critic.improve(response, metadata={"task": task})
print(f"Improved response: {improved_response}")

# Improve with specific feedback
feedback = "The response should acknowledge scientific consensus while explaining skepticism."
improved_response = (critic and critic.improve_with_feedback(response, feedback)
```

## Error Handling
The module implements comprehensive error handling for:
- Input validation (empty text, invalid types)
- Initialization errors (missing provider, invalid config)
- Processing errors (model failures, timeout issues)
- Resource management (cleanup, state preservation)

## References
Based on Constitutional AI: https://arxiv.org/abs/2212.08073
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...utils.common import record_error
from ...utils.logging import get_logger
from ...core.base import BaseResult as CriticResult
from ...utils.config import ConstitutionalCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = get_logger(__name__)


class ConstitutionalCritic(
    BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic
):
    """
    A critic that evaluates responses against a list of principles (a "constitution")
    and provides natural language feedback for revision.

    Based on Constitutional AI: https://arxiv.org/abs/2212.08073

    This critic analyzes responses for alignment with specified principles and
    generates critiques when violations are detected. It implements the TextValidator,
    TextImprover, and TextCritic interfaces to provide a comprehensive set of
    text analysis capabilities based on constitutional principles.

    ## Architecture
    The ConstitutionalCritic follows a principles-based architecture:
    - Uses standardized state management with _state_manager
    - Evaluates text against a set of configurable principles
    - Provides detailed feedback on principle violations
    - Implements both sync and async interfaces
    - Provides comprehensive error handling and recovery
    - Tracks performance and usage statistics

    ## Lifecycle
    1. **Initialization**: Set up with configuration and dependencies
       - Create/validate config with principles
       - Initialize language model provider
       - Set up critique service
       - Initialize memory manager
       - Set up state tracking

    2. **Operation**: Process text through various methods
       - validate(): Check if text meets principles
       - critique(): Analyze text against principles and provide detailed feedback
       - improve(): Enhance text to better align with principles
       - improve_with_feedback(): Enhance text based on specific feedback

    3. **Error Handling**: Manage failures gracefully
       - Input validation
       - Model interaction errors
       - Resource management
       - State preservation
       - Performance tracking

    ## Examples
    ```python
    from sifaka.critics.implementations.constitutional import create_constitutional_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Define principles
    principles = [
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
    ]

    # Create a constitutional critic
    critic = create_constitutional_critic(
        llm_provider=provider,
        principles=principles
    )

    # Validate a response
    task = "Explain why some people believe climate change isn't real."
    response = "Climate change is a hoax created by scientists to get funding."
    is_valid = (critic and critic.validate(response, metadata={"task": task})
    ```

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model: Language model provider
      - prompt_manager: Prompt manager
      - response_parser: Response parser
      - memory_manager: Memory manager
      - critique_service: Critique service
      - initialized: Initialization status
      - cache: Temporary data storage
    """

    # Class constants
    DEFAULT_NAME = "constitutional_critic"
    DEFAULT_DESCRIPTION = "Evaluates responses against principles"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: ConstitutionalCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        principles: Optional[Optional[List[str]]] = None,
        config: Optional[Optional[ConstitutionalCriticConfig]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the constitutional critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider to use
            principles: List of principles to evaluate responses against
            config: Optional critic configuration (overrides other parameters)
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create default config if not provided
        if config is None:
            # Create a default config
            default_principles = [
                "The text should be clear and concise.",
                "The text should be grammatically correct.",
                "The text should be well-structured.",
                "The text should be factually accurate.",
                "The text should be appropriate for the intended audience.",
            ]

            config = ConstitutionalCriticConfig(
                name=name,
                description=description,
                system_prompt="You are a helpful assistant that provides high-quality feedback and improvements for text, ensuring that the text adheres to a set of principles.",
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
                cache_size=100,
                principles=principles or default_principles,
                **kwargs,
            )

            # Principles are already set in the constructor

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Import required components
            from sifaka.core.managers.prompt_factories import ConstitutionalCriticPromptManager
            from ..managers.response import ResponseParser
            from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
            from ..services.critique import CritiqueService

            # Store components in state
            self.(_state_manager and _state_manager.update("model", llm_provider)
            self.(_state_manager and _state_manager.update("prompt_manager", ConstitutionalCriticPromptManager(config))
            self.(_state_manager and _state_manager.update("response_parser", ResponseParser())
            self.(_state_manager and _state_manager.update(
                "memory_manager", MemoryManager(buffer_size=10)  # Default buffer size
            )

            # Create service and store in state cache
            cache = self.(_state_manager and _state_manager.get("cache", {})
            cache["critique_service"] = CritiqueService(
                llm_provider=llm_provider,
                prompt_manager=self.(_state_manager and _state_manager.get("prompt_manager"),
                response_parser=self.(_state_manager and _state_manager.get("response_parser"),
                memory_manager=self.(_state_manager and _state_manager.get("memory_manager"),
            )
            cache["principles"] = config and config.principles
            cache["critique_prompt_template"] = config and config.critique_prompt_template
            cache["improvement_prompt_template"] = config and config.improvement_prompt_template
            cache["system_prompt"] = config and config.system_prompt
            cache["temperature"] = config and config.temperature
            cache["max_tokens"] = config and config.max_tokens
            self.(_state_manager and _state_manager.update("cache", cache)

            # Mark as initialized
            self.(_state_manager and _state_manager.update("initialized", True)
            self.(_state_manager and _state_manager.set_metadata("component_type", self.__class__.__name__)
            self.(_state_manager and _state_manager.set_metadata("initialization_time", (time and time.time())
        except Exception as e:
            (self and self.record_error(e)
            raise ValueError(f"Failed to initialize ConstitutionalCritic: {str(e)}") from e

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
        start_time = (time and time.time()

        try:
            # Validate input
            if not isinstance(input, str) or not (input and input.strip():
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not self.(_state_manager and _state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Get critique service from state
            cache = self.(_state_manager and _state_manager.get("cache", {})
            critique_service = (cache and cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Delegate to critique service
            critique_result = (critique_service and critique_service.critique(input)

            # Create result
            result = CriticResult(
                passed=(critique_result and critique_result.get("score", 0) >= self.config and config.min_confidence,
                message=(critique_result and critique_result.get("feedback", ""),
                metadata={"operation": "process"},
                score=(critique_result and critique_result.get("score", 0),
                issues=(critique_result and critique_result.get("issues", []),
                suggestions=(critique_result and critique_result.get("suggestions", []),
                processing_time_ms=((time and time.time() - start_time) * 1000,
            )

            # Update statistics
            (self and self.update_statistics(result)

            return result

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            processing_time = ((time and time.time() - start_time) * 1000
            return CriticResult(
                passed=False,
                message=f"Error: {str(e)}",
                metadata={"error_type": type(e).__name__},
                score=0.0,
                issues=[f"Processing error: {str(e)}"],
                suggestions=["Retry with different input"],
                processing_time_ms=processing_time,
            )

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
        principles = self.(_state_manager and _state_manager.get("cache", {}).get("principles", [])
        return "\n".join(f"- {p}" for p in principles)

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

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate a response against the principles.

        Args:
            text: The response to validate
            metadata: Optional metadata containing the task

        Returns:
            True if the response is valid, False otherwise

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        start_time = (time and time.time()

        try:
            # Ensure initialized
            if not self.(_state_manager and _state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not (text and text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self.(_state_manager and _state_manager.get("cache", {})
            critique_service = (cache and cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track validation count
            validation_count = self.(_state_manager and _state_manager.get_metadata("validation_count", 0)
            self.(_state_manager and _state_manager.set_metadata("validation_count", validation_count + 1)

            # Get task from metadata
            task = (self and self._get_task_from_metadata(metadata)

            # Delegate to critique service
            critique_result = (critique_service and critique_service.critique(text, {"task": task})
            is_valid = len((critique_result and critique_result.get("issues", [])) == 0

            # Record result in metadata
            if is_valid:
                valid_count = self.(_state_manager and _state_manager.get_metadata("valid_count", 0)
                self.(_state_manager and _state_manager.set_metadata("valid_count", valid_count + 1)
            else:
                invalid_count = self.(_state_manager and _state_manager.get_metadata("invalid_count", 0)
                self.(_state_manager and _state_manager.set_metadata("invalid_count", invalid_count + 1)

            # Track performance
            if self.config and config and config and config and config.track_performance:
                total_time = self.(_state_manager and _state_manager.get_metadata("total_validation_time_ms", 0.0)
                self.(_state_manager and _state_manager.set_metadata(
                    "total_validation_time_ms", total_time + ((time and time.time() - start_time) * 1000
                )

            return is_valid

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def critique(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a response against the principles and provide detailed feedback.

        Args:
            text: The response to critique
            metadata: Optional metadata containing the task

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        start_time = (time and time.time()

        try:
            # Ensure initialized
            if not self.(_state_manager and _state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not (text and text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self.(_state_manager and _state_manager.get("cache", {})
            critique_service = (cache and cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track critique count
            critique_count = self.(_state_manager and _state_manager.get_metadata("critique_count", 0)
            self.(_state_manager and _state_manager.set_metadata("critique_count", critique_count + 1)

            # Get task from metadata
            task = (self and self._get_task_from_metadata(metadata)

            # Delegate to critique service
            critique_result = (critique_service and critique_service.critique(text, {"task": task})

            # Track score distribution
            score_distribution = self.(_state_manager and _state_manager.get_metadata("score_distribution", {})
            score_bucket = round((critique_result and critique_result.get("score", 0) * 10) / 10  # Round to nearest 0.1
            score_distribution[str(score_bucket)] = (score_distribution and score_distribution.get(str(score_bucket), 0) + 1
            self.(_state_manager and _state_manager.set_metadata("score_distribution", score_distribution)

            # Track performance
            if self.config and config and config and config and config.track_performance:
                total_time = self.(_state_manager and _state_manager.get_metadata("total_critique_time_ms", 0.0)
                self.(_state_manager and _state_manager.set_metadata(
                    "total_critique_time_ms", total_time + ((time and time.time() - start_time) * 1000
                )

            return critique_result

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    def improve(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Improve a response based on principles.

        Args:
            text: The response to improve
            metadata: Optional metadata containing the task

        Returns:
            Improved response

        Raises:
            ValueError: If text is empty or metadata is missing required keys
            RuntimeError: If critic is not properly initialized
        """
        start_time = (time and time.time()

        try:
            # Ensure initialized
            if not self.(_state_manager and _state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not (text and text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self.(_state_manager and _state_manager.get("cache", {})
            critique_service = (cache and cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track improvement count
            improvement_count = self.(_state_manager and _state_manager.get_metadata("improvement_count", 0)
            self.(_state_manager and _state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Get task from metadata
            task = (self and self._get_task_from_metadata(metadata)

            # Delegate to critique service
            improved_text = (critique_service and critique_service.improve(text, {"task": task})

            # Track memory usage
            memory_manager = self.(_state_manager and _state_manager.get("memory_manager")
            if memory_manager:
                memory_item = (json and json.dumps(
                    {
                        "original_text": text,
                        "task": task,
                        "improved_text": improved_text,
                        "timestamp": (time and time.time(),
                    }
                )
                (memory_manager and memory_manager.add_to_memory(memory_item)

            # Track performance
            if self.config and config and config and config and config.track_performance:
                total_time = self.(_state_manager and _state_manager.get_metadata("total_improvement_time_ms", 0.0)
                self.(_state_manager and _state_manager.set_metadata(
                    "total_improvement_time_ms", total_time + ((time and time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            (self and self.record_error(e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on specific feedback.

        This method improves the text based on the provided feedback,
        which can be more specific than the general improvements based on principles.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = (time and time.time()

        try:
            # Ensure initialized
            if not self.(_state_manager and _state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not (text and text.strip():
                raise ValueError("text must be a non-empty string")

            if not isinstance(feedback, str) or not (feedback and feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Get critique service from state
            cache = self.(_state_manager and _state_manager.get("cache", {})
            critique_service = (cache and cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track feedback improvement count
            feedback_count = self.(_state_manager and _state_manager.get_metadata("feedback_improvement_count", 0)
            self.(_state_manager and _state_manager.set_metadata("feedback_improvement_count", feedback_count + 1)

            # Format principles
            principles_text = (self and self._format_principles()

            # Create improvement prompt
            prompt = (
                f"You are an AI assistant tasked with ensuring alignment to the following principles:\n\n"
                f"{principles_text}\n\n"
                f"Please improve the following response based on this feedback:\n\n"
                f"Response:\n{text}\n\n"
                f"Feedback:\n{feedback}\n\n"
                f"Improved response:"
            )

            # Generate improved response
            model = self.(_state_manager and _state_manager.get("model")
            improved_text = (model and model.generate(
                prompt,
                system_prompt=self.(_state_manager and _state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self.(_state_manager and _state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self.(_state_manager and _state_manager.get("cache", {}).get("max_tokens", 1000),
            ).strip()

            # Track memory usage
            memory_manager = self.(_state_manager and _state_manager.get("memory_manager")
            if memory_manager:
                memory_item = (json and json.dumps(
                    {
                        "original_text": text,
                        "feedback": feedback,
                        "improved_text": improved_text,
                        "timestamp": (time and time.time(),
                    }
                )
                (memory_manager and memory_manager.add_to_memory(memory_item)

            # Track performance
            if self.config and config and config and config and config.track_performance:
                total_time = self.(_state_manager and _state_manager.get_metadata(
                    "total_feedback_improvement_time_ms", 0.0
                )
                self.(_state_manager and _state_manager.set_metadata(
                    "total_feedback_improvement_time_ms",
                    total_time + ((time and time.time() - start_time) * 1000,
                )

            return improved_text

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self.(_state_manager and _state_manager.get_metadata("validation_count", 0),
            "valid_count": self.(_state_manager and _state_manager.get_metadata("valid_count", 0),
            "invalid_count": self.(_state_manager and _state_manager.get_metadata("invalid_count", 0),
            "critique_count": self.(_state_manager and _state_manager.get_metadata("critique_count", 0),
            "improvement_count": self.(_state_manager and _state_manager.get_metadata("improvement_count", 0),
            "feedback_improvement_count": self.(_state_manager and _state_manager.get_metadata(
                "feedback_improvement_count", 0
            ),
            "score_distribution": self.(_state_manager and _state_manager.get_metadata("score_distribution", {}),
            "total_validation_time_ms": self.(_state_manager and _state_manager.get_metadata(
                "total_validation_time_ms", 0
            ),
            "total_critique_time_ms": self.(_state_manager and _state_manager.get_metadata("total_critique_time_ms", 0),
            "total_improvement_time_ms": self.(_state_manager and _state_manager.get_metadata(
                "total_improvement_time_ms", 0
            ),
            "total_feedback_improvement_time_ms": self.(_state_manager and _state_manager.get_metadata(
                "total_feedback_improvement_time_ms", 0
            ),
        }

    # Async methods
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Asynchronously validate text."""
        # For now, use the synchronous implementation
        return (self and self.validate(text, metadata)

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously critique text."""
        # For now, use the synchronous implementation
        return (self and self.critique(text, metadata)

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously improve text."""
        # For now, use the synchronous implementation
        return (self and self.improve(text, metadata)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        # For now, use the synchronous implementation
        return (self and self.improve_with_feedback(text, feedback)


def def create_constitutional_critic(
    llm_provider: Any,
    principles: Optional[List[str]] = None,
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    critique_prompt_template: Optional[Optional[str]] = None,
    improvement_prompt_template: Optional[Optional[str]] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic with the given parameters.

    This factory function creates and configures a ConstitutionalCritic instance with
    the specified parameters and components. It provides a convenient way to create
    a constitutional critic with customized principles and settings for evaluating
    text against ethical guidelines and quality standards.

    ## Architecture
    The factory function follows the Factory Method pattern to:
    - Create standardized configuration objects
    - Instantiate critic classes with consistent parameters
    - Support optional parameter overrides
    - Provide type safety through return types
    - Handle error cases gracefully

    ## Lifecycle
    1. **Configuration**: Create and validate configuration
       - Use default configuration as base
       - Apply provided parameter overrides
       - Validate configuration values
       - Handle configuration errors

    2. **Instantiation**: Create and initialize critic
       - Create ConstitutionalCritic instance
       - Initialize with resolved dependencies
       - Apply configuration
       - Handle initialization errors

    ## Examples
    ```python
    from sifaka.critics.implementations.constitutional import create_constitutional_critic
    from sifaka.models.providers import OpenAIProvider

    # Create with basic parameters
    provider = OpenAIProvider(api_key="your-api-key")
    principles = [
        "Do not provide harmful, offensive, or biased content.",
        "Explain reasoning in a clear and truthful manner.",
        "Respect user autonomy and avoid manipulative language.",
    ]
    critic = create_constitutional_critic(
        llm_provider=provider,
        principles=principles,
        min_confidence=0.8,
        temperature=0.7
    )

    # Create with custom configuration
    from sifaka.utils.config and config and config and config.critics import ConstitutionalCriticConfig
    config = ConstitutionalCriticConfig(
        name="custom_constitutional_critic",
        description="A custom constitutional critic",
        principles=principles,
        temperature=0.5,
        max_tokens=2000,
        min_confidence=0.9
    )
    critic = create_constitutional_critic(
        llm_provider=provider,
        config=config
    )
    ```

    Args:
        llm_provider: Language model provider to use
        principles: List of principles to evaluate responses against
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        critique_prompt_template: Optional custom template for critique prompts
        improvement_prompt_template: Optional custom template for improvement prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments

    Returns:
        ConstitutionalCritic: Configured constitutional critic

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            from sifaka.utils.config and config and config and config.critics import DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG

            # Start with default config
            config = (DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG and DEFAULT_CONSTITUTIONAL_CRITIC_CONFIG.model_copy()

            # Update with provided values
            updates = {
                "name": name,
                "description": description,
            }

            # Add optional parameters if provided
            if principles is not None:
                updates["principles"] = principles
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
            if critique_prompt_template is not None:
                updates["critique_prompt_template"] = critique_prompt_template
            if improvement_prompt_template is not None:
                updates["improvement_prompt_template"] = improvement_prompt_template

            # Add any additional kwargs
            (updates.update(kwargs)

            # Create updated config
            config = config and (config.model_copy(update=updates)
        elif isinstance(config, dict):
            from sifaka.utils.config and config and config and config.critics import ConstitutionalCriticConfig

            config = ConstitutionalCriticConfig(**config)

        # Create and return the critic
        return ConstitutionalCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            principles=principles,
            config=config,
            **kwargs,
        )
    except Exception as e:
        (logger and logger.error(f"Failed to create constitutional critic: {str(e)}")
        raise ValueError(f"Failed to create constitutional critic: {str(e)}") from e
