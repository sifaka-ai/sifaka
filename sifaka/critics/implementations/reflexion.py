"""
Reflexion critic module for Sifaka.

This module implements the Reflexion approach for critics, which enables language model
agents to learn from feedback without requiring weight updates. It maintains reflections
in memory to improve future text generation.

## Overview
The ReflexionCritic is a specialized implementation of the critic interface
that uses memory-augmented generation to improve text quality over time. It
maintains a buffer of past reflections and uses them to guide future text
improvements, enabling a form of learning without model weight updates.

## Components
- **ReflexionCritic**: Main class implementing TextValidator, TextImprover, and TextCritic
- **create_reflexion_critic**: Factory function for creating ReflexionCritic instances
- **ReflexionCriticPromptManager**: Creates prompts with reflection context
- **MemoryManager**: Manages the memory buffer of past reflections

## Architecture
The ReflexionCritic follows a memory-augmented architecture:
- Uses standardized state management with _state_manager
- Maintains a buffer of past reflections in memory
- Incorporates reflections into prompts for improved generation
- Implements both sync and async interfaces
- Provides comprehensive error handling and recovery
- Tracks performance and usage statistics

## Usage Examples
```python
from sifaka.critics.implementations.reflexion import create_reflexion_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a reflexion critic
critic = create_reflexion_critic(
    llm_provider=provider,
    system_prompt="You are an expert editor that learns from past feedback",
    memory_buffer_size=5,
    reflection_depth=2
)

# Validate text
text = "The quick brown fox jumps over the lazy dog."
is_valid = critic.validate(text)

# Critique text
critique = critic.critique(text)
print(f"Score: {critique.score}")
print(f"Feedback: {critique.message}")

# Improve text with feedback
feedback = "The text needs more detail and better structure."
improved_text = critic.improve(text, feedback)

# Improve again - this time it will use the previous reflection
improved_text_2 = critic.improve(
    "Another sample text that needs improvement.",
    "Make it more concise."
)
```

## Error Handling
The module implements comprehensive error handling for:
- Input validation (empty text, invalid types)
- Initialization errors (missing provider, invalid config)
- Processing errors (model failures, timeout issues)
- Resource management (cleanup, state preservation)

## Component Lifecycle
1. **Initialization Phase**
   - Configuration validation
   - Provider setup
   - Memory buffer initialization
   - Resource allocation

2. **Operation Phase**
   - Text validation
   - Critique generation
   - Text improvement
   - Reflection generation
   - Memory management

3. **Cleanup Phase**
   - Resource cleanup
   - State reset
   - Error recovery
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...utils.logging import get_logger
from ...core.base import BaseResult as CriticResult
from ...utils.config import ReflexionCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = get_logger(__name__)


class ReflexionCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """A critic that uses reflection to improve text quality.

    This critic implements the Reflexion approach, which enables learning from
    feedback without requiring weight updates. It maintains a memory buffer of
    past reflections to improve future text generation. It implements
    the TextValidator, TextImprover, and TextCritic interfaces to provide a comprehensive
    set of text analysis capabilities with memory-augmented generation.

    ## Architecture
    The ReflexionCritic follows a component-based architecture with memory-augmented generation:
    - Uses standardized state management with _state_manager
    - Maintains a buffer of past reflections in memory
    - Incorporates reflections into prompts for improved generation
    - Implements both sync and async interfaces
    - Provides comprehensive error handling and recovery
    - Tracks performance and usage statistics

    ## Lifecycle
    1. **Initialization**: Set up with configuration and dependencies
       - Create/validate config
       - Initialize language model provider
       - Set up critique service
       - Initialize memory manager with buffer size
       - Set up state tracking

    2. **Operation**: Process text through various methods
       - validate(): Check if text meets quality standards
       - critique(): Analyze text and provide detailed feedback
       - improve(): Enhance text based on feedback and reflections
       - Memory management: Store and retrieve reflections

    3. **Error Handling**: Manage failures gracefully
       - Input validation
       - Model interaction errors
       - Resource management
       - State preservation
       - Performance tracking

    ## Examples
    ```python
    from sifaka.critics.implementations.reflexion import create_reflexion_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a reflexion critic
    critic = create_reflexion_critic(
        llm_provider=provider,
        system_prompt="You are an expert editor that learns from past feedback",
        memory_buffer_size=5,
        reflection_depth=2
    )

    # Validate text
    text = "The quick brown fox jumps over the lazy dog."
    is_valid = critic.validate(text)

    # Critique text
    critique = critic.critique(text)
    print(f"Score: {critique.score}")
    print(f"Feedback: {critique.message}")

    # Improve text with feedback
    feedback = "The text needs more detail and better structure."
    improved_text = critic.improve(text, feedback)
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
    DEFAULT_NAME = "reflexion_critic"
    DEFAULT_DESCRIPTION = "Improves text using reflections on past feedback"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: ReflexionCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[ReflexionCriticConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the reflexion critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: The language model provider to use
            prompt_factory: Optional prompt factory to use
            config: Optional configuration for the critic
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create default config if not provided
        if config is None:
            # Create a default config
            config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt="You are a helpful assistant that provides high-quality feedback and improvements for text, using reflections on past feedback to guide your improvements.",
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
                cache_size=100,
                memory_buffer_size=5,
                reflection_depth=1,
                **kwargs,
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Import required components
            from sifaka.core.managers.prompt import ReflexionCriticPromptManager
            from sifaka.core.managers.response import ResponseParser
            from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager
            from ..services.critique import CritiqueService

            # Store components in state
            self._state_manager.update("model", llm_provider)
            self._state_manager.update(
                "prompt_manager", prompt_factory or ReflexionCriticPromptManager(config)
            )
            self._state_manager.update("response_parser", ResponseParser())
            self._state_manager.update(
                "memory_manager", MemoryManager(buffer_size=config.memory_buffer_size)
            )

            # Create service and store in state cache
            cache = self._state_manager.get("cache", {})
            cache["critique_service"] = CritiqueService(
                llm_provider=llm_provider,
                prompt_manager=self._state_manager.get("prompt_manager"),
                response_parser=self._state_manager.get("response_parser"),
                memory_manager=self._state_manager.get("memory_manager"),
            )
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
            raise ValueError(f"Failed to initialize ReflexionCritic: {str(e)}") from e

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
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Delegate to critique service
            critique_result = critique_service.critique(input)

            # Create result
            result = CriticResult(
                passed=critique_result.get("score", 0) >= self.config.min_confidence,
                message=critique_result.get("feedback", ""),
                metadata={"operation": "process"},
                score=critique_result.get("score", 0),
                issues=critique_result.get("issues", []),
                suggestions=critique_result.get("suggestions", []),
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

    def _format_feedback(self, feedback: Any) -> str:
        """Format feedback into a string.

        Args:
            feedback: The feedback to format, which can be a string, list, or None

        Returns:
            str: The formatted feedback string
        """
        if isinstance(feedback, list):
            if not feedback:
                return "No issues found."

            result = "The following issues were found:\n"
            for i, violation in enumerate(feedback):
                rule_name = violation.get("rule_name", f"Rule {i+1}")
                message = violation.get("message", "Unknown issue")
                result += f"- {rule_name}: {message}\n"
            return result
        elif feedback is None:
            return "Please improve this text."
        return feedback

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

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
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track validation count
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Delegate to critique service
            is_valid = critique_service.validate(text)

            # Record result in metadata
            if is_valid:
                valid_count = self._state_manager.get_metadata("valid_count", 0)
                self._state_manager.set_metadata("valid_count", valid_count + 1)
            else:
                invalid_count = self._state_manager.get_metadata("invalid_count", 0)
                self._state_manager.set_metadata("invalid_count", invalid_count + 1)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_validation_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_validation_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return is_valid

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def improve(self, text: str, feedback: str = None) -> str:
        """Improve text based on feedback and reflections.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Format feedback
            feedback_str = self._format_feedback(feedback)

            # Track improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Delegate to critique service
            improved_text = critique_service.improve(text, feedback_str)

            # Track memory usage
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager:
                memory_item = json.dumps(
                    {
                        "original_text": text,
                        "feedback": feedback_str,
                        "improved_text": improved_text,
                        "timestamp": time.time(),
                    }
                )
                memory_manager.add_to_memory(memory_item)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_improvement_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_improvement_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or feedback is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Validate feedback
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Delegate to improve method
            improved_text = self.improve(text, feedback)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata(
                    "total_feedback_improvement_time_ms", 0.0
                )
                self._state_manager.set_metadata(
                    "total_feedback_improvement_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def critique(self, text: str) -> CriticResult:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            CriticResult: Object containing critique details

        Raises:
            ValueError: If text is empty
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track critique count
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Delegate to critique service
            critique_result = critique_service.critique(text)

            # Create result
            result = CriticResult(
                passed=critique_result.get("score", 0) >= self.config.min_confidence,
                message=critique_result.get("feedback", ""),
                metadata={"operation": "critique"},
                score=critique_result.get("score", 0),
                issues=critique_result.get("issues", []),
                suggestions=critique_result.get("suggestions", []),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Track score distribution
            score_distribution = self._state_manager.get_metadata("score_distribution", {})
            score_bucket = round(result.score * 10) / 10  # Round to nearest 0.1
            score_distribution[str(score_bucket)] = score_distribution.get(str(score_bucket), 0) + 1
            self._state_manager.set_metadata("score_distribution", score_distribution)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_critique_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_critique_time_ms", total_time + (time.time() - start_time) * 1000
                )

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

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self._state_manager.get_metadata("validation_count", 0),
            "valid_count": self._state_manager.get_metadata("valid_count", 0),
            "invalid_count": self._state_manager.get_metadata("invalid_count", 0),
            "critique_count": self._state_manager.get_metadata("critique_count", 0),
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "score_distribution": self._state_manager.get_metadata("score_distribution", {}),
            "total_validation_time_ms": self._state_manager.get_metadata(
                "total_validation_time_ms", 0
            ),
            "total_critique_time_ms": self._state_manager.get_metadata("total_critique_time_ms", 0),
            "total_improvement_time_ms": self._state_manager.get_metadata(
                "total_improvement_time_ms", 0
            ),
            "total_feedback_improvement_time_ms": self._state_manager.get_metadata(
                "total_feedback_improvement_time_ms", 0
            ),
        }

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text."""
        # For now, use the synchronous implementation
        return self.validate(text)

    async def acritique(self, text: str) -> CriticResult:
        """Asynchronously critique text."""
        # For now, use the synchronous implementation
        return self.critique(text)

    async def aimprove(self, text: str, feedback: str = None) -> str:
        """Asynchronously improve text."""
        # For now, use the synchronous implementation
        return self.improve(text, feedback)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        # For now, use the synchronous implementation
        return self.improve_with_feedback(text, feedback)


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert editor that improves text through reflection.
You maintain a memory of past improvements and use these reflections to guide
future improvements. Focus on learning patterns from past feedback and applying
them to new situations."""


def create_reflexion_critic(
    llm_provider: Any,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
    track_performance: bool = True,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    prompt_factory: Optional[Any] = None,
    config: Optional[Union[Dict[str, Any], ReflexionCriticConfig]] = None,
    **kwargs: Any,
) -> ReflexionCritic:
    """Create a reflexion critic with the given parameters.

    This factory function creates and configures a ReflexionCritic instance with
    the specified parameters and components. It provides a convenient way to create
    a reflexion critic with customized settings for memory-augmented text improvement.

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
       - Create ReflexionCritic instance
       - Initialize with resolved dependencies
       - Apply configuration
       - Handle initialization errors

    ## Examples
    ```python
    from sifaka.critics.implementations.reflexion import create_reflexion_critic
    from sifaka.models.providers import OpenAIProvider

    # Create with basic parameters
    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_reflexion_critic(
        llm_provider=provider,
        system_prompt="You are an expert editor that learns from past feedback",
        memory_buffer_size=5,
        reflection_depth=2
    )

    # Create with custom configuration
    from sifaka.utils.config import ReflexionCriticConfig
    config = ReflexionCriticConfig(
        name="custom_reflexion_critic",
        description="A custom reflexion critic",
        system_prompt="You are an expert editor that learns from past feedback",
        temperature=0.5,
        max_tokens=2000,
        memory_buffer_size=10,
        reflection_depth=3
    )
    critic = create_reflexion_critic(
        llm_provider=provider,
        config=config
    )
    ```

    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for text generation
        max_tokens: Maximum tokens for text generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        memory_buffer_size: Size of the memory buffer for reflections
        reflection_depth: Depth of reflections
        track_performance: Whether to track performance metrics
        cache_size: Size of the cache for memoization
        priority: Priority of the critic
        cost: Cost of using the critic
        prompt_factory: Optional prompt factory to use
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments

    Returns:
        ReflexionCritic: Configured reflexion critic

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If llm_provider is not a valid provider
    """
    try:
        # Create config if not provided
        if config is None:
            # Create a default config with provided values
            config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt=system_prompt
                or "You are a helpful assistant that provides high-quality feedback and improvements for text, using reflections on past feedback to guide your improvements.",
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 1000,
                min_confidence=min_confidence or 0.7,
                max_attempts=max_attempts or 3,
                cache_size=cache_size or 100,
                memory_buffer_size=memory_buffer_size or 5,
                reflection_depth=reflection_depth or 1,
                track_performance=track_performance if track_performance is not None else True,
                priority=priority,
                cost=cost,
                **kwargs,
            )
        elif isinstance(config, dict):
            # Convert dict to ReflexionCriticConfig
            config = ReflexionCriticConfig(**config)

        # Create and return the critic
        return ReflexionCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            prompt_factory=prompt_factory,
            config=config,
        )
    except Exception as e:
        logger.error(f"Failed to create reflexion critic: {str(e)}")
        raise ValueError(f"Failed to create reflexion critic: {str(e)}") from e
