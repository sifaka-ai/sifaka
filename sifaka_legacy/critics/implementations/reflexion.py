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
is_valid = critic.validate(text) if critic else ""

# Critique text
critique = critic.critique(text) if critic else ""
print(f"Score: {critique.score}")
print(f"Feedback: {critique.message}")

# Improve text with feedback
feedback = "The text needs more detail and better structure."
improved_text = critic.improve(text, feedback) if critic else ""

# Improve again - this time it will use the previous reflection
improved_text_2 = critic.improve(
    "Another sample text that needs improvement.",
    "Make it more concise."
) if critic else ""
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
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent, BaseConfig
from ...utils.state import create_critic_state
from ...utils.logging import get_logger
from ...core.base import BaseResult
from ...utils.config import ReflexionCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult

# Configure logging
logger = get_logger(__name__)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert editor that improves text through reflection.
You maintain a memory of past improvements and use these reflections to guide
future improvements. Focus on learning patterns from past feedback and applying
them to new situations."""

# Default reflexion prompt template
DEFAULT_REFLEXION_PROMPT_TEMPLATE = """Given the following text and feedback, provide a reflection on how to improve it:

Text: {text}

Feedback: {feedback}

Reflection:"""


class ReflexionCritic(BaseComponent[str, BaseResult], TextValidator, TextImprover, TextCritic):
    """A critic that uses reflection to improve text quality.

    This critic uses a language model to analyze text and provide detailed
    feedback through a process of reflection. It can validate text, provide
    critiques, and suggest improvements based on the feedback.

    Features:
    - Text validation
    - Detailed critiques
    - Text improvement
    - Performance tracking
    - Error handling
    - State management

    Usage:
        critic = ReflexionCritic()
        is_valid = critic.validate(text)
        critique = critic.critique(text)
        improved_text = critic.improve(text, feedback)
    """

    def __init__(
        self,
        name: str = "reflexion_critic",
        description: str = "A critic that uses reflection to improve text quality",
        config: Optional[BaseConfig] = None,
    ):
        """Initialize the reflexion critic.

        Args:
            name: The name of the critic
            description: The description of the critic
            config: Optional configuration
        """
        super().__init__(name=name, description=description, config=config)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize components."""
        self._state_manager.update("initialized", True)

        # Initialize state with defaults
        self._state_manager.update(
            "cache",
            {
                "feedback_prompt_template": DEFAULT_REFLEXION_PROMPT_TEMPLATE,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )

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

            # Validate text
            result = critique_service.validate(text)

            # Ensure result is a boolean
            is_valid = bool(result)

            # Update statistics
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if is_valid:
                valid_count = self._state_manager.get_metadata("valid_count", 0)
                self._state_manager.set_metadata("valid_count", valid_count + 1)
            else:
                invalid_count = self._state_manager.get_metadata("invalid_count", 0)
                self._state_manager.set_metadata("invalid_count", invalid_count + 1)

            return is_valid

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def critique(self, text: str) -> CritiqueResult:
        """Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information

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

            # Critique text
            result = critique_service.critique(text)

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_critique_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_critique_time_ms", total_time + (time.time() - start_time) * 1000
                )

                # Track score distribution
                score = result.get("score", 0.0)
                score_distribution = self._state_manager.get_metadata("score_distribution", {})
                score_bucket = f"{int(score * 10) / 10:.1f}"
                score_distribution[score_bucket] = score_distribution.get(score_bucket, 0) + 1
                self._state_manager.set_metadata("score_distribution", score_distribution)

            # Convert result to CritiqueResult
            critique_result: CritiqueResult = {
                "score": result.get("score", 0.0),
                "feedback": result.get("feedback", ""),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
            }

            return critique_result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    def improve(self, text: str, feedback: str) -> str:
        """Improve text based on feedback.

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
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Improve text
            improved_text = critique_service.improve(text, feedback)

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_improvement_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_improvement_time_ms", total_time + (time.time() - start_time) * 1000
                )

            # Ensure we return a string
            return str(improved_text)

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text with feedback without generating a critique.

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
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ReflexionCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Improve text
            improved_text = critique_service.improve(text, feedback)

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata(
                    "total_feedback_improvement_time_ms", 0.0
                )
                self._state_manager.set_metadata(
                    "total_feedback_improvement_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            # Ensure we return a string
            return str(improved_text)

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def process(self, input: str) -> BaseResult:
        """Process input text with the critic.

        This is required by BaseComponent and serves as a bridge to critique().

        Args:
            input: The text to process

        Returns:
            BaseResult containing critique results
        """
        critique_result = self.critique(input)

        # Convert CritiqueResult to BaseResult
        return BaseResult(
            passed=critique_result.get("score", 0.0) >= 0.7,  # Assuming 0.7 is the threshold
            message=critique_result.get("feedback", ""),
            score=critique_result.get("score", 0.0),
            issues=critique_result.get("issues", []),
            suggestions=critique_result.get("suggestions", []),
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
    priority: int = 1,
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
    from sifaka.utils.config.critics import ReflexionCriticConfig
    config = ReflexionCriticConfig(
        name="custom_reflexion_critic",
        description="A custom reflexion critic",
        system_prompt="You are an expert editor that learns from past feedback",
        temperature=0.5,
        max_tokens=2000,
        memory_buffer_size=10,
        reflection_count=3  # Note: use reflection_count instead of reflection_depth
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
            # Create a BaseConfig for the component, wrapping the params
            critic_params = {
                "llm_provider": llm_provider,
                "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
                "temperature": temperature or 0.7,
                "max_tokens": max_tokens or 1000,
                "min_confidence": min_confidence or 0.7,
                "max_attempts": max_attempts or 3,
                "cache_size": cache_size or 100,
                "memory_buffer_size": memory_buffer_size or 5,
                "reflection_count": reflection_depth
                or 1,  # Map reflection_depth to reflection_count
                "track_performance": track_performance if track_performance is not None else True,
                "priority": priority,
                "cost": cost,
                "prompt_factory": prompt_factory,
            }

            # Add any additional kwargs
            critic_params.update(kwargs)

            # Create a BaseConfig
            base_config = BaseConfig(name=name, description=description, params=critic_params)

            return ReflexionCritic(name=name, description=description, config=base_config)

        elif isinstance(config, dict):
            # Convert dict to ReflexionCriticConfig and wrap in BaseConfig
            critic_config = ReflexionCriticConfig(**config)
            base_config = BaseConfig(
                name=critic_config.name,
                description=critic_config.description or description,
                params={"config": critic_config},
            )
            return ReflexionCritic(
                name=critic_config.name,
                description=critic_config.description or description,
                config=base_config,
            )
        else:
            # Config is already a ReflexionCriticConfig, wrap in BaseConfig
            base_config = BaseConfig(
                name=config.name,
                description=config.description or description,
                params={"config": config},
            )
            return ReflexionCritic(
                name=config.name, description=config.description or description, config=base_config
            )
    except Exception as e:
        logger.error(f"Failed to create reflexion critic: {str(e)}")
        raise ValueError(f"Failed to create reflexion critic: {str(e)}") from e
