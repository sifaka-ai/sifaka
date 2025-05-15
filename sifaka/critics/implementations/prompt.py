"""
Implementation of a prompt critic using a language model.

This module provides a critic that uses language models to evaluate,
validate, and improve text outputs based on rule violations.

## Overview
The PromptCritic is a comprehensive implementation of the critic interface
that uses language models to analyze and improve text. It provides methods
for text validation, critique, and improvement, with synchronous interfaces.

## Components
- **PromptCritic**: Main class implementing TextValidator, TextImprover, and TextCritic
- **create_prompt_critic**: Factory function for creating PromptCritic instances
- **CritiqueService**: Internal service for text analysis (imported from services)

## Architecture
The PromptCritic follows a component-based architecture:
- Uses standardized state management with _state_manager
- Delegates to specialized services for text processing
- Provides comprehensive error handling and recovery
- Tracks performance and usage statistics

## Usage Examples
```python
from sifaka.critics.implementations.prompt import create_prompt_critic
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a prompt critic
critic = create_prompt_critic(
    llm_provider=provider,
    system_prompt="You are an expert editor focusing on clarity and conciseness",
    temperature=0.7
)

# Validate text
text = "The quick brown fox jumps over the lazy dog."
is_valid = critic.validate(text) if critic else ""

# Critique text
critique = critic.critique(text) if critic else ""
print(f"Score: {critique['score']}")
print(f"Feedback: {critique['feedback']}")

# Improve text
improved_text = critic.improve(text, "Make the text more descriptive") if critic else ""

# Complete feedback loop
original = "Summarize the benefits of AI"
generator_response = "AI is good for many things."
improved, report = critic.close_feedback_loop(original, generator_response) if critic else ""
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
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, cast, Union, TypeVar, TypeAlias

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent, BaseConfig
from ...utils.state import StateManager, create_critic_state
from ...utils.common import record_error
from ...core.base import BaseResult
from ...utils.config import PromptCriticConfig
from ...interfaces.critic import TextCritic, TextImprover, TextValidator, CritiqueResult

# For improved type checking
T = TypeVar("T")

# Default system prompt used if none is provided
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides high-quality feedback and improvements for text."
)

# Default prompt templates
DEFAULT_PROMPT_TEMPLATES = {
    "validation": "Please validate the following text. Is it clear, concise, and effective?",
    "critique": "Please critique the following text. Provide detailed feedback on strengths and weaknesses.",
    "improvement": "Please improve the following text based on the provided feedback.",
}


class PromptCritic(BaseComponent[str, BaseResult[Any]], TextValidator, TextImprover, TextCritic):
    """A critic that uses a language model to evaluate and improve text.

    This critic uses a language model to analyze text quality and provide
    detailed feedback. It can validate text, provide critiques, and suggest
    improvements based on the feedback.

    Features:
    - Text validation
    - Detailed critiques
    - Text improvement
    - Performance tracking
    - Error handling
    - State management

    Usage:
        critic = PromptCritic()
        is_valid = critic.validate(text)
        critique = critic.critique(text)
        improved_text = critic.improve(text, feedback)
    """

    _state_manager: StateManager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str = "prompt_critic",
        description: str = "A critic that uses prompts to improve text",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the prompt critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            config: Optional configuration dictionary
        """
        # Convert dict config to BaseConfig for BaseComponent
        base_config = BaseConfig(name=name, description=description)
        if config:
            base_config.params = config

        super().__init__(name=name, description=description, config=base_config)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize components."""
        self._state_manager.update("initialized", True)

        # Initialize state with defaults
        self._state_manager.update(
            "cache",
            {
                "prompt_templates": DEFAULT_PROMPT_TEMPLATES,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )

    def process(self, input_text: str) -> BaseResult[Any]:
        """Process input text with the critic.

        This is the main entry point required by BaseComponent.

        Args:
            input_text: The text to process

        Returns:
            BaseResult: Result of the processing
        """
        critique_result = self.critique(input_text)

        # Convert CritiqueResult to regular dict for metadata
        critique_dict: Dict[str, Any] = {
            "score": critique_result.get("score", 0.0),
            "feedback": critique_result.get("feedback", ""),
            "issues": critique_result.get("issues", []),
            "suggestions": critique_result.get("suggestions", []),
        }

        return BaseResult(
            passed=critique_dict["score"] >= 0.5,
            message=critique_dict["feedback"],
            score=critique_dict["score"],
            metadata=critique_dict,
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
                raise RuntimeError("PromptCritic not properly initialized")

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
            if not isinstance(result, bool):
                result = bool(result)

            # Update statistics
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if result:
                success_count = self._state_manager.get_metadata("success_count", 0)
                self._state_manager.set_metadata("success_count", success_count + 1)
            else:
                failure_count = self._state_manager.get_metadata("failure_count", 0)
                self._state_manager.set_metadata("failure_count", failure_count + 1)

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return bool(result)

        except Exception as e:
            if hasattr(self, "record_error"):
                self.record_error(e)
            else:
                record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def critique(self, text: str) -> CritiqueResult:
        """Analyze text and provide detailed feedback.

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
                raise RuntimeError("PromptCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Get critique
            result = critique_service.critique(text)

            # Ensure result is a dictionary with the correct structure
            if not isinstance(result, dict):
                result = {"score": 0.0, "feedback": str(result), "issues": [], "suggestions": []}

            # Ensure the result has all required fields for CritiqueResult
            if "score" not in result:
                result["score"] = 0.0
            if "feedback" not in result:
                result["feedback"] = ""
            if "issues" not in result:
                result["issues"] = []
            if "suggestions" not in result:
                result["suggestions"] = []

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return cast(CritiqueResult, result)

        except Exception as e:
            if hasattr(self, "record_error"):
                self.record_error(e)
            else:
                record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    def improve(self, text: str, feedback: Optional[str] = None) -> str:
        """Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
            TypeError: If model returns non-string output
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("PromptCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Set default feedback if none provided
            if feedback is None:
                feedback = "Please improve this text for clarity and effectiveness."

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Improve text
            improved_text = critique_service.improve(text, feedback)

            # Ensure the result is a string
            if not isinstance(improved_text, str):
                improved_text = str(improved_text)

            # Track improvement in memory manager
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager:
                memory_item = json.dumps(
                    {
                        "original_text": text,
                        "feedback": feedback,
                        "improved_text": improved_text,
                        "timestamp": time.time(),
                    }
                )
                memory_manager.add_to_memory(memory_item)

            # Update statistics
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            self._state_manager.set_metadata("last_improvement_time", time.time())

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return str(improved_text)

        except Exception as e:
            if hasattr(self, "record_error"):
                self.record_error(e)
            else:
                record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def close_feedback_loop(
        self, text: str, generator_response: str, feedback: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Complete a feedback loop between a generator and this critic.

        Args:
            text: The original input text
            generator_response: The response produced by a generator
            feedback: Optional specific feedback (will generate critique if None)

        Returns:
            Tuple containing the improved text and a report dictionary with details

        Raises:
            ValueError: If text or generator_response is empty
            TypeError: If model returns non-string output
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("PromptCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if not isinstance(generator_response, str) or not generator_response.strip():
                raise ValueError("generator_response must be a non-empty string")

            # Generate feedback if not provided
            if feedback is None:
                critique = self.critique(generator_response)
                feedback = critique["feedback"]

            # Improve the response
            improved_text = self.improve(generator_response, feedback)

            # Create report
            report = {
                "original_input": text,
                "generator_response": generator_response,
                "critic_feedback": feedback,
                "improved_response": improved_text,
                "has_changes": improved_text != generator_response,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

            # Track performance
            track_performance = getattr(self.config, "track_performance", True)
            if track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms",
                    total_time + (time.time() - start_time) * 1000,
                )

            return improved_text, report

        except Exception as e:
            if hasattr(self, "record_error"):
                self.record_error(e)
            else:
                record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to close feedback loop: {str(e)}") from e

    def record_error(self, error: Exception) -> None:
        """Record an error in the state manager.

        Args:
            error: The exception to record
        """
        record_error(self._state_manager, error)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "validation_count": self._state_manager.get_metadata("validation_count", 0),
            "success_count": self._state_manager.get_metadata("success_count", 0),
            "failure_count": self._state_manager.get_metadata("failure_count", 0),
            "critique_count": self._state_manager.get_metadata("critique_count", 0),
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "total_processing_time_ms": self._state_manager.get_metadata(
                "total_processing_time_ms", 0
            ),
        }


def create_prompt_critic(
    llm_provider: Optional[Any] = None,
    name: str = "prompt_critic",
    description: str = "A critic that uses prompts to improve text",
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    cache_size: Optional[int] = None,
    priority: Optional[int] = None,
    cost: Optional[float] = None,
    track_performance: Optional[bool] = None,
    track_errors: Optional[bool] = None,
    eager_initialization: Optional[bool] = None,
    memory_buffer_size: Optional[int] = None,
    prompt_factory: Optional[Any] = None,
    config: Optional[PromptCriticConfig] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> PromptCritic:
    """
    Create a prompt critic with the given parameters.

    This factory function creates a configured PromptCritic instance with standardized
    state management and lifecycle handling. It uses the dependency injection system
    to resolve dependencies if not explicitly provided.

    ## Architecture
    The factory function follows the Factory Method pattern to:
    - Create standardized configuration objects
    - Instantiate critic classes with consistent parameters
    - Support dependency injection through kwargs
    - Provide type safety through return types
    - Handle error cases gracefully

    ## Lifecycle
    1. **Dependency Resolution**: Resolve required dependencies
       - Resolve llm_provider if not provided
       - Resolve prompt_factory if not provided
       - Handle dependency resolution errors

    2. **Configuration**: Create and validate configuration
       - Use default configuration as base
       - Apply provided parameter overrides
       - Validate configuration values
       - Handle configuration errors

    3. **Instantiation**: Create and initialize critic
       - Create PromptCritic instance
       - Initialize with resolved dependencies
       - Apply configuration
       - Handle initialization errors

    ## Examples
    ```python
    from sifaka.critics.implementations.prompt import create_prompt_critic
    from sifaka.models.providers import OpenAIProvider

    # Create with explicit provider
    provider = OpenAIProvider(api_key="your-api-key")
    critic = create_prompt_critic(
        llm_provider=provider,
        system_prompt="You are an expert editor",
        temperature=0.7
    )

    # Create with dependency injection
    critic = create_prompt_critic(
        system_prompt="You are an expert editor",
        temperature=0.7
    )

    # Create with custom configuration
    from sifaka.utils.config and config.critics import PromptCriticConfig
    config = PromptCriticConfig(
        name="custom_critic",
        description="A custom prompt critic",
        system_prompt="You are an expert editor",
        temperature=0.5,
        max_tokens=2000
    )
    critic = create_prompt_critic(
        llm_provider=provider,
        config=config
    )
    ```

    Args:
        llm_provider: The language model provider to use (injected if not provided)
        name: The name of the critic
        description: A description of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        cache_size: The size of the cache
        priority: The priority of the critic
        cost: The cost of the critic
        track_performance: Whether to track performance
        track_errors: Whether to track errors
        eager_initialization: Whether to initialize components eagerly
        memory_buffer_size: Size of the memory buffer
        prompt_factory: Optional prompt factory to use (injected if not provided)
        config: Optional configuration for the critic
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies
        **kwargs: Additional configuration parameters

    Returns:
        A configured PromptCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider
        DependencyError: If required dependencies cannot be resolved
    """
    try:
        # Resolve dependencies if not provided
        if llm_provider is None:
            from sifaka.core.dependency.provider import DependencyProvider
            from sifaka.core.dependency.provider import DependencyError

            # Get dependency provider
            provider = DependencyProvider()

            try:
                # Try to get by name first
                llm_provider = provider.get("model_provider", None, session_id, request_id)
            except DependencyError:
                try:
                    # Try to get by type if not found by name
                    from sifaka.interfaces.model import ModelProviderProtocol

                    # Using get method instead of get_by_type which might not exist
                    llm_provider = provider.get(
                        ModelProviderProtocol.__name__, None, session_id, request_id
                    )
                except (DependencyError, ImportError):
                    # This is a required dependency, so we need to raise an error
                    raise ValueError("Model provider is required for prompt critic")

        # Resolve prompt_factory if not provided
        if prompt_factory is None:
            from sifaka.core.dependency.provider import DependencyProvider
            from sifaka.core.dependency.provider import DependencyError

            # Get dependency provider
            provider = DependencyProvider()

            try:
                # Try to get by name
                prompt_factory = provider.get("prompt_factory", None, session_id, request_id)
            except DependencyError:
                # Prompt factory is optional, so we can continue without it
                pass

        # Create config if not provided
        if config is None:
            # Build the config dictionary with proper types
            config_kwargs: Dict[str, Any] = {}

            # Add all provided parameters to config_kwargs with proper typing
            config_kwargs["name"] = name
            config_kwargs["description"] = description

            if system_prompt is not None:
                config_kwargs["system_prompt"] = system_prompt
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            if max_tokens is not None:
                config_kwargs["max_tokens"] = max_tokens
            if min_confidence is not None:
                config_kwargs["min_confidence"] = min_confidence
            if max_attempts is not None:
                config_kwargs["max_attempts"] = max_attempts
            if cache_size is not None:
                config_kwargs["cache_size"] = cache_size
            if priority is not None:
                config_kwargs["priority"] = priority
            if cost is not None:
                config_kwargs["cost"] = cost
            if track_performance is not None:
                config_kwargs["track_performance"] = track_performance
            if track_errors is not None:
                config_kwargs["track_errors"] = track_errors
            if eager_initialization is not None:
                config_kwargs["eager_initialization"] = eager_initialization
            if memory_buffer_size is not None:
                config_kwargs["memory_buffer_size"] = memory_buffer_size

            # Add any additional kwargs to params
            params = kwargs.pop("params", {})
            for key, value in kwargs.items():
                if key not in config_kwargs and key not in ["session_id", "request_id"]:
                    params[key] = value

            if params:
                config_kwargs["params"] = params

            # Create a new config object
            config = PromptCriticConfig(**config_kwargs)

        # Create critic instance with configured parameters
        critic_config_dict = {
            "llm_provider": llm_provider,
            "prompt_factory": prompt_factory,
            "config": config,
        }

        # Create a BaseConfig for the component
        base_config = BaseConfig(name=name, description=description, params=critic_config_dict)

        # Use the created BaseConfig object
        critic = PromptCritic(name=name, description=description, config=base_config.params)

        return critic

    except Exception as e:
        # Log the error and re-raise with a clear message
        from ...utils.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Failed to create prompt critic: {str(e)}")
        raise ValueError(f"Failed to create prompt critic: {str(e)}") from e
