"""
Constitutional critic module for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on Constitutional AI: https://arxiv.org/abs/2212.08073

Example:
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
    is_valid = critic.validate(response, metadata={"task": task})
    print(f"Response is valid: {is_valid}")

    # Get critique for a response
    critique = critic.critique(response, metadata={"task": task})
    print(f"Critique: {critique}")

    # Improve a response
    improved_response = critic.improve(response, metadata={"task": task})
    print(f"Improved response: {improved_response}")
    ```
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...core.base import BaseResult as CriticResult
from ..config import ConstitutionalCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = logging.getLogger(__name__)


class ConstitutionalCritic(
    BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic
):
    """
    A critic that evaluates responses against a list of principles (a "constitution")
    and provides natural language feedback for revision.

    Based on Constitutional AI: https://arxiv.org/abs/2212.08073

    This critic analyzes responses for alignment with specified principles and
    generates critiques when violations are detected.

    ## Architecture

    The ConstitutionalCritic follows a component-based architecture with principles-based evaluation:

    1. **Core Components**
       - **ConstitutionalCritic**: Main class that implements the critic interfaces
       - **PrinciplesManager**: Manages the list of principles (the "constitution")
       - **CritiqueGenerator**: Evaluates responses against principles
       - **ResponseImprover**: Improves responses based on critiques
       - **PromptManager**: Creates specialized prompts for critique and improvement

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

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        principles: Optional[List[str]] = None,
        config: Optional[ConstitutionalCriticConfig] = None,
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
            from ..config import DEFAULT_CONSTITUTIONAL_CONFIG

            config = DEFAULT_CONSTITUTIONAL_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

            # Override principles if provided
            if principles is not None:
                config.principles = principles

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Import required components
            from ..managers.prompt_factories import ConstitutionalCriticPromptManager
            from ..managers.response import ResponseParser
            from ..managers.memory import MemoryManager
            from ..services.critique import CritiqueService

            # Store components in state
            self._state_manager.update("model", llm_provider)
            self._state_manager.update("prompt_manager", ConstitutionalCriticPromptManager(config))
            self._state_manager.update("response_parser", ResponseParser())
            self._state_manager.update(
                "memory_manager", MemoryManager(buffer_size=10)  # Default buffer size
            )

            # Create service and store in state cache
            cache = self._state_manager.get("cache", {})
            cache["critique_service"] = CritiqueService(
                llm_provider=llm_provider,
                prompt_manager=self._state_manager.get("prompt_manager"),
                response_parser=self._state_manager.get("response_parser"),
                memory_manager=self._state_manager.get("memory_manager"),
            )
            cache["principles"] = config.principles
            cache["critique_prompt_template"] = config.critique_prompt_template
            cache["improvement_prompt_template"] = config.improvement_prompt_template
            cache["system_prompt"] = config.system_prompt
            cache["temperature"] = config.temperature
            cache["max_tokens"] = config.max_tokens
            self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            self.record_error(e)
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
        start_time = time.time()

        try:
            # Validate input
            if not isinstance(input, str) or not input.strip():
                raise ValueError("Input must be a non-empty string")

            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

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

    def _format_principles(self) -> str:
        """
        Format principles as a bulleted list.

        Returns:
            Formatted principles as a string
        """
        principles = self._state_manager.get("cache", {}).get("principles", [])
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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

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

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Delegate to critique service
            critique_result = critique_service.critique(text, {"task": task})
            is_valid = len(critique_result.get("issues", [])) == 0

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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

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

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Delegate to critique service
            critique_result = critique_service.critique(text, {"task": task})

            # Track score distribution
            score_distribution = self._state_manager.get_metadata("score_distribution", {})
            score_bucket = round(critique_result.get("score", 0) * 10) / 10  # Round to nearest 0.1
            score_distribution[str(score_bucket)] = score_distribution.get(str(score_bucket), 0) + 1
            self._state_manager.set_metadata("score_distribution", score_distribution)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_critique_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_critique_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return critique_result

        except Exception as e:
            self.record_error(e)
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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

            # Validate input
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Track improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Get task from metadata
            task = self._get_task_from_metadata(metadata)

            # Delegate to critique service
            improved_text = critique_service.improve(text, {"task": task})

            # Track memory usage
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager:
                memory_item = json.dumps(
                    {
                        "original_text": text,
                        "task": task,
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
        start_time = time.time()

        try:
            # Ensure initialized
            if not self._state_manager.get("initialized", False):
                raise RuntimeError("ConstitutionalCritic not properly initialized")

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

            # Track feedback improvement count
            feedback_count = self._state_manager.get_metadata("feedback_improvement_count", 0)
            self._state_manager.set_metadata("feedback_improvement_count", feedback_count + 1)

            # Format principles
            principles_text = self._format_principles()

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
            model = self._state_manager.get("model")
            improved_text = model.generate(
                prompt,
                system_prompt=self._state_manager.get("cache", {}).get("system_prompt", ""),
                temperature=self._state_manager.get("cache", {}).get("temperature", 0.7),
                max_tokens=self._state_manager.get("cache", {}).get("max_tokens", 1000),
            ).strip()

            # Track memory usage
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
            "feedback_improvement_count": self._state_manager.get_metadata(
                "feedback_improvement_count", 0
            ),
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
    async def avalidate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Asynchronously validate text."""
        # For now, use the synchronous implementation
        return self.validate(text, metadata)

    async def acritique(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously critique text."""
        # For now, use the synchronous implementation
        return self.critique(text, metadata)

    async def aimprove(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Asynchronously improve text."""
        # For now, use the synchronous implementation
        return self.improve(text, metadata)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        # For now, use the synchronous implementation
        return self.improve_with_feedback(text, feedback)


def create_constitutional_critic(
    llm_provider: Any,
    principles: List[str] = None,
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = None,
    max_attempts: int = None,
    cache_size: int = None,
    priority: int = None,
    cost: float = None,
    system_prompt: str = None,
    temperature: float = None,
    max_tokens: int = None,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic with the given parameters.

    This factory function creates and configures a ConstitutionalCritic instance with
    the specified parameters and components.

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
            from ..config import DEFAULT_CONSTITUTIONAL_CONFIG

            # Start with default config
            config = DEFAULT_CONSTITUTIONAL_CONFIG.model_copy()

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
            updates.update(kwargs)

            # Create updated config
            config = config.model_copy(update=updates)
        elif isinstance(config, dict):
            from ..config import ConstitutionalCriticConfig

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
        logger.error(f"Failed to create constitutional critic: {str(e)}")
        raise ValueError(f"Failed to create constitutional critic: {str(e)}") from e
