"""
Implementation of a prompt critic using a language model.

This module provides a critic that uses language models to evaluate,
validate, and improve text outputs based on rule violations.

## Component Lifecycle

### Prompt Critic Lifecycle

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

### Component Interactions

1. **Language Model Provider**
   - Receives formatted prompts
   - Returns model responses
   - Handles model-specific formatting
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import PrivateAttr, ConfigDict, Field

from ...core.base import BaseComponent
from ...utils.state import create_critic_state
from ...utils.common import record_error
from ...core.base import BaseResult as CriticResult
from ..config import PromptCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator


class PromptCritic(BaseComponent[str, CriticResult], TextValidator, TextImprover, TextCritic):
    """A critic that uses a language model to evaluate and improve text.

    This critic analyzes text for clarity, ambiguity, completeness, and effectiveness
    using a language model to generate feedback and validation scores.

    ## Architecture

    The PromptCritic follows a component-based architecture with clear separation of concerns:

    1. **Core Components**
       - **PromptCritic**: Main class that implements the critic interfaces
       - **CritiqueService**: Service that handles the core critique functionality
       - **PromptManager**: Manages prompt creation and formatting
       - **ResponseParser**: Parses and validates model responses
       - **MemoryManager**: Manages history of improvements and critiques

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

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    config: PromptCriticConfig = Field(description="Critic configuration")

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[PromptCriticConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize a prompt critic.

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
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_PROMPT_CONFIG

            config = DEFAULT_PROMPT_CONFIG.model_copy(
                update={"name": name, "description": description, **kwargs}
            )

        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        try:
            # Initialize state
            self._state_manager.update("initialized", False)
            self._state_manager.update("model", llm_provider)
            self._state_manager.update("prompt_factory", prompt_factory)
            self._state_manager.update("cache", {})

            # Store prompt factory in state for lazy initialization
            self._state_manager.update("prompt_factory", prompt_factory)

            # Set metadata
            self._state_manager.set_metadata("component_type", "critic")
            self._state_manager.set_metadata("critic_type", self.__class__.__name__)
            self._state_manager.set_metadata("name", name)
            self._state_manager.set_metadata("description", description)
            self._state_manager.set_metadata("creation_time", time.time())
            self._state_manager.set_metadata("validation_count", 0)
            self._state_manager.set_metadata("critique_count", 0)
            self._state_manager.set_metadata("improvement_count", 0)

            # Lazy initialization - components will be created when needed
            if self.config.eager_initialization:
                self._initialize_components()

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise ValueError(f"Failed to initialize PromptCritic: {str(e)}") from e

    def _initialize_components(self) -> None:
        """
        Initialize components needed for the critic.

        This method lazily initializes the components when they are first needed.
        It creates the prompt manager, response parser, memory manager, and critique service.

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

            # Create prompt manager if needed
            if not self._state_manager.get("prompt_manager"):
                from sifaka.core.managers.prompt_factories import PromptCriticPromptManager

                prompt_factory = self._state_manager.get("prompt_factory")
                self._state_manager.update(
                    "prompt_manager", prompt_factory or PromptCriticPromptManager(self.config)
                )

            # Create response parser if needed
            if not self._state_manager.get("response_parser"):
                from ..managers.response import ResponseParser

                self._state_manager.update("response_parser", ResponseParser())

            # Create memory manager if needed
            if not self._state_manager.get("memory_manager"):
                from sifaka.core.managers.memory import BufferMemoryManager as MemoryManager

                self._state_manager.update(
                    "memory_manager",
                    MemoryManager(buffer_size=self.config.memory_buffer_size or 10),
                )

            # Create critique service if needed
            cache = self._state_manager.get("cache", {})
            if "critique_service" not in cache:
                from ..services.critique import CritiqueService

                cache["critique_service"] = CritiqueService(
                    llm_provider=self._state_manager.get("model"),
                    prompt_manager=self._state_manager.get("prompt_manager"),
                    response_parser=self._state_manager.get("response_parser"),
                    memory_manager=self._state_manager.get("memory_manager"),
                )
                self._state_manager.update("cache", cache)

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("initialization_time", time.time())

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to initialize components: {str(e)}") from e

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
                raise RuntimeError("PromptCritic not properly initialized")

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
            # Use the standardized utility function
            record_error(self._state_manager, e)
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

    def improve(self, text: str, feedback: str = None) -> str:
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

            # Delegate to critique service
            improved_text = critique_service.improve(text, feedback)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
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
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("feedback must be a non-empty string")

            # Get critique service from state
            cache = self._state_manager.get("cache", {})
            critique_service = cache.get("critique_service")
            if not critique_service:
                raise RuntimeError("Critique service not initialized")

            # Delegate to critique service
            improved_text = critique_service.improve(text, feedback)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to improve text with feedback: {str(e)}") from e

    def improve_with_history(
        self, text: str, feedback: str = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Improve text and return both the result and improvement history.

        This method provides a way to track the improvement process and maintain
        a record of the changes made.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (optional)

        Returns:
            Tuple containing improved text and the improvement history as a list of dictionaries

        Raises:
            ValueError: If text is empty
            TypeError: If model returns non-string output
            RuntimeError: If critic is not properly initialized
        """
        start_time = time.time()

        try:
            # Improve the text
            improved_text = self.improve(text, feedback)

            # Get memory items and parse them
            memory_manager = self._state_manager.get("memory_manager")
            if not memory_manager:
                return improved_text, []

            memory_items = memory_manager.get_memory()
            parsed_items = []

            for item in memory_items:
                try:
                    parsed_items.append(json.loads(item))
                except json.JSONDecodeError:
                    # Skip items that can't be parsed
                    continue

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text, parsed_items

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to improve text with history: {str(e)}") from e

    def close_feedback_loop(
        self, text: str, generator_response: str, feedback: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Complete a feedback loop between a generator and this critic.

        This method explicitly handles the feedback loop by:
        1. Taking the original text and generator's response
        2. Providing feedback on the generator's response
        3. Improving the response based on feedback
        4. Returning both the improved response and a report of the process

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text, report

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
            raise RuntimeError(f"Failed to close feedback loop: {str(e)}") from e

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

            # Delegate to critique service
            result = critique_service.validate(text)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to validate text: {str(e)}") from e

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            dict: A dictionary containing critique information

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

            # Delegate to critique service
            result = critique_service.critique(text)

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Track performance
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to critique text: {str(e)}") from e

    async def aimprove(self, text: str, feedback: str = None) -> str:
        """Asynchronously improve text based on feedback.

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

            # Check if service supports async
            if hasattr(critique_service, "aimprove"):
                improved_text = await critique_service.aimprove(text, feedback)
            else:
                # Fallback to sync method in async context
                import asyncio

                improved_text = await asyncio.to_thread(critique_service.improve, text, feedback)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously improve text: {str(e)}") from e

    async def aclose_feedback_loop(
        self, text: str, generator_response: str, feedback: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Asynchronously complete a feedback loop between a generator and this critic.

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
                critique = await self.acritique(generator_response)
                feedback = critique["feedback"]

            # Improve the response
            improved_text = await self.aimprove(generator_response, feedback)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return improved_text, report

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously close feedback loop: {str(e)}") from e

    async def avalidate(self, text: str) -> bool:
        """Asynchronously check if text meets quality standards.

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

            # Check if service supports async
            if hasattr(critique_service, "avalidate"):
                result = await critique_service.avalidate(text)
            else:
                # Fallback to sync method in async context
                import asyncio

                result = await asyncio.to_thread(critique_service.validate, text)

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
            if self.config.track_performance:
                total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
                self._state_manager.set_metadata(
                    "total_processing_time_ms", total_time + (time.time() - start_time) * 1000
                )

            return result

        except Exception as e:
            self.record_error(e)
            raise RuntimeError(f"Failed to asynchronously validate text: {str(e)}") from e

    async def acritique(self, text: str) -> dict:
        """Asynchronously analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            dict: A dictionary containing critique information

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

            # Check if service supports async
            if hasattr(critique_service, "acritique"):
                result = await critique_service.acritique(text)
            else:
                # Fallback to sync method in async context
                import asyncio

                result = await asyncio.to_thread(critique_service.critique, text)

            # Update statistics
            critique_count = self._state_manager.get_metadata("critique_count", 0)
            self._state_manager.set_metadata("critique_count", critique_count + 1)

            # Track performance
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
            # Initialize components
            self._initialize_components()

            # Set warm-up metadata
            self._state_manager.set_metadata("warm_up_time", time.time())

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)
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

            # Release memory manager resources
            memory_manager = self._state_manager.get("memory_manager")
            if memory_manager and hasattr(memory_manager, "cleanup"):
                memory_manager.cleanup()

            # Release prompt manager resources
            prompt_manager = self._state_manager.get("prompt_manager")
            if prompt_manager and hasattr(prompt_manager, "cleanup"):
                prompt_manager.cleanup()

            # Release response parser resources
            response_parser = self._state_manager.get("response_parser")
            if response_parser and hasattr(response_parser, "cleanup"):
                response_parser.cleanup()

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
                "last_improvement_time": self._state_manager.get_metadata("last_improvement_time"),
                "model_provider": (
                    str(self._state_manager.get("model").__class__.__name__)
                    if self._state_manager.get("model")
                    else None
                ),
            }
        )

        return stats


def create_prompt_critic(
    llm_provider: Any = None,
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
            from sifaka.core.dependency import DependencyProvider, DependencyError

            # Get dependency provider
            provider = DependencyProvider()

            try:
                # Try to get by name first
                llm_provider = provider.get("model_provider", None, session_id, request_id)
            except DependencyError:
                try:
                    # Try to get by type if not found by name
                    from sifaka.interfaces.model import ModelProvider

                    llm_provider = provider.get_by_type(ModelProvider, None, session_id, request_id)
                except (DependencyError, ImportError):
                    # This is a required dependency, so we need to raise an error
                    raise ValueError("Model provider is required for prompt critic")

        # Resolve prompt_factory if not provided
        if prompt_factory is None:
            from sifaka.core.dependency import DependencyProvider, DependencyError

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
            from ..config import DEFAULT_PROMPT_CONFIG

            # Start with default config
            config = DEFAULT_PROMPT_CONFIG.model_copy()

            # Create updates dictionary with all provided parameters
            updates = {}

            # Add all provided parameters to updates
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
            if track_performance is not None:
                updates["track_performance"] = track_performance
            if track_errors is not None:
                updates["track_errors"] = track_errors
            if eager_initialization is not None:
                updates["eager_initialization"] = eager_initialization
            if memory_buffer_size is not None:
                updates["memory_buffer_size"] = memory_buffer_size

            # Add any additional kwargs to params
            params = kwargs.pop("params", {})
            for key, value in kwargs.items():
                if key not in updates and key not in ["session_id", "request_id"]:
                    params[key] = value

            if params:
                updates["params"] = params

            # Update config with all parameters
            config = config.model_copy(update=updates)

        # Create and return the critic with standardized state management
        return PromptCritic(
            name=name,
            description=description,
            llm_provider=llm_provider,
            prompt_factory=prompt_factory,
            config=config,
        )
    except Exception as e:
        # Log the error and re-raise with a clear message
        from ...utils.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Failed to create prompt critic: {str(e)}")
        raise ValueError(f"Failed to create prompt critic: {str(e)}") from e
