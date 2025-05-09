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

from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable, Optional, Dict, List, Tuple

from pydantic import PrivateAttr, ConfigDict

from ..base import BaseCritic
from ..config import PromptCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator


class PromptCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
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
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[PromptCriticConfig] = None,
    ):
        """
        Initialize a prompt critic.

        Args:
            name: The name of the critic
            description: A description of the critic
            llm_provider: The language model provider to use
            prompt_factory: Optional prompt factory to use
            config: Optional configuration for the critic

        Raises:
            ValueError: If configuration is invalid
            TypeError: If llm_provider is not a valid provider
        """
        # Create config if not provided
        if config is None:
            from ..config import DEFAULT_PROMPT_CONFIG

            config = DEFAULT_PROMPT_CONFIG.model_copy(
                update={"name": name, "description": description}
            )

        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()
        self._state.initialized = False

        # Create components
        from ..managers.prompt_factories import PromptCriticPromptManager
        from ..managers.response import ResponseParser
        from ..services.critique import CritiqueService

        # Import memory manager
        from ..managers.memory import MemoryManager

        # Store components in state
        self._state.model = llm_provider
        self._state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
        self._state.response_parser = ResponseParser()
        self._state.memory_manager = MemoryManager(
            buffer_size=10
        )  # Same as ImprovementHistory max_history

        # Create services and store in state cache
        self._state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._state.prompt_manager,
            response_parser=self._state.response_parser,
            memory_manager=self._state.memory_manager,
        )

        # Mark as initialized
        self._state.initialized = True

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
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        improved_text = critique_service.improve(text, feedback)

        # Track improvement in memory manager
        import json

        memory_item = json.dumps(
            {
                "original_text": text,
                "feedback": feedback,
                "improved_text": improved_text,
                "timestamp": __import__("time").time(),
            }
        )
        self._state.memory_manager.add_to_memory(memory_item)

        return improved_text

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
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        improved_text = critique_service.improve(text, feedback)

        # Track improvement in memory manager
        import json

        memory_item = json.dumps(
            {
                "original_text": text,
                "feedback": feedback,
                "improved_text": improved_text,
                "timestamp": __import__("time").time(),
            }
        )
        self._state.memory_manager.add_to_memory(memory_item)

        return improved_text

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
        """
        improved_text = self.improve(text, feedback)

        # Get memory items and parse them
        import json

        memory_items = self._state.memory_manager.get_memory()
        parsed_items = []

        for item in memory_items:
            try:
                parsed_items.append(json.loads(item))
            except json.JSONDecodeError:
                # Skip items that can't be parsed
                continue

        return improved_text, parsed_items

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
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

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
        }

        return improved_text, report

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            dict: A dictionary containing critique information

        Raises:
            ValueError: If text is empty
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.critique(text)

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
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Get critique service from state
        critique_service = self._state.cache.get("critique_service")
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
        import json

        memory_item = json.dumps(
            {
                "original_text": text,
                "feedback": feedback,
                "improved_text": improved_text,
                "timestamp": __import__("time").time(),
            }
        )
        self._state.memory_manager.add_to_memory(memory_item)

        return improved_text

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
        """
        # Ensure initialized
        if not self._state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

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
        }

        return improved_text, report


def create_prompt_critic(
    llm_provider: Any,
    name: str = "prompt_critic",
    description: str = "A critic that uses prompts to improve text",
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_confidence: Optional[float] = None,
    max_attempts: Optional[int] = None,
    prompt_factory: Optional[Any] = None,
    config: Optional[PromptCriticConfig] = None,
) -> PromptCritic:
    """
    Create a prompt critic with the given parameters.

    Args:
        llm_provider: The language model provider to use
        name: The name of the critic
        description: A description of the critic
        system_prompt: The system prompt to use
        temperature: The temperature to use for generation
        max_tokens: The maximum number of tokens to generate
        min_confidence: The minimum confidence threshold
        max_attempts: The maximum number of attempts
        prompt_factory: Optional prompt factory to use
        config: Optional configuration for the critic

    Returns:
        A configured PromptCritic instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If llm_provider is not a valid provider
    """
    # Create config if not provided
    if config is None:
        from ..config import DEFAULT_PROMPT_CONFIG

        config = DEFAULT_PROMPT_CONFIG.model_copy()

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

        config = config.model_copy(update=updates)

    # Create and return the critic
    return PromptCritic(
        name=name,
        description=description,
        llm_provider=llm_provider,
        prompt_factory=prompt_factory,
        config=config,
    )
