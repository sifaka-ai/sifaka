"""
Implementation of a prompt critic using composition over inheritance.

This module provides a critic implementation that uses language models to evaluate,
validate, and improve text outputs based on rule violations. It follows the
composition over inheritance pattern.

## Component Lifecycle

### Prompt Critic Implementation Lifecycle

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

## Examples

```python
from sifaka.critics.implementations.prompt_implementation import PromptCriticImplementation
from sifaka.critics.base import CompositionCritic, create_composition_critic
from sifaka.critics.models import PromptCriticConfig
from sifaka.models.providers import OpenAIProvider

# Create a language model provider
provider = OpenAIProvider(api_key="your-api-key")

# Create a prompt critic configuration
config = PromptCriticConfig(
    name="my_critic",
    description="A critic for improving technical documentation",
    system_prompt="You are an expert technical writer.",
    temperature=0.7,
    max_tokens=1000
)

# Create a prompt critic implementation
implementation = PromptCriticImplementation(config, provider)

# Create a critic with the implementation
critic = create_composition_critic(
    name="my_critic",
    description="A critic for improving technical documentation",
    implementation=implementation
)

# Use the critic
text = "This is a sample technical document."
is_valid = critic.validate(text)
critique = critic.critique(text)
improved_text = critic.improve(text, "Add more detail")
```
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ConfigDict, PrivateAttr

from ..models import PromptCriticConfig, CriticConfig, CriticMetadata
from ..protocols import CriticImplementation
from ...utils.state import CriticState, create_critic_state

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert editor and writing coach who helps improve text quality.
Your task is to evaluate, critique, and improve text based on clarity, coherence, grammar, and effectiveness.
Provide specific, actionable feedback and suggestions for improvement."""

# Default prompt critic configuration
DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="default_prompt_critic",
    description="Default prompt critic configuration",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
)


class PromptCriticImplementation:
    """
    Implementation of a prompt critic using language models.

    This class implements the CriticImplementation protocol for a prompt-based critic
    that uses language models to evaluate, validate, and improve text.

    ## Lifecycle Management

    The PromptCriticImplementation manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes prompt factory
       - Allocates resources

    2. **Operation**
       - Validates text input
       - Generates critiques
       - Improves text quality
       - Processes feedback

    3. **Cleanup**
       - Releases resources
       - Resets state
       - Logs final status

    ## Error Handling

    The implementation handles various error conditions:
    - Empty or invalid input text
    - Model generation failures
    - Response parsing errors
    - State initialization issues
    """

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        config: Union[CriticConfig, PromptCriticConfig],
        llm_provider: Any,
        prompt_factory: Any = None,
    ) -> None:
        """
        Initialize the prompt critic implementation.

        Args:
            config: Configuration for the critic
            llm_provider: Language model provider
            prompt_factory: Optional prompt factory

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider setup fails
        """
        self.config = config

        # Initialize state
        state = self._state_manager.get_state()

        # Create components
        from ..managers.prompt_factories import PromptCriticPromptManager
        from ..managers.response import ResponseParser
        from ..services.critique import CritiqueService
        from ..managers.memory import MemoryManager

        # Store components in state
        state.model = llm_provider
        state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
        state.response_parser = ResponseParser()
        state.memory_manager = MemoryManager(buffer_size=10)

        # Create service and store in state cache
        state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
            memory_manager=state.memory_manager,
        )

        # Mark as initialized
        state.initialized = True

    def validate_impl(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            return False

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    def improve_impl(self, text: str, feedback: Optional[Any] = None) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If improvement fails
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        improved_text = critique_service.improve(text, feedback)

        # Track improvement in memory manager
        memory_item = json.dumps(
            {
                "original_text": text,
                "feedback": feedback,
                "improved_text": improved_text,
                "timestamp": __import__("time").time(),
            }
        )
        state.memory_manager.add_to_memory(memory_item)

        return improved_text

    def critique_impl(self, text: str) -> Dict[str, Any]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            Dict[str, Any]: A dictionary containing critique information

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        # Get state
        state = self._state_manager.get_state()

        if not state.initialized:
            raise RuntimeError("PromptCriticImplementation not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        critique = critique_service.critique(text)

        # Convert to dictionary format
        return {
            "score": critique.score,
            "feedback": critique.feedback,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
        }

    def warm_up_impl(self) -> None:
        """
        Warm up the critic implementation.

        This method initializes any resources needed by the critic implementation.
        """
        # Get state to ensure it's initialized
        state = self._state_manager.get_state()

        # Already initialized in __init__, but we can check and re-initialize if needed
        if not state.initialized:
            # Create components
            from ..managers.prompt_factories import PromptCriticPromptManager
            from ..managers.response import ResponseParser
            from ..services.critique import CritiqueService
            from ..managers.memory import MemoryManager

            # Initialize components if not already done
            if not hasattr(state, "prompt_manager") or state.prompt_manager is None:
                state.prompt_manager = PromptCriticPromptManager(self.config)

            if not hasattr(state, "response_parser") or state.response_parser is None:
                state.response_parser = ResponseParser()

            if not hasattr(state, "memory_manager") or state.memory_manager is None:
                state.memory_manager = MemoryManager(buffer_size=10)

            # Initialize critique service if not already done
            if "critique_service" not in state.cache:
                critique_service = CritiqueService(
                    llm_provider=state.model,
                    prompt_manager=state.prompt_manager,
                    response_parser=state.response_parser,
                    memory_manager=state.memory_manager,
                )
                state.cache["critique_service"] = critique_service

            # Mark as initialized
            state.initialized = True
