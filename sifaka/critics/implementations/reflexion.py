"""
Reflexion critic module for Sifaka.

This module implements the Reflexion approach for critics, which enables language model
agents to learn from feedback without requiring weight updates. It maintains reflections
in memory to improve future text generation.

Example:
    ```python
    from sifaka.critics.implementations.reflexion import create_reflexion_critic
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a reflexion critic
    critic = create_reflexion_critic(llm_provider=provider)

    # Improve text with feedback
    text = "This is a sample technical document."
    feedback = "The text needs more detail and better structure."
    improved_text = critic.improve(text, feedback)
    ```
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import PrivateAttr, ConfigDict

from ..base import BaseCritic
from ..config import ReflexionCriticConfig
from ..interfaces.critic import TextCritic, TextImprover, TextValidator

# Configure logging
logger = logging.getLogger(__name__)


# Note: This class is kept for backward compatibility with tests
# In production code, use ReflexionCriticPromptManager from managers.prompt_factories instead
class ReflexionPromptFactory:
    """Factory for creating reflexion-specific prompts."""

    def create_validation_prompt(self, text: str) -> str:
        """Create a prompt for validating text."""
        return f"""Please validate the following text:

        TEXT TO VALIDATE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        VALID: [true/false]
        REASON: [reason for validation result]

        VALIDATION:"""

    def create_critique_prompt(self, text: str) -> str:
        """Create a prompt for critiquing text."""
        return f"""Please critique the following text:

        TEXT TO CRITIQUE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        CRITIQUE:"""

    def create_improvement_prompt(
        self, text: str, feedback: str, reflections: List[str] = None
    ) -> str:
        """Create a prompt for improving text with reflections."""
        reflection_text = ""
        if reflections and len(reflections) > 0:
            reflection_text = "\n\nPREVIOUS REFLECTIONS:\n"
            for i, reflection in enumerate(reflections):
                reflection_text += f"{i+1}. {reflection}\n"

        return f"""Please improve the following text:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}{reflection_text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [improved text]

        IMPROVEMENT:"""

    def create_reflection_prompt(self, text: str, feedback: str, improved_text: str) -> str:
        """Create a prompt for generating a reflection."""
        return f"""Please reflect on the following text improvement process:

        ORIGINAL TEXT:
        {text}

        FEEDBACK RECEIVED:
        {feedback}

        IMPROVED TEXT:
        {improved_text}

        Reflect on what went well, what went wrong, and what could be improved in future iterations.
        Focus on specific patterns, mistakes, or strategies that could be applied to similar tasks.

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        REFLECTION: [your reflection]

        REFLECTION:"""


class ReflexionCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """A critic that uses reflection to improve text quality.

    This critic implements the Reflexion approach, which enables learning from
    feedback without requiring weight updates. It maintains a memory buffer of
    past reflections to improve future text generation.

    ## Architecture

    The ReflexionCritic follows a component-based architecture with memory-augmented generation:

    1. **Core Components**
       - **ReflexionCritic**: Main class that implements the critic interfaces
       - **CritiqueService**: Service that handles the core critique functionality
       - **ReflexionCriticPromptManager**: Creates prompts with reflection context
       - **ResponseParser**: Parses and validates model responses
       - **MemoryManager**: Manages the memory buffer of past reflections
    """

    # Class constants
    DEFAULT_NAME = "reflexion_critic"
    DEFAULT_DESCRIPTION = "Improves text using reflections on past feedback"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using direct state
    _state = PrivateAttr(default_factory=lambda: None)

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        llm_provider: Any = None,
        prompt_factory: Any = None,
        config: ReflexionCriticConfig = None,
    ) -> None:
        """Initialize the reflexion critic."""
        # Validate required parameters
        if llm_provider is None:
            from pydantic import ValidationError

            raise ValidationError.from_exception_data(
                "Field required",
                [{"loc": ("llm_provider",), "msg": "Field required", "type": "missing"}],
            )

        # Create default config if not provided
        if config is None:
            from ..config import DEFAULT_REFLEXION_CONFIG

            config = DEFAULT_REFLEXION_CONFIG.model_copy(
                update={"name": name, "description": description}
            )

        # Initialize base class
        super().__init__(config)

        # Initialize state
        from ...utils.state import CriticState

        self._state = CriticState()

        # Import required components
        from ..managers.prompt_factories import ReflexionCriticPromptManager
        from ..managers.response import ResponseParser
        from ..managers.memory import MemoryManager
        from ..services.critique import CritiqueService

        # Store components in state
        self._state.model = llm_provider
        self._state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
        self._state.response_parser = ResponseParser()
        self._state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

        # Create service and store in state cache (not directly in state)
        self._state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._state.prompt_manager,
            response_parser=self._state.response_parser,
            memory_manager=self._state.memory_manager,
        )

        # Mark as initialized
        self._state.initialized = True

    @property
    def config(self) -> ReflexionCriticConfig:
        """Get the reflexion critic configuration."""
        return cast(ReflexionCriticConfig, self._config)

    def _check_input(self, text: str) -> None:
        """Validate input text and initialization state."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if not self._state.initialized or "critique_service" not in self._state.cache:
            raise RuntimeError("ReflexionCritic not properly initialized")

    def _format_feedback(self, feedback: Any) -> str:
        """Format feedback into a string."""
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
        """Check if text meets quality standards."""
        self._check_input(text)
        return self._state.cache["critique_service"].validate(text)

    def improve(self, text: str, feedback: str = None) -> str:
        """Improve text based on feedback and reflections."""
        self._check_input(text)
        feedback_str = self._format_feedback(feedback)
        return self._state.cache["critique_service"].improve(text, feedback_str)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback."""
        return self.improve(text, feedback)

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback."""
        self._check_input(text)
        return self._state.cache["critique_service"].critique(text)

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text."""
        self._check_input(text)
        return await self._state.cache["critique_service"].avalidate(text)

    async def acritique(self, text: str) -> dict:
        """Asynchronously critique text."""
        self._check_input(text)
        return await self._state.cache["critique_service"].acritique(text)

    async def aimprove(self, text: str, feedback: str = None) -> str:
        """Asynchronously improve text."""
        self._check_input(text)
        feedback_str = self._format_feedback(feedback)
        return await self._state.cache["critique_service"].aimprove(text, feedback_str)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        return await self.aimprove(text, feedback)


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
    config: Optional[Union[Dict[str, Any], ReflexionCriticConfig]] = None,
) -> ReflexionCritic:
    """Create a reflexion critic with the given parameters."""
    # Create config if not provided
    if config is None:
        from ..config import ReflexionCriticConfig

        config = ReflexionCriticConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            memory_buffer_size=memory_buffer_size,
            reflection_depth=reflection_depth,
        )
    elif isinstance(config, dict):
        from ..config import ReflexionCriticConfig

        config = ReflexionCriticConfig(**config)

    # Create and return the critic
    return ReflexionCritic(
        config=config, llm_provider=llm_provider, name=name, description=description
    )
