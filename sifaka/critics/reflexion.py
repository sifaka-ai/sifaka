"""
Reflexion critic module for Sifaka.

This module implements the Reflexion approach for critics, which enables language model
agents to learn from feedback without requiring weight updates. It maintains reflections
in memory to improve future text generation.

Example:
    ```python
    from sifaka.critics.reflexion import create_reflexion_critic
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

from .base import BaseCritic
from .models import ReflexionCriticConfig
from .protocols import TextCritic, TextImprover, TextValidator
from .prompt import LanguageModel
from ..utils.state import create_critic_state

# Configure logging
logger = logging.getLogger(__name__)


# Import the Pydantic ReflexionCriticConfig from models.py


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

    ## Lifecycle Management

    The ReflexionCritic manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates configuration
       - Sets up language model provider
       - Initializes memory buffer
       - Allocates resources

    2. **Operation**
       - Validates text
       - Critiques text
       - Improves text based on feedback and reflections
       - Stores reflections in memory buffer

    3. **Cleanup**
       - Releases resources
       - Clears memory buffer

    ## Architecture

    The ReflexionCritic uses a memory-based architecture:

    1. **Memory Manager**: Stores and retrieves past reflections and feedback
    2. **Prompt Manager**: Creates prompts that incorporate past reflections
    3. **Response Parser**: Parses responses from language models
    4. **Critique Service**: Coordinates the critique and improvement process

    ## Error Handling

    The ReflexionCritic handles errors through:
    - Input validation
    - State validation
    - Exception propagation
    - Graceful degradation

    ## Examples

    Basic usage:

    ```python
    from sifaka.critics.reflexion import create_reflexion_critic
    from sifaka.models.openai import create_openai_provider

    # Create a language model provider
    provider = create_openai_provider(api_key="your-api-key")

    # Create a reflexion critic
    critic = create_reflexion_critic(
        llm_provider=provider,
        name="my_reflexion_critic",
        description="A critic that learns from past feedback",
        memory_buffer_size=5,
        reflection_depth=2
    )

    # Validate text
    text = "This is a sample technical document."
    is_valid = critic.validate(text)
    print(f"Is valid: {is_valid}")

    # Get critique
    critique = critic.critique(text)
    print(f"Feedback: {critique['feedback']}")

    # Improve text with feedback
    feedback = "The text needs more detail and better structure."
    improved_text = critic.improve_with_feedback(text, feedback)
    print(f"Improved text: {improved_text}")
    ```

    Advanced usage with custom configuration:

    ```python
    from sifaka.critics.reflexion import create_reflexion_critic, ReflexionCriticConfig
    from sifaka.models.anthropic import create_anthropic_provider

    # Create a language model provider
    provider = create_anthropic_provider(api_key="your-api-key")

    # Create a custom configuration
    config = ReflexionCriticConfig(
        name="custom_critic",
        description="A critic with custom prompts",
        system_prompt="You are an expert editor.",
        memory_buffer_size=10,
        reflection_depth=3,
        temperature=0.7,
        max_tokens=1000
    )

    # Create a reflexion critic with custom configuration
    critic = create_reflexion_critic(
        llm_provider=provider,
        config=config
    )

    # Use the critic with asynchronous methods
    import asyncio

    async def validate_and_improve_text():
        text = "This is a sample technical document."
        is_valid = await critic.avalidate(text)
        if not is_valid:
            critique = await critic.acritique(text)
            improved_text = await critic.aimprove(text, critique['feedback'])
            return improved_text
        return text

    improved_text = asyncio.run(validate_and_improve_text())
    ```
    """

    # Class constants
    DEFAULT_NAME = "reflexion_critic"
    DEFAULT_DESCRIPTION = "Improves text using reflections on past feedback"

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

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
            config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
                memory_buffer_size=5,
                reflection_depth=1,
            )

        # Initialize base class
        super().__init__(config)

        # Initialize state
        state = self._state_manager.get_state()

        # Import required components
        from .managers.prompt_factories import ReflexionCriticPromptManager
        from .managers.response import ResponseParser
        from .managers.memory import MemoryManager
        from .services.critique import CritiqueService

        # Store components in state
        state.model = llm_provider
        state.prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
        state.response_parser = ResponseParser()
        state.memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

        # Create service and store in state cache (not directly in state)
        state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
            memory_manager=state.memory_manager,
        )

        # Mark as initialized
        state.initialized = True

    @property
    def config(self) -> ReflexionCriticConfig:
        """Get the reflexion critic configuration."""
        return cast(ReflexionCriticConfig, self._config)

    def _check_input(self, text: str) -> None:
        """Validate input text and initialization state."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        state = self._state_manager.get_state()
        if not state.initialized or "critique_service" not in state.cache:
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
        state = self._state_manager.get_state()
        return state.cache["critique_service"].validate(text)

    def improve(self, text: str, feedback: str = None) -> str:
        """Improve text based on feedback and reflections."""
        self._check_input(text)
        feedback_str = self._format_feedback(feedback)
        state = self._state_manager.get_state()
        return state.cache["critique_service"].improve(text, feedback_str)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback."""
        return self.improve(text, feedback)

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback."""
        self._check_input(text)
        state = self._state_manager.get_state()
        return state.cache["critique_service"].critique(text)

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text."""
        self._check_input(text)
        state = self._state_manager.get_state()
        return await state.cache["critique_service"].avalidate(text)

    async def acritique(self, text: str) -> dict:
        """Asynchronously critique text."""
        self._check_input(text)
        state = self._state_manager.get_state()
        return await state.cache["critique_service"].acritique(text)

    async def aimprove(self, text: str, feedback: str = None) -> str:
        """Asynchronously improve text."""
        self._check_input(text)
        feedback_str = self._format_feedback(feedback)
        state = self._state_manager.get_state()
        return await state.cache["critique_service"].aimprove(text, feedback_str)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback."""
        return await self.aimprove(text, feedback)


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert editor that improves text through reflection.
You maintain a memory of past improvements and use these reflections to guide
future improvements. Focus on learning patterns from past feedback and applying
them to new situations."""


def create_reflexion_critic(
    llm_provider: LanguageModel,
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
    """Create a reflexion critic with the given parameters.

    Args:
        llm_provider: Language model provider
        name: Name of the critic
        description: Description of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        memory_buffer_size: Maximum number of reflections to store
        reflection_depth: How many levels of reflection to perform
        config: Optional pre-configured config

    Returns:
        ReflexionCritic: Configured reflexion critic
    """
    # Create config if not provided
    if config is None:
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
        config = ReflexionCriticConfig(**config)

    # Create and return the critic
    return ReflexionCritic(
        config=config, llm_provider=llm_provider, name=name, description=description
    )


"""
@misc{shinn2023reflexion,
      title={Reflexion: Language Agents with Verbal Reinforcement Learning},
      author={Noah Shinn and Federico Cassano and Edward Berman and Ashwin Gopinath and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2303.11366},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
"""
