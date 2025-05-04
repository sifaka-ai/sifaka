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

2. **Prompt Factory**
   - Creates specialized prompts
   - Manages template variables
   - Validates prompt formats

3. **Response Parser**
   - Parses model responses
   - Validates response formats
   - Extracts structured data

### Error Handling and Recovery

1. **Input Validation Errors**
   - Empty or invalid text inputs
   - Invalid feedback format
   - Invalid configuration
   - Recovery: Return appropriate error messages

2. **Model Interaction Errors**
   - Provider connection failures
   - Response parsing errors
   - Format validation failures
   - Recovery: Retry with fallback strategies

3. **Resource Errors**
   - Memory allocation failures
   - Configuration issues
   - Recovery: Resource cleanup and state preservation

## Examples

```python
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
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

# Create a prompt critic
critic = PromptCritic(
    name="my_critic",
    description="A critic for improving technical documentation",
    llm_provider=provider,
    config=config
)

# Validate text
text = "This is a sample technical document."
is_valid = critic.validate(text)
print(f"Text is valid: {is_valid}")

# Critique text
critique = critic.critique(text)
print(f"Critique: {critique}")

# Improve text
improved_text = critic.improve(text)
print(f"Improved text: {improved_text}")
```
"""

from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable, Optional, Dict

from pydantic import PrivateAttr

from .base import (
    BaseCritic,
    CriticConfig,
    TextCritic,
    TextImprover,
    TextValidator,
)
from ..utils.state import create_critic_state


@runtime_checkable
class LanguageModel(Protocol):
    """Protocol for language model interfaces."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...

    def invoke(self, prompt: str) -> Any:
        """Invoke the model with a prompt and return structured output."""
        ...

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...


@dataclass(frozen=True)
class PromptCriticConfig(CriticConfig):
    """Configuration for the prompt critic."""

    system_prompt: str = "You are an expert editor that improves text."
    temperature: float = 0.7
    max_tokens: int = 1000

    def __post_init__(self) -> None:
        """Validate prompt critic specific configuration."""
        super().__post_init__()
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


class PromptCritic(BaseCritic, TextValidator, TextImprover, TextCritic):
    """A critic that uses a language model to evaluate and improve text.

    This critic analyzes text for clarity, ambiguity, completeness, and effectiveness
    using a language model to generate feedback and validation scores.

    ## Lifecycle Management

    The PromptCritic manages its lifecycle through three main phases:

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

    The PromptCritic implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies configuration

    2. **Model Interaction**
       - Handles provider errors
       - Manages response parsing
       - Validates output formats

    3. **Resource Management**
       - Handles allocation failures
       - Manages cleanup errors
       - Preserves valid state

    ## Examples

    ```python
    from sifaka.critics.prompt import PromptCritic, PromptCriticConfig
    from sifaka.models.providers import OpenAIProvider

    # Create a language model provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create a prompt critic
    critic = PromptCritic(
        name="my_critic",
        description="A critic for improving technical documentation",
        llm_provider=provider
    )

    # Validate text
    text = "This is a sample technical document."
    is_valid = critic.validate(text)
    print(f"Text is valid: {is_valid}")

    # Critique text
    critique = critic.critique(text)
    print(f"Critique: {critique}")

    # Improve text
    improved_text = critic.improve(text)
    print(f"Improved text: {improved_text}")
    ```

    This class follows the component-based architecture pattern by delegating to
    specialized components for prompt management, response parsing, and memory management.
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_critic_state)

    def __init__(
        self,
        name: str = "prompt_critic",
        description: str = "Evaluates and improves text using language models",
        llm_provider: Any = None,
        prompt_factory: Any = None,
        config: Optional[PromptCriticConfig] = None,
    ) -> None:
        """Initialize the prompt critic.

        Lifecycle:
        1. Configuration validation
        2. Provider setup
        3. Factory initialization
        4. Resource allocation

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider
            prompt_factory: Prompt factory
            config: Configuration for the critic

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider setup fails
        """
        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        if llm_provider is None:
            from pydantic import ValidationError

            # Create a simple ValidationError for testing
            error = ValidationError.from_exception_data(
                "Field required",
                [{"loc": ("llm_provider",), "msg": "Field required", "type": "missing"}],
            )
            raise error

        if config is None:
            config = PromptCriticConfig(
                name=name,
                description=description,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
            )

        super().__init__(config)

        # Create components
        from .managers.prompt_factories import PromptCriticPromptManager
        from .managers.response import ResponseParser
        from .services.critique import CritiqueService

        # Store components in state
        state.model = llm_provider
        state.prompt_manager = prompt_factory or PromptCriticPromptManager(config)
        state.response_parser = ResponseParser()
        state.memory_manager = None

        # Create service and store in state cache
        state.cache["critique_service"] = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=state.prompt_manager,
            response_parser=state.response_parser,
            memory_manager=state.memory_manager,
        )

        # Mark as initialized
        state.initialized = True

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
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.improve(text, feedback)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback.

        This method is similar to improve() but requires feedback to be provided.
        It uses the feedback to guide the improvement process.

        Args:
            text: The text to improve
            feedback: Required feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or feedback is empty
            TypeError: If model returns non-string output
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.improve(text, feedback)

    def critique(self, text: str) -> dict:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            TypeError: If model returns invalid output
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.critique(text)

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return critique_service.validate(text)

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """
        Asynchronously validate text.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return await critique_service.avalidate(text)

    async def acritique(self, text: str) -> dict:
        """
        Asynchronously critique text.

        Args:
            text: The text to critique

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return await critique_service.acritique(text)

    async def aimprove(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        # Get state
        state = self._state_manager.get_state()

        # Ensure initialized
        if not state.initialized:
            raise RuntimeError("PromptCritic not properly initialized")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Get critique service from state
        critique_service = state.cache.get("critique_service")
        if not critique_service:
            raise RuntimeError("Critique service not initialized")

        # Delegate to critique service
        return await critique_service.aimprove(text, feedback)


# Default configurations
DEFAULT_SYSTEM_PROMPT: Final[
    str
] = """You are an expert editor that improves text
while maintaining its core meaning and purpose. Focus on clarity, correctness,
and effectiveness."""

DEFAULT_PROMPT_CONFIG = PromptCriticConfig(
    name="Default Prompt Critic",
    description="Evaluates and improves text using language models",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
)


class DefaultPromptFactory:
    """Default implementation of a prompt factory.

    This class provides standard prompt templates for validation, critique,
    and improvement operations.

    ## Lifecycle Management

    The DefaultPromptFactory manages its lifecycle through three main phases:

    1. **Initialization**
       - Template setup
       - Format validation
       - Error handling setup

    2. **Operation**
       - Creates standard prompts
       - Handles template variables
       - Validates formats

    3. **Cleanup**
       - Resets state
       - Logs final status

    ## Error Handling

    The DefaultPromptFactory implements comprehensive error handling:

    1. **Input Validation**
       - Validates text input
       - Checks feedback format
       - Verifies template variables

    2. **Template Management**
       - Handles missing variables
       - Validates template syntax
       - Manages format errors

    ## Examples

    ```python
    from sifaka.critics.prompt import DefaultPromptFactory

    # Create a prompt factory
    factory = DefaultPromptFactory()

    # Create different types of prompts
    text = "This is a sample text."

    # Validation prompt
    validation_prompt = factory.create_validation_prompt(text)
    print(validation_prompt)

    # Critique prompt
    critique_prompt = factory.create_critique_prompt(text)
    print(critique_prompt)

    # Improvement prompt
    feedback = "The text needs more detail."
    improvement_prompt = factory.create_improvement_prompt(text, feedback)
    print(improvement_prompt)
    ```
    """

    def create_validation_prompt(self, text: str) -> str:
        """
        Create a prompt for validating text.

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt
        """
        return f"""Please Validate the following text:

        TEXT TO VALIDATE:
        {text}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        VALID: [true/false]
        REASON: [reason for validation result]

        VALIDATION:"""

    def create_critique_prompt(self, text: str) -> str:
        """
        Create a prompt for critiquing text.

        Args:
            text: The text to critique

        Returns:
            str: The critique prompt
        """
        return f"""Please Critique the following text:

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

    def create_improvement_prompt(self, text: str, feedback: str) -> str:
        """
        Create a prompt for improving text.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improvement prompt
        """
        return f"""Please Improve the following text:

        TEXT TO IMPROVE:
        {text}

        FEEDBACK:
        {feedback}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        IMPROVED_TEXT: [improved text]

        IMPROVEMENT:"""

    @staticmethod
    def create_critic(
        llm_provider: LanguageModel,
        config: PromptCriticConfig = None,
    ) -> PromptCritic:
        """
        Create a prompt critic with the given language model provider and configuration.

        Args:
            llm_provider: Language model provider to use for critiquing
            config: Optional configuration (uses default if None)

        Returns:
            PromptCritic: Configured prompt critic
        """
        if config is None:
            config = DEFAULT_PROMPT_CONFIG
        return PromptCritic(config=config, llm_provider=llm_provider)

    @staticmethod
    def create_with_custom_prompt(
        llm_provider: LanguageModel,
        system_prompt: str,
        min_confidence: float = 0.7,
        temperature: float = 0.7,
    ) -> PromptCritic:
        """
        Create a prompt critic with a custom system prompt.

        Args:
            llm_provider: Language model provider to use for critiquing
            system_prompt: Custom system prompt
            min_confidence: Minimum confidence threshold
            temperature: Temperature for model generation

        Returns:
            PromptCritic: Configured prompt critic
        """
        config = PromptCriticConfig(
            name="Custom Prompt Critic",
            description="Custom prompt critic",
            system_prompt=system_prompt,
            temperature=temperature,
            min_confidence=min_confidence,
        )
        return PromptCritic(config=config, llm_provider=llm_provider)


# Function to create a prompt critic
def create_prompt_critic(
    llm_provider: LanguageModel,
    name: str = "factory_critic",
    description: str = "Evaluates and improves text using language models",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
) -> PromptCritic:
    """
    Create a prompt critic with the given parameters.

    Args:
        llm_provider: Language model provider to use for critiquing
        name: Name of the critic
        description: Description of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        min_confidence: Minimum confidence threshold

    Returns:
        PromptCritic: Configured prompt critic
    """
    config = PromptCriticConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        min_confidence=min_confidence,
    )
    return PromptCritic(
        config=config, llm_provider=llm_provider, name=name, description=description
    )
