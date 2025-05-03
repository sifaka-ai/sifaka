"""
Reflexion critic module for Sifaka.

This module implements the Reflexion approach for critics, which enables language model
agents to learn from feedback without requiring weight updates. It employs a process
where agents reflect on feedback they receive and maintain these reflections in memory
to improve future decision-making.

The core concept behind Reflexion is verbal reinforcement learning - using language
itself as the mechanism for agent improvement. This approach allows language agents to
verbally reflect on task feedback signals and maintain these reflections in an episodic
memory buffer, which influences subsequent decision-making.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Final, List, cast

from .base import BaseCritic, CriticConfig
from .protocols import TextCritic, TextImprover, TextValidator
from .prompt import LanguageModel

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReflexionCriticConfig(CriticConfig):
    """Configuration for the reflexion critic."""

    system_prompt: str = "You are an expert editor that improves text through reflection."
    temperature: float = 0.7
    max_tokens: int = 1000
    memory_buffer_size: int = 5
    reflection_depth: int = 1  # How many levels of reflection to perform

    def __post_init__(self) -> None:
        """Validate reflexion critic specific configuration."""
        super().__post_init__()
        if not self.system_prompt or not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.memory_buffer_size < 0:
            raise ValueError("memory_buffer_size must be non-negative")
        if self.reflection_depth < 1:
            raise ValueError("reflection_depth must be positive")


class ReflexionPromptFactory:
    """Factory for creating reflexion-specific prompts."""

    def create_validation_prompt(self, text: str) -> str:
        """
        Create a prompt for validating text.

        Args:
            text: The text to validate

        Returns:
            str: The validation prompt
        """
        return f"""Please validate the following text:

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
        """
        Create a prompt for improving text with reflections.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement
            reflections: Optional list of previous reflections

        Returns:
            str: The improvement prompt
        """
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
        """
        Create a prompt for generating a reflection.

        Args:
            text: The original text
            feedback: Feedback received
            improved_text: The improved text

        Returns:
            str: The reflection prompt
        """
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
    """A critic that uses the Reflexion approach to improve text.

    This critic maintains a memory of reflections on previous improvements,
    which it uses to guide future improvements. It follows the Reflexion
    framework's approach of verbal reinforcement learning.

    This class follows the component-based architecture pattern by delegating to
    specialized components for prompt management, response parsing, and memory management.
    """

    def __init__(
        self,
        name: str = "reflexion_critic",
        description: str = "Improves text using reflections on past feedback",
        llm_provider: Any = None,
        prompt_factory: Any = None,
        config: ReflexionCriticConfig = None,
    ) -> None:
        """Initialize the reflexion critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider
            prompt_factory: Prompt factory
            config: Configuration for the critic
        """

        if llm_provider is None:
            from pydantic import ValidationError

            # Create a simple ValidationError for testing
            error = ValidationError.from_exception_data(
                "Field required",
                [{"loc": ("llm_provider",), "msg": "Field required", "type": "missing"}],
            )
            raise error

        if config is None:
            config = ReflexionCriticConfig(
                name=name,
                description=description,
                system_prompt="You are an expert editor that improves text through reflection.",
                temperature=0.7,
                max_tokens=1000,
                min_confidence=0.7,
                max_attempts=3,
                memory_buffer_size=5,
                reflection_depth=1,
            )

        super().__init__(config)

        # Create components
        from .managers.prompt_factories import ReflexionCriticPromptManager
        from .managers.response import ResponseParser
        from .managers.memory import MemoryManager
        from .services.critique import CritiqueService

        # Initialize components
        self._prompt_manager = prompt_factory or ReflexionCriticPromptManager(config)
        self._response_parser = ResponseParser()
        self._memory_manager = MemoryManager(buffer_size=config.memory_buffer_size)

        # Create service
        self._critique_service = CritiqueService(
            llm_provider=llm_provider,
            prompt_manager=self._prompt_manager,
            response_parser=self._response_parser,
            memory_manager=self._memory_manager,
        )

        # Store the language model provider
        self._model = llm_provider

    @property
    def config(self) -> ReflexionCriticConfig:
        """Get the reflexion critic configuration."""
        return cast(ReflexionCriticConfig, self._config)

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Delegate to critique service
        return self._critique_service.validate(text)

    def improve(self, text: str, feedback: str = None) -> str:
        """Improve text based on feedback and reflections.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement (can be a string or a list of violations)

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        elif feedback is None:
            feedback_str = "Please improve this text."
        else:
            feedback_str = feedback

        # Delegate to critique service - the critique service handles reflection generation
        return self._critique_service.improve(text, feedback_str)

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """Improve text based on specific feedback.

        This method implements the required abstract method from BaseCritic.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
        """
        return self.improve(text, feedback)

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Delegate to critique service
        return self._critique_service.critique(text)

    def _violations_to_feedback(self, violations: List[Dict[str, Any]]) -> str:
        """Convert rule violations to feedback text.

        Args:
            violations: List of rule violations

        Returns:
            str: Formatted feedback
        """
        if not violations:
            return "No issues found."

        feedback = "The following issues were found:\n"
        for i, violation in enumerate(violations):
            rule_name = violation.get("rule_name", f"Rule {i+1}")
            message = violation.get("message", "Unknown issue")
            feedback += f"- {rule_name}: {message}\n"

        return feedback

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """Parse a critique response string into a structured format.

        Args:
            response: The response string from the model

        Returns:
            dict: Structured critique data
        """
        result = {
            "score": 0.0,
            "feedback": "",
            "issues": [],
            "suggestions": [],
        }

        try:
            # Extract score
            if "SCORE:" in response:
                score_line = response.split("SCORE:")[1].split("\n")[0].strip()
                try:
                    result["score"] = float(score_line)
                except ValueError:
                    pass

            # Extract feedback
            if "FEEDBACK:" in response:
                feedback_parts = response.split("FEEDBACK:")[1].split("ISSUES:")[0].strip()
                result["feedback"] = feedback_parts

            # Extract issues
            if "ISSUES:" in response:
                issues_part = response.split("ISSUES:")[1]
                if "SUGGESTIONS:" in issues_part:
                    issues_part = issues_part.split("SUGGESTIONS:")[0]

                issues = []
                for line in issues_part.strip().split("\n"):
                    if line.strip().startswith("-"):
                        issues.append(line.strip()[1:].strip())
                result["issues"] = issues

            # Extract suggestions
            if "SUGGESTIONS:" in response:
                suggestions_part = response.split("SUGGESTIONS:")[1].strip()
                suggestions = []
                for line in suggestions_part.split("\n"):
                    if line.strip().startswith("-"):
                        suggestions.append(line.strip()[1:].strip())
                result["suggestions"] = suggestions

        except Exception:
            # Return default values if parsing fails
            pass

        return result

    def _get_relevant_reflections(self) -> List[str]:
        """Get relevant reflections from the memory buffer.

        Returns:
            List[str]: Relevant reflections
        """
        # Get reflections from memory manager
        return self._memory_manager.get_memory()

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Delegate to critique service
        return await self._critique_service.avalidate(text)

    async def acritique(self, text: str) -> dict:
        """
        Asynchronously critique text.

        Args:
            text: The text to critique

        Returns:
            Dictionary containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            TypeError: If model returns invalid output
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Delegate to critique service
        return await self._critique_service.acritique(text)

    async def aimprove(self, text: str, feedback: str = None) -> str:
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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Handle different feedback types
        if isinstance(feedback, list):
            # Convert violations to feedback string
            feedback_str = self._violations_to_feedback(feedback)
        elif feedback is None:
            feedback_str = "Please improve this text."
        else:
            feedback_str = feedback

        # Delegate to critique service - the critique service handles reflection generation
        return await self._critique_service.aimprove(text, feedback_str)

    async def aimprove_with_feedback(self, text: str, feedback: str) -> str:
        """Asynchronously improve text based on specific feedback.

        This method implements the required async abstract method.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty
        """
        return await self.aimprove(text, feedback)


# Default configurations
DEFAULT_REFLEXION_SYSTEM_PROMPT: Final[
    str
] = """You are an expert editor that improves text through reflection.
You maintain a memory of past improvements and use these reflections to guide
future improvements. Focus on learning patterns from past feedback and applying
them to new situations."""

DEFAULT_REFLEXION_CONFIG = ReflexionCriticConfig(
    name="Default Reflexion Critic",
    description="Evaluates and improves text using reflections on past feedback",
    system_prompt=DEFAULT_REFLEXION_SYSTEM_PROMPT,
    temperature=0.7,
    max_tokens=1000,
    min_confidence=0.7,
    max_attempts=3,
    memory_buffer_size=5,
    reflection_depth=1,
)


# Function to create a reflexion critic
def create_reflexion_critic(
    llm_provider: LanguageModel,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    system_prompt: str = DEFAULT_REFLEXION_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
) -> ReflexionCritic:
    """
    Create a reflexion critic with the given parameters.

    Args:
        llm_provider: Language model provider to use for critiquing
        name: Name of the critic
        description: Description of the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        min_confidence: Minimum confidence threshold
        memory_buffer_size: Maximum number of reflections to store
        reflection_depth: How many levels of reflection to perform

    Returns:
        ReflexionCritic: Configured reflexion critic
    """
    config = ReflexionCriticConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        min_confidence=min_confidence,
        memory_buffer_size=memory_buffer_size,
        reflection_depth=reflection_depth,
    )

    return ReflexionCritic(
        config=config, llm_provider=llm_provider, name=name, description=description
    )
