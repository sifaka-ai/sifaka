"""
Implementation of a prompt critic using a language model.

This module provides a critic that uses language models to evaluate,
validate, and improve text outputs based on rule violations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Protocol, runtime_checkable

from .base import (
    BaseCritic,
    CriticConfig,
    CriticMetadata,
    TextCritic,
    TextImprover,
    TextValidator,
)


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
    """Configuration for prompt critics."""

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
    """

    def __init__(
        self,
        name: str = "prompt_critic",
        description: str = "Evaluates and improves text using language models",
        llm_provider: Any = None,
        prompt_factory: Any = None,
        config: PromptCriticConfig = None,
        model: LanguageModel = None,
    ) -> None:
        """Initialize the prompt critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            llm_provider: Language model provider
            prompt_factory: Prompt factory
            config: Configuration for the critic
            model: Language model to use for critiquing (deprecated, use llm_provider instead)
        """
        # For backward compatibility
        if model is not None and llm_provider is None:
            llm_provider = model

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

        self._model = llm_provider
        self._prompt_factory = prompt_factory or DefaultPromptFactory()
        self.name = name
        self.description = description
        self.llm_provider = llm_provider
        self.prompt_factory = self._prompt_factory

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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if feedback is None:
            feedback = "Please improve this text for clarity and effectiveness."

        # Create improvement prompt using the prompt factory
        improve_prompt = self._prompt_factory.create_improvement_prompt(text, feedback)

        # Get improved version from the model
        try:
            response = self._model.invoke(improve_prompt)

            # Handle different response types
            if isinstance(response, dict) and "improved_text" in response:
                return response["improved_text"]
            elif isinstance(response, str):
                return response.strip()
            else:
                return "Failed to improve text: Invalid response format"
        except Exception as e:
            raise ValueError(f"Failed to improve text: {str(e)}") from e

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

        # Create critique prompt using the prompt factory
        critique_prompt = self._prompt_factory.create_critique_prompt(text)

        try:
            # Get response from the model
            response = self._model.invoke(critique_prompt)

            # Handle different response types
            if isinstance(response, dict):
                # Ensure all required fields are present
                result = {
                    "score": response.get("score", 0.5),
                    "feedback": response.get("feedback", "No feedback provided"),
                    "issues": response.get("issues", []),
                    "suggestions": response.get("suggestions", []),
                }
                return result
            elif isinstance(response, str):
                # Try to parse structured response
                sections = response.strip().split("\n")
                result = {"score": 0.5, "feedback": "", "issues": [], "suggestions": []}

                current_section = None
                for line in sections:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("SCORE:"):
                        try:
                            score_str = line.replace("SCORE:", "").strip()
                            result["score"] = float(score_str)
                        except ValueError:
                            result["score"] = 0.5
                    elif line.startswith("FEEDBACK:"):
                        result["feedback"] = line.replace("FEEDBACK:", "").strip()
                    elif line.startswith("ISSUES:"):
                        current_section = "issues"
                    elif line.startswith("SUGGESTIONS:"):
                        current_section = "suggestions"
                    elif line.startswith("-") and current_section:
                        item = line.replace("-", "").strip()
                        if item:
                            result[current_section].append(item)

                return result
            else:
                return {
                    "score": 0.0,
                    "feedback": "Failed to critique text: Invalid response format",
                    "issues": ["Invalid response format"],
                    "suggestions": ["Try again with clearer text"],
                }

        except Exception as e:
            # Return failure result if parsing fails
            return {
                "score": 0.0,
                "feedback": f"Failed to critique text: {str(e)}",
                "issues": ["Failed to parse model response"],
                "suggestions": ["Try again with clearer text"],
            }

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

        # Create validation prompt using the prompt factory
        validation_prompt = self._prompt_factory.create_validation_prompt(text)

        try:
            # Get response from the model
            response = self._model.invoke(validation_prompt)

            # Handle different response types
            if isinstance(response, dict):
                # Check if the response has a valid field
                if "valid" in response:
                    return response["valid"]
                # Fall back to score if available
                elif "score" in response:
                    return response["score"] >= self.config.min_confidence
                else:
                    return False
            elif isinstance(response, str):
                # Try to parse structured response
                if "VALID: true" in response.lower():
                    return True
                elif "VALID: false" in response.lower():
                    return False
                else:
                    # Fall back to critique if validation fails
                    critique_result = self.critique(text)
                    return critique_result.get("score", 0.0) >= self.config.min_confidence
            else:
                return False
        except ValueError as e:
            # Re-raise ValueError to match test expectations
            raise e
        except Exception:
            return False

    # Async methods
    async def avalidate(self, text: str) -> bool:
        """Asynchronously validate text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        validation_prompt = self._prompt_factory.create_validation_prompt(text)

        try:
            response = await self._model.ainvoke(validation_prompt)
            if isinstance(response, dict) and "valid" in response:
                return response["valid"]
            else:
                return False
        except Exception as e:
            raise ValueError(f"Failed to validate text: {str(e)}") from e

    async def acritique(self, text: str) -> dict:
        """Asynchronously critique text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        critique_prompt = self._prompt_factory.create_critique_prompt(text)

        try:
            response = await self._model.ainvoke(critique_prompt)
            if isinstance(response, dict):
                return response
            else:
                return {
                    "score": 0.0,
                    "feedback": "Failed to critique text: Invalid response format",
                    "issues": ["Invalid response format"],
                    "suggestions": ["Try again with clearer text"],
                }
        except Exception as e:
            raise ValueError(f"Failed to critique text: {str(e)}") from e

    async def aimprove(self, text: str, feedback: str) -> str:
        """Asynchronously improve text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        improve_prompt = self._prompt_factory.create_improvement_prompt(text, feedback)

        try:
            response = await self._model.ainvoke(improve_prompt)
            if isinstance(response, dict) and "improved_text" in response:
                return response["improved_text"]
            elif isinstance(response, str):
                return response.strip()
            else:
                return "Failed to improve text: Invalid response format"
        except Exception as e:
            raise ValueError(f"Failed to improve text: {str(e)}") from e


# CriticMetadata class for backward compatibility
class CriticMetadata(dict):
    """Metadata for critic results."""

    def __init__(
        self,
        score: float = 0.0,
        feedback: str = "",
        issues: list = None,
        suggestions: list = None,
        processing_time_ms: float = 0.0,
        **kwargs,
    ):
        """Initialize with critic metadata."""
        super().__init__(
            score=score,
            feedback=feedback,
            issues=issues or [],
            suggestions=suggestions or [],
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    @property
    def score(self) -> float:
        """Get the score."""
        return self.get("score", 0.0)

    @property
    def feedback(self) -> str:
        """Get the feedback."""
        return self.get("feedback", "")

    @property
    def issues(self) -> list:
        """Get the issues."""
        return self.get("issues", [])

    @property
    def suggestions(self) -> list:
        """Get the suggestions."""
        return self.get("suggestions", [])

    @property
    def processing_time_ms(self) -> float:
        """Get the processing time in milliseconds."""
        return self.get("processing_time_ms", 0.0)


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
    """Factory for creating prompt critics with default configurations."""

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
        model: LanguageModel,
        config: PromptCriticConfig = None,
    ) -> PromptCritic:
        """
        Create a prompt critic with the given model and configuration.

        Args:
            model: Language model to use for critiquing
            config: Optional configuration (uses default if None)

        Returns:
            PromptCritic: Configured prompt critic
        """
        if config is None:
            config = DEFAULT_PROMPT_CONFIG
        return PromptCritic(config=config, model=model)

    @staticmethod
    def create_with_custom_prompt(
        model: LanguageModel,
        system_prompt: str,
        min_confidence: float = 0.7,
        temperature: float = 0.7,
    ) -> PromptCritic:
        """
        Create a prompt critic with a custom system prompt.

        Args:
            model: Language model to use for critiquing
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
        return PromptCritic(config=config, model=model)


# Function to create a prompt critic
def create_prompt_critic(
    model: LanguageModel,
    name: str = "prompt_critic",
    description: str = "Evaluates and improves text using language models",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    min_confidence: float = 0.7,
) -> PromptCritic:
    """
    Create a prompt critic with the given parameters.

    Args:
        model: Language model to use for critiquing
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
    return PromptCritic(config=config, model=model)
