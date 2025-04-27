"""
Implementation of a prompt critic using a language model.

This module provides a critic that uses language models to evaluate,
validate, and improve text outputs based on rule violations.
"""

from typing import Dict, Any, List, Protocol, runtime_checkable, Final
from dataclasses import dataclass
import time

from .base import (
    BaseCritic,
    CriticConfig,
    CriticMetadata,
    TextValidator,
    TextImprover,
    TextCritic,
)


@runtime_checkable
class LanguageModel(Protocol):
    """Protocol for language model interfaces."""

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
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
        config: PromptCriticConfig,
        model: LanguageModel,
    ) -> None:
        """Initialize the prompt critic.

        Args:
            config: Configuration for the critic
            model: Language model to use for critiquing
        """
        super().__init__(config)
        if not isinstance(model, LanguageModel):
            raise TypeError("model must implement LanguageModel protocol")
        self._model = model

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on rule violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            str: The improved text

        Raises:
            ValueError: If text is empty or violations is empty
            TypeError: If model returns non-string output
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")
        if not violations:
            raise ValueError("violations list cannot be empty")

        # Construct improvement prompt
        violation_text = "\n".join(
            f"- {v.get('rule', 'Unknown')}: {v.get('message', 'No message')}" for v in violations
        )

        improve_prompt = f"""{self.config.system_prompt}

        Please improve the following text to fix these violations:

        VIOLATIONS:
        {violation_text}

        ORIGINAL TEXT:
        {text}

        REQUIREMENTS:
        1. Fix all violations while preserving the key information
        2. Return ONLY the improved text, with no additional explanations
        3. Ensure the output maintains the original format
        4. Keep the length reasonable and appropriate

        IMPROVED TEXT:"""

        # Get improved version from the model
        try:
            improved = self._model.generate(improve_prompt)
            if not isinstance(improved, str):
                raise TypeError("Model must return a string")
            return improved.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to improve text: {str(e)}") from e

    def critique(self, text: str) -> CriticMetadata:
        """Analyze text and provide detailed feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing score, feedback, issues, and suggestions

        Raises:
            ValueError: If text is empty
            TypeError: If model returns invalid output
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        start_time = time.time()

        # Construct critique prompt
        critique_prompt = f"""{self.config.system_prompt}

        Please evaluate the following text and provide structured feedback:

        TEXT TO EVALUATE:
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

        Consider:
        1. Is the text clear and unambiguous?
        2. Is it complete and well-structured?
        3. Is it appropriate for its purpose?
        4. Could it be improved significantly?

        EVALUATION:"""

        try:
            # Get and parse response
            response = self._model.generate(critique_prompt)
            if not isinstance(response, str):
                raise TypeError("Model must return a string")

            # Parse structured response
            sections = response.strip().split("\n")
            result = {"score": 0.0, "feedback": "", "issues": [], "suggestions": []}

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

            # Create metadata with timing information
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return CriticMetadata(
                score=result["score"],
                feedback=result["feedback"],
                issues=result["issues"],
                suggestions=result["suggestions"],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            # Return failure metadata if parsing fails
            return CriticMetadata(
                score=0.0,
                feedback=f"Failed to critique text: {str(e)}",
                issues=["Failed to parse model response"],
                suggestions=["Try again with clearer text"],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def validate(self, text: str) -> bool:
        """Check if text meets quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text meets quality standards

        Raises:
            ValueError: If text is empty
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        try:
            metadata = self.critique(text)
            return metadata.score >= self.config.min_confidence
        except Exception:
            return False


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
