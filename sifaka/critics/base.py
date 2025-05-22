"""Base critic implementations for Sifaka.

This module provides base critic implementations that can be used to critique
and improve text. Critics analyze text, identify issues, and provide suggestions
for improvement.

Critics are used in the Sifaka chain to improve the quality of generated text
by providing feedback that can be used to generate better text in subsequent
iterations.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Model
from sifaka.core.thought import CriticFeedback, Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class ReflexionCritic:
    """Critic that uses a language model to critique and improve text.

    This critic uses a language model to analyze text, identify issues,
    and provide suggestions for improvement. It then uses the same model
    to generate improved text based on the critique.

    Attributes:
        model: The language model to use for critique and improvement.
        critique_prompt_template: Template for the critique prompt.
        improve_prompt_template: Template for the improvement prompt.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        # Set up the model
        if model:
            self.model = model
        elif model_name:
            self.model = create_model(model_name, **model_kwargs)
        else:
            # Default to a mock model for testing
            self.model = create_model("mock:default", **model_kwargs)

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Please critique the following text and identify any issues or areas for improvement. "
            "Focus on clarity, coherence, accuracy, and relevance to the original prompt.\n\n"
            "Original prompt: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Please provide your critique in the following format:\n"
            "Issues:\n- [List issues here]\n\n"
            "Suggestions:\n- [List suggestions here]"
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version of the text that addresses the issues "
            "identified in the critique while staying true to the original prompt."
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text and provide feedback.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.

        Raises:
            ImproverError: If the critique fails.
        """
        with critic_context(
            critic_name="ReflexionCritic",
            operation="critique",
            message_prefix="Failed to critique text",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    "No text available for critique",
                    suggestions=["Provide text to critique"],
                )

            # Format the critique prompt
            critique_prompt = self.critique_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
            )

            # Generate the critique
            logger.debug("Generating critique")
            critique_text = self.model.generate(critique_prompt)
            logger.debug(f"Generated critique of length {len(critique_text)}")

            # Parse the critique
            issues = []
            suggestions = []

            # Simple parsing logic - can be improved
            in_issues = False
            in_suggestions = False

            for line in critique_text.split("\n"):
                line = line.strip()
                if line.lower().startswith("issues:"):
                    in_issues = True
                    in_suggestions = False
                    continue
                elif line.lower().startswith("suggestions:"):
                    in_issues = False
                    in_suggestions = True
                    continue
                elif not line or line.startswith("#"):
                    continue

                if in_issues and line.startswith("-"):
                    issues.append(line[1:].strip())
                elif in_suggestions and line.startswith("-"):
                    suggestions.append(line[1:].strip())

            # Create the critique result
            critique_result = {
                "critique_text": critique_text,
                "issues": issues,
                "suggestions": suggestions,
            }

            # Add critic feedback to the thought
            feedback = CriticFeedback(
                critic_name="ReflexionCritic",
                issues=issues,
                suggestions=suggestions,
            )

            return critique_result

    def improve(self, thought: Thought) -> str:
        """Improve text based on critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the improvement fails.
        """
        with critic_context(
            critic_name="ReflexionCritic",
            operation="improvement",
            message_prefix="Failed to improve text",
        ):
            # Check if text and critique are available
            if not thought.text:
                raise ImproverError(
                    "No text available for improvement",
                    suggestions=["Provide text to improve"],
                )

            if not thought.critique:
                raise ImproverError(
                    "No critique available for improvement",
                    suggestions=["Run critique before improvement"],
                )

            # Format the improvement prompt
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                critique=thought.critique.get("critique_text", ""),
            )

            # Generate the improved text
            logger.debug("Generating improved text")
            improved_text = self.model.generate(improve_prompt)
            logger.debug(f"Generated improved text of length {len(improved_text)}")

            return improved_text
