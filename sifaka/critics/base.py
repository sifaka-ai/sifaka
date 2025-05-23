"""Base critic implementations for Sifaka.

This module provides base critic implementations that can be used to critique
and improve text. Critics analyze text, identify issues, and provide suggestions
for improvement.

Critics are used in the Sifaka chain to improve the quality of generated text
by providing feedback that can be used to generate better text in subsequent
iterations.

Note: The ReflexionCritic has been moved to its own module at sifaka.critics.reflexion
"""

from typing import Any, Dict, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class BaseCritic(ContextAwareMixin):
    """Base critic class that provides common functionality for text critique and improvement.

    This base class provides a standard implementation for critics that use a language
    model to analyze text, identify issues, and provide suggestions for improvement.
    Other critics can inherit from this class and override specific methods as needed.

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

        Critics no longer handle retrieval - the Chain orchestrates all retrieval.
        Critics just use whatever context is already in the Thought container.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        # Set up the model (no retriever needed - Chain handles retrieval)
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
            "Retrieved context:\n{context}\n\n"
            "Please provide your critique in the following format:\n"
            "Issues:\n- [List issues here]\n\n"
            "Suggestions:\n- [List suggestions here]\n\n"
            "Consider how well the text uses information from the retrieved context (if available)."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version of the text that addresses the issues "
            "identified in the critique while staying true to the original prompt. "
            "Better incorporate relevant information from the context if available."
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text and provide feedback.

        Critics no longer handle retrieval - the Chain orchestrates all retrieval.
        Critics just use whatever context is already in the Thought container.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.

        Raises:
            ImproverError: If the critique fails.
        """
        with critic_context(
            critic_name="BaseCritic",
            operation="critique",
            message_prefix="Failed to critique text",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    "No text available for critique",
                    suggestions=["Provide text to critique"],
                )

            # Prepare context from retrieved documents (using mixin)
            context = self._prepare_context(thought)

            # Log context usage
            if self._has_context(thought):
                context_summary = self._get_context_summary(thought)
                logger.debug(f"BaseCritic using context: {context_summary}")

            # Format the critique prompt with context
            critique_prompt = self.critique_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
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

            # Determine if improvement is needed
            needs_improvement = len(issues) > 0 or "improvement" in critique_text.lower()

            # Create the critique result
            critique_result = {
                "needs_improvement": needs_improvement,
                "message": critique_text,
                "critique": critique_text,
                "critique_text": critique_text,
                "issues": issues,
                "suggestions": suggestions,
            }

            return critique_result

    def improve(self, thought: Thought) -> str:
        """Improve text based on critique.

        Critics no longer handle retrieval - the Chain orchestrates all retrieval.
        Critics just use whatever context is already in the Thought container.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the improvement fails.
        """
        with critic_context(
            critic_name="BaseCritic",
            operation="improvement",
            message_prefix="Failed to improve text",
        ):
            # Check if text and critique are available
            if not thought.text:
                raise ImproverError(
                    "No text available for improvement",
                    suggestions=["Provide text to improve"],
                )

            if not thought.critic_feedback:
                raise ImproverError(
                    "No critique available for improvement",
                    suggestions=["Run critique before improvement"],
                )

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Log context usage
            if self._has_context(thought):
                context_summary = self._get_context_summary(thought)
                logger.debug(f"BaseCritic using context for improvement: {context_summary}")

            # Format the improvement prompt with context
            critique_text = ""
            if thought.critic_feedback:
                # Extract critique text from the first critic feedback
                first_feedback = thought.critic_feedback[0]
                if hasattr(first_feedback, "feedback") and first_feedback.feedback:
                    critique_text = first_feedback.feedback.get("critique_text", "")
                elif hasattr(first_feedback, "violations") and first_feedback.violations:
                    critique_text = "; ".join(first_feedback.violations)
                elif hasattr(first_feedback, "suggestions") and first_feedback.suggestions:
                    critique_text = "; ".join(first_feedback.suggestions)

            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                critique=critique_text,
                context=context,
            )

            # Generate the improved text
            logger.debug("Generating improved text")
            improved_text = self.model.generate(improve_prompt)
            logger.debug(f"Generated improved text of length {len(improved_text)}")

            return improved_text
