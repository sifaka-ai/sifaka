"""Prompt critic for Sifaka.

This module provides a general-purpose critic that uses language models to evaluate,
validate, and improve text outputs. It's a flexible critic that can be customized
with different prompts and system instructions for various critique tasks.

The PromptCritic is designed to be a versatile, general-purpose critic that can
be adapted for different use cases through prompt engineering.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class PromptCritic(ContextAwareMixin):
    """General-purpose critic that uses language models for text evaluation.

    This critic uses customizable prompts to evaluate and improve text. It's
    designed to be flexible and adaptable for various critique tasks through
    prompt engineering.

    Attributes:
        model: The language model to use for critique and improvement.
        system_prompt: The system prompt that defines the critic's behavior.
        critique_focus: Areas of focus for the critique (e.g., clarity, accuracy).
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        critique_focus: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Prompt critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            system_prompt: The system prompt that defines the critic's behavior.
            critique_focus: Areas of focus for the critique.
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

        # Set up system prompt
        self.system_prompt = system_prompt or (
            "You are an expert text critic and editor. Your role is to analyze text "
            "for quality, clarity, accuracy, and effectiveness. Provide constructive "
            "feedback and specific suggestions for improvement."
        )

        # Set up critique focus
        self.critique_focus = critique_focus or (
            "clarity, accuracy, coherence, completeness, and overall effectiveness"
        )

        # Set up prompt templates with context support
        self.critique_prompt_template = critique_prompt_template or (
            "Please analyze the following text and provide a detailed critique.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Focus your critique on: {focus}\n\n"
            "Please provide:\n"
            "1. An overall assessment of the text quality\n"
            "2. Specific strengths of the text\n"
            "3. Areas that need improvement\n"
            "4. Concrete suggestions for enhancement\n"
            "5. A score from 1-10 for overall quality\n"
            "6. How well the text uses the retrieved context (if available)\n\n"
            "Be specific, constructive, and actionable in your feedback."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique and feedback:\n{critique}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses all issues identified in the critique\n"
            "2. Maintains the core message and purpose\n"
            "3. Enhances clarity, accuracy, and effectiveness\n"
            "4. Stays true to the original task requirements\n"
            "5. Better incorporates relevant information from the context (if available)"
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text using the configured prompts.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.
        """
        start_time = time.time()

        with critic_context(
            critic_name="PromptCritic",
            operation="critique",
            message_prefix="Failed to critique text with Prompt critic",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "message": "No text available for critique",
                    "issues": ["Text is empty or None"],
                    "suggestions": ["Provide text to critique"],
                    "score": 0.0,
                }

            # Prepare context from retrieved documents (using mixin)
            context = self._prepare_context(thought)

            # Create critique prompt with context
            critique_prompt = self.critique_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                focus=self.critique_focus,
                context=context,
            )

            # Generate critique
            critique_response = self.model.generate(
                prompt=critique_prompt,
                system_prompt=self.system_prompt,
            )

            # Extract score and determine if improvement is needed
            score = self._extract_score(critique_response)
            needs_improvement = score < 7.0  # Threshold for improvement

            # Extract issues and suggestions
            issues = self._extract_issues(critique_response)
            suggestions = self._extract_suggestions(critique_response)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"PromptCritic: Critique completed in {processing_time:.2f}ms, "
                f"score: {score}, needs_improvement: {needs_improvement}"
            )

            return {
                "needs_improvement": needs_improvement,
                "message": critique_response,
                "critique": critique_response,
                "score": score,
                "issues": issues,
                "suggestions": suggestions,
                "focus_areas": self.critique_focus,
                "processing_time_ms": processing_time,
            }

    def improve(self, thought: Thought) -> str:
        """Improve text based on the critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.
        """
        start_time = time.time()

        with critic_context(
            critic_name="PromptCritic",
            operation="improve",
            message_prefix="Failed to improve text with Prompt critic",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="PromptCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            critique = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "PromptCritic":
                        critique = feedback.feedback.get("critique", "")
                        break

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with context
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                critique=critique,
                context=context,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt=self.system_prompt,
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"PromptCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _extract_score(self, critique: str) -> float:
        """Extract numerical score from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A score between 0.0 and 10.0.
        """
        import re

        # Look for score patterns like "score: 7", "7/10", "7 out of 10"
        score_patterns = [
            r"score[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)[/\s]*(?:out of\s*)?10",
            r"quality[:\s]+(\d+(?:\.\d+)?)",
            r"rating[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, critique.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-10 scale if needed
                    if score <= 1.0:
                        score *= 10
                    return min(10.0, max(0.0, score))
                except ValueError:
                    continue

        # Default score based on critique sentiment
        if any(
            word in critique.lower() for word in ["excellent", "great", "outstanding", "perfect"]
        ):
            return 8.5
        elif any(word in critique.lower() for word in ["good", "solid", "well"]):
            return 7.0
        elif any(word in critique.lower() for word in ["poor", "bad", "terrible", "awful"]):
            return 3.0
        else:
            return 5.5  # Neutral default

    def _extract_issues(self, critique: str) -> List[str]:
        """Extract issues from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of identified issues.
        """
        issues = []
        lines = critique.split("\n")

        for line in lines:
            line_lower = line.lower().strip()
            if any(
                indicator in line_lower
                for indicator in [
                    "issue",
                    "problem",
                    "weakness",
                    "flaw",
                    "error",
                    "mistake",
                    "unclear",
                    "confusing",
                    "missing",
                    "lacks",
                    "needs improvement",
                ]
            ):
                issues.append(line.strip())

        return issues[:5]  # Limit to top 5 issues

    def _extract_suggestions(self, critique: str) -> List[str]:
        """Extract suggestions from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of improvement suggestions.
        """
        suggestions = []
        lines = critique.split("\n")

        for line in lines:
            line_lower = line.lower().strip()
            if any(
                indicator in line_lower
                for indicator in [
                    "suggest",
                    "recommend",
                    "consider",
                    "should",
                    "could",
                    "try",
                    "improve",
                    "enhance",
                    "add",
                    "remove",
                    "clarify",
                ]
            ):
                suggestions.append(line.strip())

        return suggestions[:5]  # Limit to top 5 suggestions


def create_prompt_critic(
    model: Optional[Model] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    critique_focus: Optional[str] = None,
    **model_kwargs: Any,
) -> PromptCritic:
    """Create a Prompt critic.

    Args:
        model: The language model to use for critique and improvement.
        model_name: The name of the model to use if model is not provided.
        system_prompt: The system prompt that defines the critic's behavior.
        critique_focus: Areas of focus for the critique.
        **model_kwargs: Additional keyword arguments for model creation.

    Returns:
        A PromptCritic instance.
    """
    return PromptCritic(
        model=model,
        model_name=model_name,
        system_prompt=system_prompt,
        critique_focus=critique_focus,
        **model_kwargs,
    )
