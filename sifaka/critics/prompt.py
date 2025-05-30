"""Prompt-based critic for Sifaka.

This module implements a simple, customizable prompt-based critic that allows
users to define their own critique criteria through custom prompts.

The PromptCritic provides a flexible foundation for creating domain-specific
critics without requiring complex implementations.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.critics.mixins.validation_aware import ValidationAwareMixin
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.validators.validation_context import create_validation_context

logger = get_logger(__name__)


class PromptCritic(BaseCritic, ValidationAwareMixin):
    """A simple, customizable prompt-based critic with validation awareness.

    This critic allows users to define their own critique criteria through
    custom prompts, making it easy to create domain-specific critics without
    complex implementations.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting prompt-based suggestions.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Prompt critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            critique_prompt_template: Custom template for the critique prompt.
            improve_prompt_template: Custom template for the improvement prompt.
            system_prompt: Custom system prompt for the critic.
            criteria: List of specific criteria to evaluate.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        self.criteria = criteria or [
            "Clarity and readability",
            "Accuracy and factual correctness",
            "Completeness and thoroughness",
            "Relevance to the task",
        ]

        self.system_prompt = system_prompt or (
            "You are an expert critic providing detailed, constructive feedback on text quality."
        )

        # Set up prompt templates
        criteria_text = "\n".join(f"- {criterion}" for criterion in self.criteria)

        self.critique_prompt_template = critique_prompt_template or (
            "Please critique the following text based on these criteria:\n\n"
            f"{criteria_text}\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please provide your critique in the following format:\n\n"
            "Issues:\n- [List specific issues here]\n\n"
            "Suggestions:\n- [List specific suggestions here]\n\n"
            "Overall Assessment: [Brief overall assessment]\n\n"
            "Be specific and constructive in your feedback. Consider how well the text "
            "uses information from the retrieved context (if available)."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text based on the critique provided.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version that addresses the issues identified "
            "in the critique while maintaining the core message and staying true to "
            "the original task. Better incorporate relevant information from the context if available.\n\n"
            "Improved text:"
        )

        # Store the last improvement prompt used for debugging/logging
        self.last_improvement_prompt = None

    def _perform_critique(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using custom prompt.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt with context
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique
        critique_response = self.model.generate(
            prompt=critique_prompt,
            system_prompt=self.system_prompt,
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)

        # Determine if improvement is needed based on critique content
        needs_improvement = self._needs_improvement(critique_response)

        logger.debug("PromptCritic: Critique completed")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.7,  # Default confidence for prompt-based critic
            "metadata": {
                "criteria": self.criteria,
                "system_prompt": self.system_prompt,
            },
        }

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using custom prompt asynchronously.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt with context
        critique_prompt = self.critique_prompt_template.format(
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique asynchronously if the model supports it
        if hasattr(self.model, "generate_async"):
            critique_response = await self.model.generate_async(
                prompt=critique_prompt,
                system_prompt=self.system_prompt,
            )
        else:
            # Fall back to sync method
            critique_response = self.model.generate(
                prompt=critique_prompt,
                system_prompt=self.system_prompt,
            )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)

        # Determine if improvement is needed based on critique content
        needs_improvement = self._needs_improvement(critique_response)

        logger.debug("PromptCritic: Async critique completed")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.7,  # Default confidence for prompt-based critic
            "metadata": {
                "criteria": self.criteria,
                "system_prompt": self.system_prompt,
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text based on prompt-based critique.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the improvement fails.
        """
        # Use the enhanced method with validation context from thought
        validation_context = create_validation_context(getattr(thought, "validation_results", None))
        return self.improve_with_validation_context(thought, validation_context)

    def improve_with_validation_context(
        self, thought: Thought, validation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Improve text with validation context awareness.

        Args:
            thought: The Thought container with the text to improve and critique.
            validation_context: Optional validation context for constraint awareness.

        Returns:
            The improved text that prioritizes validation constraints.

        Raises:
            ImproverError: If the improvement fails.
        """
        start_time = time.time()

        with critic_context(
            critic_name="PromptCritic",
            operation="improve",
            message_prefix="Failed to improve text with prompt-based critic",
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
                        critique = feedback.feedback
                        break

            # If no critique available, generate one
            if not critique:
                logger.debug("No critique found in thought, generating new critique")
                critique_result = self._perform_critique(thought)
                critique = critique_result["message"]

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Parse critique to extract suggestions for filtering
            _, suggestions = self._parse_critique(critique)

            # Create improvement prompt with validation awareness
            if validation_context:
                # Use enhanced prompt with validation awareness
                improve_prompt = self._create_enhanced_improvement_prompt(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                    validation_context=validation_context,
                    critic_suggestions=suggestions,
                )
            else:
                # Use original prompt template
                improve_prompt = self.improve_prompt_template.format(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                )

            # Store the actual prompt for logging/debugging
            self.last_improvement_prompt = improve_prompt

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt=self.system_prompt,
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"PromptCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _parse_critique(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse critique text to extract issues and suggestions.

        Args:
            critique: The critique text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        # Simple parsing logic
        in_issues = False
        in_suggestions = False

        for line in critique.split("\n"):
            line = line.strip()
            if line.lower().startswith("issues:"):
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith("overall assessment:"):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # If no structured format found, extract from general content
        if not issues and not suggestions:
            critique_lower = critique.lower()
            if any(word in critique_lower for word in ["issue", "problem", "error", "unclear"]):
                issues.append("General issues identified in critique")
            if any(word in critique_lower for word in ["improve", "suggest", "consider", "should"]):
                suggestions.append("See critique for improvement suggestions")

        return issues, suggestions

    def _needs_improvement(self, critique: str) -> bool:
        """Determine if text needs improvement based on critique content.

        Args:
            critique: The critique text to analyze.

        Returns:
            True if improvement is needed, False otherwise.
        """
        # Simple heuristic based on common phrases in critiques
        no_improvement_phrases = [
            "no issues",
            "looks good",
            "well written",
            "excellent",
            "great job",
            "perfect",
            "no improvement needed",
            "already excellent",
            "no changes needed",
            "well-structured",
            "clear and concise",
            "high quality",
        ]

        improvement_phrases = [
            "could be improved",
            "needs improvement",
            "issues",
            "problems",
            "unclear",
            "confusing",
            "missing",
            "incorrect",
            "should be",
            "consider",
            "suggest",
            "recommend",
            "enhance",
            "revise",
        ]

        critique_lower = critique.lower()

        # Check for explicit "no improvement" indicators
        for phrase in no_improvement_phrases:
            if phrase in critique_lower:
                return False

        # Check for improvement indicators
        for phrase in improvement_phrases:
            if phrase in critique_lower:
                return True

        # Default to needing improvement if unclear
        return True
