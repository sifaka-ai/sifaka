"""Self-Refine critic for Sifaka.

This module implements the Self-Refine approach for critics, which enables language
models to iteratively critique and revise their own outputs without requiring
external feedback. The critic uses the same language model to generate critiques
and revisions in multiple rounds.

Based on Self-Refine: https://arxiv.org/abs/2303.17651

The SelfRefineCritic implements a multi-round refinement process where the model
critiques its own output and then revises it based on that critique, leading to
progressively improved results.
"""

import time
from typing import Any, Dict, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class SelfRefineCritic(ContextAwareMixin):
    """Critic that implements iterative self-refinement.

    This critic uses the Self-Refine approach to iteratively improve text through
    self-critique and revision. It uses the same language model to critique its
    own output and then revise it based on that critique.

    Attributes:
        model: The language model to use for critique and improvement.
        max_iterations: Maximum number of refinement iterations.
        stopping_threshold: Threshold for stopping refinement early.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 3,
        stopping_threshold: float = 0.8,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            max_iterations: Maximum number of refinement iterations.
            stopping_threshold: Quality threshold for stopping refinement early.
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

        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Please critique the following text and identify areas for improvement.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please provide a detailed critique focusing on:\n"
            "1. How well does the text address the original task?\n"
            "2. Are there any factual errors or inconsistencies?\n"
            "3. Is the text clear and well-structured?\n"
            "4. What specific improvements could be made?\n"
            "5. How well does the text use information from the retrieved context (if available)?\n\n"
            "If the text is already excellent and needs no improvement, please state that clearly.\n"
            "Otherwise, provide specific, actionable feedback for improvement."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version that addresses the issues identified "
            "in the critique while maintaining the core message and staying true to "
            "the original task. Better incorporate relevant information from the context if available."
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text using Self-Refine approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRefineCritic",
            operation="critique",
            message_prefix="Failed to critique text with Self-Refine",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "message": "No text available for critique",
                    "issues": ["Text is empty or None"],
                    "suggestions": ["Provide text to critique"],
                    "iteration": 0,
                }

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
                system_prompt="You are an expert critic providing detailed, constructive feedback.",
            )

            # Determine if improvement is needed based on critique content
            needs_improvement = self._needs_improvement(critique_response)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfRefineCritic: Critique completed in {processing_time:.2f}ms")

            return {
                "needs_improvement": needs_improvement,
                "message": critique_response,
                "critique": critique_response,
                "iteration": 1,
                "processing_time_ms": processing_time,
            }

    def improve(self, thought: Thought) -> str:
        """Improve text using iterative Self-Refine approach.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text after iterative refinement.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRefineCritic",
            operation="improve",
            message_prefix="Failed to improve text with Self-Refine",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="SelfRefineCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            current_text = thought.text
            iteration_history = []

            # Prepare context once for all iterations (using mixin)
            context = self._prepare_context(thought)

            # Iterative refinement process
            for iteration in range(self.max_iterations):
                logger.debug(
                    f"SelfRefineCritic: Starting iteration {iteration + 1}/{self.max_iterations}"
                )

                # Generate critique for current text with context
                critique_prompt = self.critique_prompt_template.format(
                    prompt=thought.prompt,
                    text=current_text,
                    context=context,
                )

                critique = self.model.generate(
                    prompt=critique_prompt,
                    system_prompt="You are an expert critic providing detailed, constructive feedback.",
                )

                # Check if improvement is needed
                if not self._needs_improvement(critique):
                    logger.debug(
                        f"SelfRefineCritic: Stopping early at iteration {iteration + 1} - no improvement needed"
                    )
                    break

                # Generate improved text with context
                improve_prompt = self.improve_prompt_template.format(
                    prompt=thought.prompt,
                    text=current_text,
                    critique=critique,
                    context=context,
                )

                improved_text = self.model.generate(
                    prompt=improve_prompt,
                    system_prompt="You are an expert editor improving text based on critique.",
                )

                # Store iteration history
                iteration_history.append(
                    {
                        "iteration": iteration + 1,
                        "critique": critique,
                        "text": current_text,
                        "improved_text": improved_text.strip(),
                    }
                )

                # Update current text for next iteration
                current_text = improved_text.strip()

                logger.debug(f"SelfRefineCritic: Completed iteration {iteration + 1}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"SelfRefineCritic: Refinement completed in {processing_time:.2f}ms "
                f"after {len(iteration_history)} iterations"
            )

            return current_text

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


def create_self_refine_critic(
    model: Optional[Model] = None,
    model_name: Optional[str] = None,
    max_iterations: int = 3,
    stopping_threshold: float = 0.8,
    **model_kwargs: Any,
) -> SelfRefineCritic:
    """Create a Self-Refine critic.

    Args:
        model: The language model to use for critique and improvement.
        model_name: The name of the model to use if model is not provided.
        max_iterations: Maximum number of refinement iterations.
        stopping_threshold: Quality threshold for stopping refinement early.
        **model_kwargs: Additional keyword arguments for model creation.

    Returns:
        A SelfRefineCritic instance.
    """
    return SelfRefineCritic(
        model=model,
        model_name=model_name,
        max_iterations=max_iterations,
        stopping_threshold=stopping_threshold,
        **model_kwargs,
    )
