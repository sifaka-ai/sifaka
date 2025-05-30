"""Self-Refine critic for Sifaka.

This module implements the Self-Refine approach for critics, which enables language
models to iteratively critique and revise their own outputs without requiring
external feedback.

Based on "Self-Refine: Iterative Refinement with Self-Feedback":
https://arxiv.org/abs/2303.17651

@misc{madaan2023selfrefineiterativerefinementselffeedback,
      title={Self-Refine: Iterative Refinement with Self-Feedback},
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Shashank Gupta and Bodhisattwa Prasad Majumder and Katherine Hermann and Sean Welleck and Amir Yazdanbakhsh and Peter Clark},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2303.17651},
}

The SelfRefineCritic implements the core Self-Refine algorithm:
1. Iterative refinement through self-feedback
2. Multi-round critique and revision cycles
3. Self-generated improvement suggestions
4. Convergence detection for stopping criteria

Note: This implementation follows the original Self-Refine paper closely,
using a simple FEEDBACK → REFINE → FEEDBACK loop without additional
learning mechanisms that were not part of the original research.
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


class SelfRefineCritic(BaseCritic, ValidationAwareMixin):
    """Critic that implements iterative self-refinement with validation awareness.

    This critic uses the Self-Refine approach to iteratively improve text through
    self-critique and revision. It uses the same language model to critique its
    own output and then revise it based on that critique.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting critic suggestions.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 3,
        improvement_criteria: Optional[List[str]] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            max_iterations: Maximum number of refinement iterations.
            improvement_criteria: Specific criteria to focus on during improvement.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        super().__init__(model=model, model_name=model_name, **model_kwargs)

        self.max_iterations = max_iterations
        self.improvement_criteria = improvement_criteria or [
            "clarity",
            "accuracy",
            "completeness",
            "coherence",
        ]

        # Set up prompt templates
        criteria_text = ", ".join(self.improvement_criteria)

        self.critique_prompt_template = critique_prompt_template or (
            f"Please critique the following text focusing on {criteria_text}.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please provide a detailed critique focusing on:\n"
            "1. How well does the text address the original task?\n"
            "2. Are there any factual errors or inconsistencies?\n"
            "3. Is the text clear and well-structured?\n"
            "4. What specific improvements could be made?\n"
            "5. How well does the text use information from the retrieved context (if available)?\n\n"
            "Format your response as:\n"
            "Issues:\n- [List specific issues here]\n\n"
            "Suggestions:\n- [List specific suggestions here]\n\n"
            "Overall Assessment: [Brief assessment]\n\n"
            "If the text is already excellent and needs no improvement, please state that clearly."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Please improve the following text based on the critique provided.\n\n"
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

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Self-Refine approach.

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
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message="You are an expert critic providing detailed, constructive feedback.",
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)

        # Determine if improvement is needed based on critique content
        needs_improvement = self._needs_improvement(critique_response)

        logger.debug("SelfRefineCritic: Critique completed")

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": 0.8,  # Default confidence for Self-Refine
            "metadata": {
                "max_iterations": self.max_iterations,
                "improvement_criteria": self.improvement_criteria,
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text using iterative Self-Refine approach.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text after iterative refinement.

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

            # Prepare context once for all iterations (using mixin)
            context = self._prepare_context(thought)

            # Iterative refinement process following original Self-Refine algorithm
            for iteration in range(self.max_iterations):
                logger.debug(
                    f"SelfRefineCritic: Starting iteration {iteration + 1}/{self.max_iterations}"
                )

                # FEEDBACK: Generate critique for current text
                critique_prompt = self.critique_prompt_template.format(
                    prompt=thought.prompt,
                    text=current_text,
                    context=context,
                )

                critique = self.model.generate(
                    prompt=critique_prompt,
                    system_prompt="You are an expert critic providing detailed, constructive feedback.",
                )

                # Check if improvement is needed (stopping criteria)
                if not self._needs_improvement(critique):
                    logger.debug(
                        f"SelfRefineCritic: Stopping early at iteration {iteration + 1} - no improvement needed"
                    )
                    break

                # Parse critique to extract suggestions for filtering
                _, suggestions = self._parse_critique(critique)

                # REFINE: Generate improved text using validation-aware prompt
                if validation_context:
                    # Use enhanced prompt with validation awareness
                    improve_prompt = self._create_enhanced_improvement_prompt(
                        prompt=thought.prompt,
                        text=current_text,
                        critique=critique,
                        context=context,
                        validation_context=validation_context,
                        critic_suggestions=suggestions,
                    )
                else:
                    # Use original prompt template
                    improve_prompt = self.improve_prompt_template.format(
                        prompt=thought.prompt,
                        text=current_text,
                        critique=critique,
                        context=context,
                    )

                # Store the actual prompt for logging/debugging
                self.last_improvement_prompt = improve_prompt

                improved_text = self.model.generate(
                    prompt=improve_prompt,
                    system_prompt="You are an expert editor improving text based on critique.",
                )

                # Update current text for next iteration
                current_text = improved_text.strip()

                logger.debug(f"SelfRefineCritic: Completed iteration {iteration + 1}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"SelfRefineCritic: Refinement completed in {processing_time:.2f}ms "
                f"after {iteration + 1} iterations"
            )

            return current_text

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
