"""Prompt building for PydanticAI chains.

This module handles building prompts for different phases of chain execution,
including initial generation and improvement iterations.
"""

from typing import List, Optional

from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Builds prompts for different chain execution phases."""

    def build_model_prompt(self, thought: Thought, original_prompt: str) -> str:
        """Build the final prompt that will be sent to the model with RAG-optimized ordering.

        Order: system_prompt, prompt-retrieved content, prompt, validation results, critic results, original content

        Args:
            thought: The current thought state.
            original_prompt: The original user prompt.

        Returns:
            The complete model prompt.
        """
        prompt_parts = []

        # 1. Add system prompt if available
        if thought.system_prompt:
            prompt_parts.append(f"System: {thought.system_prompt}")

        # 2. Add prompt-retrieved content first (context before question for better RAG)
        if thought.pre_generation_context:
            context_texts = [doc.text for doc in thought.pre_generation_context]
            context_str = "\n\n".join(context_texts)
            prompt_parts.append(f"Context:\n{context_str}")

        # 3. Add the user prompt (question with context fresh in mind)
        prompt_parts.append(f"User: {original_prompt}")

        # 4. Add validation results FIRST (highest priority if any failures)
        if thought.validation_results:
            failed_validations = [
                result for result in thought.validation_results.values() if not result.passed
            ]
            if failed_validations:
                validation_feedback = []
                for result in failed_validations:
                    validator_name = result.validator_name or "Validator"
                    issues = ", ".join(result.issues or [])
                    suggestions = ", ".join(result.suggestions or [])
                    feedback_line = f"- {validator_name}: {issues}"
                    if suggestions:
                        feedback_line += f" (Suggestions: {suggestions})"
                    validation_feedback.append(feedback_line)
                prompt_parts.append("Validation Issues:\n" + "\n".join(validation_feedback))

        # 5. Add critic results (detailed feedback from previous iterations)
        if hasattr(thought, "critic_feedback") and thought.critic_feedback:
            critic_feedback = []
            for feedback in thought.critic_feedback[-3:]:  # Last 3 feedbacks
                # Prioritize full detailed feedback over parsed suggestions
                if (
                    hasattr(feedback, "feedback")
                    and feedback.feedback
                    and feedback.feedback.strip()
                ):
                    critic_feedback.append(f"- {feedback.critic_name}: {feedback.feedback}")
                elif hasattr(feedback, "suggestions") and feedback.suggestions:
                    # Filter out generic suggestions
                    useful_suggestions = [
                        s
                        for s in feedback.suggestions
                        if s.strip() and s != "See critique for improvement suggestions"
                    ]
                    if useful_suggestions:
                        suggestions_text = ", ".join(useful_suggestions)
                        critic_feedback.append(f"- {feedback.critic_name}: {suggestions_text}")
            if critic_feedback:
                prompt_parts.append("Critic Feedback:\n" + "\n".join(critic_feedback))

        # 6. Add original content (previous attempt) if this is an improvement iteration
        if thought.iteration > 1 and thought.text:
            prompt_parts.append(f"Previous Attempt:\n{thought.text}")

        return "\n\n".join(prompt_parts)

    def build_improvement_prompt(self, thought: Thought, original_prompt: str) -> str:
        """Build an improvement prompt with RAG-optimized ordering.

        Order: system_prompt, prompt-retrieved content, prompt, validation results, critic results, original content

        Args:
            thought: The current thought state with feedback.
            original_prompt: The original user prompt.

        Returns:
            The improvement prompt.
        """
        prompt_parts = []

        # 1. Add system prompt if available
        if thought.system_prompt:
            prompt_parts.append(f"System: {thought.system_prompt}")

        # 2. Add prompt-retrieved content first (context before question for better RAG)
        if thought.pre_generation_context:
            context_texts = [doc.text for doc in thought.pre_generation_context]
            context_str = "\n\n".join(context_texts)
            prompt_parts.append(f"Context:\n{context_str}")

        # 3. Add the user prompt (question with context fresh in mind)
        prompt_parts.append(f"Original Request: {original_prompt}")

        # 4. Add validation feedback FIRST (highest priority for model to address)
        validation_feedback_added = False
        if thought.validation_results:
            failed_validations = [
                result for result in thought.validation_results.values() if not result.passed
            ]
            if failed_validations:
                validation_feedback = []
                for result in failed_validations:
                    validator_name = result.validator_name or "Validator"
                    issues = ", ".join(result.issues or [])
                    suggestions = ", ".join(result.suggestions or [])
                    feedback_line = f"- {validator_name}: {issues}"
                    if suggestions:
                        feedback_line += f" (Suggestions: {suggestions})"
                    validation_feedback.append(feedback_line)
                prompt_parts.append("Validation Issues:\n" + "\n".join(validation_feedback))
                validation_feedback_added = True

        # 5. Add critic feedback (detailed feedback with full suggestions)
        if thought.critic_feedback:
            critic_suggestions = []
            for feedback in thought.critic_feedback:
                # Always include critic feedback if it exists, regardless of needs_improvement flag
                if feedback.feedback and feedback.feedback.strip():
                    # Use the full detailed feedback text, not just parsed suggestions
                    critic_line = f"- {feedback.critic_name}: {feedback.feedback}"
                    critic_suggestions.append(critic_line)
                elif feedback.suggestions and any(s.strip() for s in feedback.suggestions):
                    # Fallback to suggestions if no detailed feedback
                    useful_suggestions = [
                        s
                        for s in feedback.suggestions
                        if s.strip() and s != "See critique for improvement suggestions"
                    ]
                    if useful_suggestions:
                        suggestions_text = ", ".join(useful_suggestions)
                        critic_line = f"- {feedback.critic_name}: {suggestions_text}"
                        critic_suggestions.append(critic_line)

            if critic_suggestions:
                prompt_parts.append("Improvement Suggestions:\n" + "\n".join(critic_suggestions))

        # 6. Add original content (previous attempt for reference)
        if thought.text:
            prompt_parts.append(f"Previous Attempt:\n{thought.text}")

        # 7. Add improvement instruction with priority guidance
        if validation_feedback_added:
            prompt_parts.append(
                "Please provide an improved response that addresses the validation issues and incorporates the improvement suggestions."
            )
        else:
            prompt_parts.append(
                "Please provide an improved response that incorporates the improvement suggestions."
            )

        return "\n\n".join(prompt_parts)

    def build_improvement_prompt_with_parent_feedback(
        self, thought: Thought, original_prompt: str, parent_critic_feedback: Optional[List] = None
    ) -> str:
        """Build an improvement prompt with parent critic feedback.

        This method is used when the current thought doesn't have critic feedback yet
        (because next_iteration() clears it), but we need to include the parent's
        critic feedback in the improvement prompt.

        Args:
            thought: The current thought state (without critic feedback).
            original_prompt: The original user prompt.
            parent_critic_feedback: Critic feedback from the parent iteration.

        Returns:
            The improvement prompt with parent feedback included.
        """
        prompt_parts = []

        # 1. Add system prompt if available
        if thought.system_prompt:
            prompt_parts.append(f"System: {thought.system_prompt}")

        # 2. Add prompt-retrieved content first (context before question for better RAG)
        if thought.pre_generation_context:
            context_texts = [doc.text for doc in thought.pre_generation_context]
            context_str = "\n\n".join(context_texts)
            prompt_parts.append(f"Context:\n{context_str}")

        # 3. Add the user prompt (question with context fresh in mind)
        prompt_parts.append(f"Original Request: {original_prompt}")

        # 4. Add validation feedback FIRST (highest priority for model to address)
        validation_feedback_added = False
        if thought.validation_results:
            failed_validations = [
                result for result in thought.validation_results.values() if not result.passed
            ]
            if failed_validations:
                validation_feedback = []
                for result in failed_validations:
                    validator_name = result.validator_name or "Validator"
                    issues = ", ".join(result.issues or [])
                    suggestions = ", ".join(result.suggestions or [])
                    feedback_line = f"- {validator_name}: {issues}"
                    if suggestions:
                        feedback_line += f" (Suggestions: {suggestions})"
                    validation_feedback.append(feedback_line)
                prompt_parts.append("Validation Issues:\n" + "\n".join(validation_feedback))
                validation_feedback_added = True

        # 5. Add parent critic feedback (detailed feedback from previous iteration)
        if parent_critic_feedback:
            critic_suggestions = []
            for feedback in parent_critic_feedback:
                # Always include critic feedback if it exists
                if (
                    hasattr(feedback, "feedback")
                    and feedback.feedback
                    and feedback.feedback.strip()
                ):
                    # Use the full detailed feedback text, not just parsed suggestions
                    critic_line = f"- {feedback.critic_name}: {feedback.feedback}"
                    critic_suggestions.append(critic_line)
                elif (
                    hasattr(feedback, "suggestions")
                    and feedback.suggestions
                    and any(s.strip() for s in feedback.suggestions)
                ):
                    # Fallback to suggestions if no detailed feedback
                    useful_suggestions = [
                        s
                        for s in feedback.suggestions
                        if s.strip() and s != "See critique for improvement suggestions"
                    ]
                    if useful_suggestions:
                        suggestions_text = ", ".join(useful_suggestions)
                        critic_line = f"- {feedback.critic_name}: {suggestions_text}"
                        critic_suggestions.append(critic_line)

            if critic_suggestions:
                prompt_parts.append("Improvement Suggestions:\n" + "\n".join(critic_suggestions))

        # 6. Add original content (previous attempt for reference)
        if thought.text:
            prompt_parts.append(f"Previous Attempt:\n{thought.text}")

        # 7. Add improvement instruction with priority guidance
        if validation_feedback_added:
            prompt_parts.append(
                "Please provide an improved response that addresses the validation issues and incorporates the improvement suggestions."
            )
        else:
            prompt_parts.append(
                "Please provide an improved response that incorporates the improvement suggestions."
            )

        return "\n\n".join(prompt_parts)
