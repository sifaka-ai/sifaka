"""Improvement prompt builder for PydanticAI chains.

This module handles building improvement prompts based on validation
and critic feedback.
"""

from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ImprovementPromptBuilder:
    """Builds improvement prompts based on feedback."""

    def create_improvement_prompt(self, thought: Thought) -> str:
        """Create an improvement prompt based on validation and critic feedback.

        Args:
            thought: The thought with feedback.

        Returns:
            Improvement prompt string.
        """
        # Build feedback text from validation and critics
        feedback_parts = []
        has_length_constraint = False

        # Add validation feedback with priority emphasis
        if hasattr(thought, "validation_results") and thought.validation_results:
            validation_issues = []
            for name, result in thought.validation_results.items():
                if not result.passed:
                    validation_issues.append(f"- {name}: {result.message}")
                    if result.suggestions:
                        validation_issues.append(f"  Suggestions: {', '.join(result.suggestions)}")

                    # Check if this is a length constraint
                    if "length" in name.lower() or "too long" in result.message.lower():
                        has_length_constraint = True

            if validation_issues:
                if has_length_constraint:
                    feedback_parts.append("CRITICAL VALIDATION REQUIREMENTS (MUST BE ADDRESSED):")
                else:
                    feedback_parts.append("Validation Issues:")
                feedback_parts.extend(validation_issues)

        # Add critic feedback with conditional filtering for length constraints
        if hasattr(thought, "critic_feedback") and thought.critic_feedback:
            critic_suggestions = []
            for feedback in thought.critic_feedback:
                # Include feedback if it needs improvement OR has useful suggestions
                if feedback.needs_improvement or (
                    feedback.suggestions and any(s.strip() for s in feedback.suggestions)
                ):
                    critic_line = f"- {feedback.critic_name}"
                    if feedback.feedback and feedback.feedback.strip():
                        critic_line += f": {feedback.feedback}"
                    critic_suggestions.append(critic_line)

                    if feedback.suggestions:
                        # Filter out generic suggestions
                        useful_suggestions = [
                            s
                            for s in feedback.suggestions
                            if s.strip() and s != "See critique for improvement suggestions"
                        ]

                        # If we have length constraints, filter out suggestions that would add content
                        if has_length_constraint:
                            useful_suggestions = self._filter_suggestions_for_length(
                                useful_suggestions
                            )

                        if useful_suggestions:
                            critic_suggestions.append(
                                f"  Suggestions: {', '.join(useful_suggestions)}"
                            )

            if critic_suggestions:
                if has_length_constraint:
                    feedback_parts.append(
                        "\nSecondary Feedback (only if compatible with length requirements):"
                    )
                else:
                    feedback_parts.append("\nCritic Feedback:")
                feedback_parts.extend(critic_suggestions)

        combined_feedback = (
            "\n".join(feedback_parts) if feedback_parts else "No specific feedback available."
        )

        # Create improvement prompt with emphasis on validation constraints
        if has_length_constraint:
            return self._create_length_constrained_prompt(thought, combined_feedback)
        else:
            return self._create_standard_prompt(thought, combined_feedback)

    def _filter_suggestions_for_length(self, suggestions: list[str]) -> list[str]:
        """Filter out suggestions that would add content when length is constrained.

        Args:
            suggestions: List of suggestions to filter.

        Returns:
            Filtered list of suggestions.
        """
        filtered_suggestions = []
        for suggestion in suggestions:
            # Skip suggestions that clearly ask for more content
            if not any(
                phrase in suggestion.lower()
                for phrase in [
                    "provide more",
                    "include more",
                    "add more",
                    "incorporate",
                    "enhance",
                    "expand",
                    "elaborate",
                    "examples",
                    "case studies",
                ]
            ):
                filtered_suggestions.append(suggestion)
        return filtered_suggestions

    def _create_length_constrained_prompt(self, thought: Thought, combined_feedback: str) -> str:
        """Create improvement prompt with length constraints emphasized.

        Args:
            thought: The thought to improve.
            combined_feedback: The combined feedback text.

        Returns:
            Length-constrained improvement prompt.
        """
        return (
            "Please improve the following text based on the feedback provided.\n\n"
            "⚠️  IMPORTANT: The text MUST meet the length requirements. This is a hard constraint "
            "that takes priority over all other suggestions. Focus on reducing content while "
            "maintaining the core message.\n\n"
            f"Original task: {thought.prompt}\n\n"
            f"Current text:\n{thought.text}\n\n"
            f"Feedback:\n{combined_feedback}\n\n"
            "Please provide an improved version that FIRST addresses the validation requirements "
            "(especially length constraints), then incorporates other feedback only if it doesn't "
            "conflict with the validation requirements. You may use your tools if needed.\n\n"
            "Improved text:"
        )

    def _create_standard_prompt(self, thought: Thought, combined_feedback: str) -> str:
        """Create standard improvement prompt.

        Args:
            thought: The thought to improve.
            combined_feedback: The combined feedback text.

        Returns:
            Standard improvement prompt.
        """
        return (
            "Please improve the following text based on the feedback provided.\n\n"
            f"Original task: {thought.prompt}\n\n"
            f"Current text:\n{thought.text}\n\n"
            f"Feedback:\n{combined_feedback}\n\n"
            "Please provide an improved version that addresses the issues identified "
            "in the feedback while maintaining the core message and staying true to "
            "the original task. You may use your tools if needed to gather additional "
            "information or validate your improvements.\n\n"
            "Improved text:"
        )
