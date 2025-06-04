"""Validation-aware mixin for critics.

This module provides a mixin that enables critics to handle validation constraints
in their improvement prompts, ensuring that validation requirements are prioritized
over conflicting critic suggestions.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import SifakaThought, ValidationContext


class ValidationAwareMixin:
    """Mixin for critics to handle validation constraints in improvement prompts.

    This mixin provides methods for:
    - Formatting validation context for prompt templates
    - Filtering suggestions that conflict with validation constraints
    - Creating priority instructions based on validation constraints
    - Generating validation-aware improvement prompts
    """

    def _format_validation_context(
        self, validation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Format validation context for prompt templates.

        Args:
            validation_context: Validation context from ValidationContext.extract_constraints()

        Returns:
            Dictionary with formatted strings for prompt template substitution
        """
        if not validation_context:
            return {
                "validation_priority_notice": "",
                "validation_feedback": "",
                "critic_feedback_header": "Critic Feedback:",
                "priority_instructions": (
                    "addresses the issues identified in the feedback while maintaining the core message "
                    "and staying true to the original task"
                ),
            }

        # Get categorized feedback templates
        feedback_categories = ValidationContext.categorize_feedback(validation_context)

        # Create priority notice
        priority_notice = ValidationContext.create_validation_priority_notice(validation_context)

        # Format validation feedback
        validation_feedback = ValidationContext.format_validation_issues(
            validation_context, feedback_categories
        )

        return {
            "validation_priority_notice": priority_notice,
            "validation_feedback": validation_feedback,
            "critic_feedback_header": feedback_categories["critic_header"],
            "priority_instructions": feedback_categories["priority_instructions"],
        }

    def _filter_suggestions_for_constraints(
        self, suggestions: List[str], validation_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Filter critic suggestions that conflict with validation constraints.

        Args:
            suggestions: List of critic suggestions
            validation_context: Validation context from ValidationContext.extract_constraints()

        Returns:
            Filtered list of suggestions that don't conflict with constraints
        """
        return ValidationContext.filter_conflicting_suggestions(suggestions, validation_context)

    def _create_priority_instructions(self, validation_context: Optional[Dict[str, Any]]) -> str:
        """Create priority instructions based on validation constraints.

        Args:
            validation_context: Validation context from ValidationContext.extract_constraints()

        Returns:
            Priority instructions string for improvement prompts
        """
        feedback_categories = ValidationContext.categorize_feedback(validation_context)
        return feedback_categories["priority_instructions"]

    def _create_enhanced_improvement_prompt(
        self,
        prompt: str,
        text: str,
        critique: str,
        context: str = "",
        validation_context: Optional[Dict[str, Any]] = None,
        critic_suggestions: Optional[List[str]] = None,
    ) -> str:
        """Create an enhanced improvement prompt with validation awareness.

        Args:
            prompt: Original task prompt
            text: Current text to improve
            critique: Critic feedback/critique
            context: Retrieved context (optional)
            validation_context: Validation context for constraint awareness
            critic_suggestions: List of critic suggestions to potentially filter

        Returns:
            Enhanced improvement prompt with validation prioritization
        """
        # Format validation context
        validation_format = self._format_validation_context(validation_context)

        # Filter critic suggestions if provided
        if critic_suggestions and validation_context:
            filtered_suggestions = self._filter_suggestions_for_constraints(
                critic_suggestions, validation_context
            )
            # Update critique to include only filtered suggestions
            if filtered_suggestions != critic_suggestions:
                critique = self._update_critique_with_filtered_suggestions(
                    critique, filtered_suggestions
                )

        # Format critic feedback section
        critic_feedback_section = ""
        if critique.strip():
            critic_header = validation_format.get("critic_feedback_header", "Critic Feedback:")
            critic_feedback_section = f"{critic_header}\n{critique}\n\n"

        # Create the enhanced prompt
        enhanced_prompt = (
            f"Please improve the following text based on the feedback provided.\n\n"
            f"{validation_format['validation_priority_notice']}"
            f"Original task: {prompt}\n\n"
            f"Current text:\n{text}\n\n"
        )

        if context.strip():
            enhanced_prompt += f"Retrieved context:\n{context}\n\n"

        enhanced_prompt += validation_format["validation_feedback"]
        enhanced_prompt += critic_feedback_section

        enhanced_prompt += (
            f"Please provide an improved version that {validation_format['priority_instructions']}.\n\n"
            f"Improved text:"
        )

        return enhanced_prompt

    def _update_critique_with_filtered_suggestions(
        self, original_critique: str, filtered_suggestions: List[str]
    ) -> str:
        """Update critique text to reflect filtered suggestions.

        Args:
            original_critique: Original critique text
            filtered_suggestions: Filtered list of suggestions

        Returns:
            Updated critique text with filtered suggestions
        """
        if not filtered_suggestions:
            return "The text meets the basic requirements. Focus on validation constraints."

        # Create new critique with filtered suggestions
        suggestions_text = "\n".join(f"- {suggestion}" for suggestion in filtered_suggestions)

        # Try to preserve the main critique message but replace suggestions
        lines = original_critique.split("\n")
        critique_lines = []
        in_suggestions = False

        for line in lines:
            line_lower = line.lower().strip()
            if any(
                keyword in line_lower for keyword in ["suggest", "recommend", "should", "could"]
            ):
                in_suggestions = True
                break
            critique_lines.append(line)

        if critique_lines:
            main_critique = "\n".join(critique_lines).strip()
            return f"{main_critique}\n\nFiltered suggestions:\n{suggestions_text}"
        else:
            return f"Suggestions:\n{suggestions_text}"

    def _get_validation_aware_context(self, thought: SifakaThought) -> str:
        """Get validation-aware context string for prompts.

        Args:
            thought: The SifakaThought to extract context from

        Returns:
            Formatted validation-aware context string
        """
        # Get validation context
        validation_context = ValidationContext.extract_constraints(thought)

        if not validation_context:
            return ""

        # Format for prompt inclusion
        validation_format = self._format_validation_context(validation_context)

        context_parts = []

        if validation_format["validation_priority_notice"]:
            context_parts.append(validation_format["validation_priority_notice"].strip())

        if validation_format["validation_feedback"]:
            context_parts.append(validation_format["validation_feedback"].strip())

        return "\n".join(context_parts) if context_parts else ""

    def _should_prioritize_validation(self, thought: SifakaThought) -> bool:
        """Check if validation should be prioritized over critic feedback.

        Args:
            thought: The SifakaThought to check

        Returns:
            True if validation should be prioritized, False otherwise
        """
        validation_context = ValidationContext.extract_constraints(thought)
        return ValidationContext.has_critical_constraints(validation_context)
