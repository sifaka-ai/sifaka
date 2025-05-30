"""Validation-aware mixin for critics.

This module provides a mixin that enables critics to handle validation constraints
in their improvement prompts, ensuring that validation requirements are prioritized
over conflicting critic suggestions.
"""

from typing import Any, Dict, List, Optional

from sifaka.validators.validation_context import ValidationContext


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
            validation_context: Validation context from ValidationContext.extract_constraints().

        Returns:
            Dictionary with formatted strings for prompt template substitution.
        """
        if not validation_context:
            return {
                "validation_priority_notice": "",
                "validation_feedback": "",
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
        validation_feedback = self._format_validation_issues(
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
        """Filter suggestions that conflict with validation constraints.

        Args:
            suggestions: List of critic suggestions to filter.
            validation_context: Validation context from ValidationContext.extract_constraints().

        Returns:
            Filtered list of suggestions that don't conflict with constraints.
        """
        return ValidationContext.filter_conflicting_suggestions(suggestions, validation_context)

    def _create_priority_instructions(self, validation_context: Optional[Dict[str, Any]]) -> str:
        """Create priority instructions based on validation constraints.

        Args:
            validation_context: Validation context from ValidationContext.extract_constraints().

        Returns:
            Priority instructions string for improvement prompts.
        """
        feedback_categories = ValidationContext.categorize_feedback(validation_context)
        return feedback_categories["priority_instructions"]

    def _format_validation_issues(
        self, validation_context: Dict[str, Any], feedback_categories: Dict[str, str]
    ) -> str:
        """Format validation issues for inclusion in improvement prompts.

        Args:
            validation_context: Validation context with constraint information.
            feedback_categories: Categorized feedback templates.

        Returns:
            Formatted validation issues string.
        """
        if not validation_context:
            return ""

        header = feedback_categories["validation_header"]

        if validation_context.get("type") == "length":
            return f"{header}\n- {validation_context['validator_name']}: {validation_context['message']}\n"

        elif validation_context.get("type") == "general":
            issues = []
            for validator, message in zip(
                validation_context.get("failed_validators", []),
                validation_context.get("messages", []),
                strict=False,
            ):
                issues.append(f"- {validator}: {message}")
            return f"{header}\n" + "\n".join(issues) + "\n"

        return f"{header}\n- Validation requirements must be met\n"

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
            prompt: Original task prompt.
            text: Current text to improve.
            critique: Critic feedback/critique.
            context: Retrieved context (optional).
            validation_context: Validation context for constraint awareness.
            critic_suggestions: List of critic suggestions to potentially filter.

        Returns:
            Enhanced improvement prompt with validation prioritization.
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
            original_critique: Original critique text.
            filtered_suggestions: Filtered list of suggestions.

        Returns:
            Updated critique text with filtered suggestions.
        """
        if not filtered_suggestions:
            return "The text meets the basic requirements. Focus on validation constraints."

        # Simple approach: append filtered suggestions to critique
        suggestions_text = "\n".join(f"- {suggestion}" for suggestion in filtered_suggestions)

        # Try to preserve the original critique structure while updating suggestions
        if "suggestions:" in original_critique.lower():
            # Replace suggestions section
            lines = original_critique.split("\n")
            new_lines = []
            in_suggestions = False

            for line in lines:
                if "suggestions:" in line.lower():
                    new_lines.append(line)
                    new_lines.append(suggestions_text)
                    in_suggestions = True
                elif in_suggestions and line.strip().startswith("-"):
                    # Skip original suggestions
                    continue
                elif in_suggestions and not line.strip():
                    # End of suggestions section
                    in_suggestions = False
                    new_lines.append(line)
                elif not in_suggestions:
                    new_lines.append(line)

            return "\n".join(new_lines)
        else:
            # Append filtered suggestions to original critique
            return f"{original_critique}\n\nFiltered Suggestions:\n{suggestions_text}"
