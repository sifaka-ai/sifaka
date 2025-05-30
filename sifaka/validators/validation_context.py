"""Validation context management for Sifaka chains.

This module provides utilities for managing validation context in improvement operations,
enabling critics to be aware of validation constraints and prioritize them appropriately.
"""

from typing import Any, Dict, List, Optional

from sifaka.core.thought import ValidationResult


class ValidationContext:
    """Helper class for managing validation context in improvement operations."""

    @staticmethod
    def extract_constraints(
        validation_results: Dict[str, ValidationResult],
    ) -> Optional[Dict[str, Any]]:
        """Extract constraint information from validation results.

        Args:
            validation_results: Dictionary of validation results from validators.

        Returns:
            Dictionary containing constraint information, or None if no constraints found.
        """
        if not validation_results:
            return None

        # Check for length constraints first (most common)
        length_constraint = ValidationContext._detect_length_constraints(validation_results)
        if length_constraint:
            return length_constraint

        # Check for other constraint types
        other_constraints = ValidationContext._detect_other_constraints(validation_results)
        if other_constraints:
            return other_constraints

        return None

    @staticmethod
    def has_length_constraints(validation_results: Dict[str, ValidationResult]) -> bool:
        """Check if validation results contain length constraints.

        Args:
            validation_results: Dictionary of validation results from validators.

        Returns:
            True if length constraints are present, False otherwise.
        """
        if not validation_results:
            return False

        for name, result in validation_results.items():
            if not result.passed and ValidationContext._is_length_constraint(name, result):
                return True

        return False

    @staticmethod
    def filter_conflicting_suggestions(
        suggestions: List[str], constraints: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Filter critic suggestions that conflict with validation constraints.

        Args:
            suggestions: List of critic suggestions.
            constraints: Constraint information from extract_constraints().

        Returns:
            Filtered list of suggestions that don't conflict with constraints.
        """
        if not constraints or not suggestions:
            return suggestions

        if (
            constraints.get("type") == "length"
            and constraints.get("constraint") == "reduce_content"
        ):
            return ValidationContext._filter_content_expansion_suggestions(suggestions)

        # Add other constraint type filtering here as needed
        return suggestions

    @staticmethod
    def create_validation_priority_notice(constraints: Optional[Dict[str, Any]]) -> str:
        """Create priority notice for validation constraints.

        Args:
            constraints: Constraint information from extract_constraints().

        Returns:
            Priority notice string for inclusion in improvement prompts.
        """
        if not constraints:
            return ""

        if constraints.get("type") == "length":
            # Extract the actual length limit from the constraint message
            constraint_msg = constraints.get("message", "")
            max_chars = "the specified limit"
            if "maximum allowed:" in constraint_msg:
                try:
                    max_chars = constraint_msg.split("maximum allowed:")[1].strip().rstrip(")")
                except (IndexError, AttributeError):
                    max_chars = "the specified limit"

            return (
                f"🚨 CRITICAL: The text MUST be under {max_chars} characters. This is a HARD CONSTRAINT "
                "that takes ABSOLUTE PRIORITY over all other suggestions. You MUST aggressively cut content, "
                "remove details, eliminate examples, and focus ONLY on the most essential points. "
                f"Count characters carefully and ensure you stay UNDER {max_chars}. "
                "Content quality is secondary to meeting the length requirement. CUT RUTHLESSLY.\n\n"
            )

        return "⚠️  IMPORTANT: Please ensure the text meets all validation requirements.\n\n"

    @staticmethod
    def categorize_feedback(constraints: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Categorize feedback based on validation constraints.

        Args:
            constraints: Constraint information from extract_constraints().

        Returns:
            Dictionary with categorized feedback templates.
        """
        if constraints and constraints.get("type") == "length":
            return {
                "validation_header": "🚨 CRITICAL VALIDATION REQUIREMENTS (MUST BE ADDRESSED FIRST):",
                "critic_header": "Secondary Feedback (IGNORE if it conflicts with length requirements):",
                "priority_instructions": (
                    "IMMEDIATELY and AGGRESSIVELY addresses the length constraints by cutting content ruthlessly. "
                    "Ignore all other feedback that would add content or increase length. "
                    "Your PRIMARY GOAL is to meet the length requirement, even if it means sacrificing detail or quality"
                ),
            }

        return {
            "validation_header": "Validation Issues:",
            "critic_header": "Critic Feedback:",
            "priority_instructions": (
                "addresses the issues identified in the feedback while maintaining the core message "
                "and staying true to the original task"
            ),
        }

    # Private helper methods

    @staticmethod
    def _detect_length_constraints(
        validation_results: Dict[str, ValidationResult],
    ) -> Optional[Dict[str, Any]]:
        """Detect length constraints from validation results."""
        for name, result in validation_results.items():
            if not result.passed and ValidationContext._is_length_constraint(name, result):
                return {
                    "type": "length",
                    "constraint": (
                        "reduce_content"
                        if "too long" in result.message.lower()
                        else "expand_content"
                    ),
                    "validator_name": name,
                    "message": result.message,
                    "suggestions": result.suggestions or [],
                }
        return None

    @staticmethod
    def _detect_other_constraints(
        validation_results: Dict[str, ValidationResult],
    ) -> Optional[Dict[str, Any]]:
        """Detect other types of constraints from validation results."""
        failed_validators = [
            (name, result) for name, result in validation_results.items() if not result.passed
        ]

        if failed_validators:
            # For now, treat all other failed validations as general constraints
            return {
                "type": "general",
                "constraint": "meet_requirements",
                "failed_validators": [name for name, _ in failed_validators],
                "messages": [result.message for _, result in failed_validators],
                "suggestions": [
                    suggestion
                    for _, result in failed_validators
                    for suggestion in (result.suggestions or [])
                ],
            }

        return None

    @staticmethod
    def _is_length_constraint(validator_name: str, result: ValidationResult) -> bool:
        """Check if a validation result represents a length constraint."""
        return "length" in validator_name.lower() or any(
            phrase in result.message.lower()
            for phrase in ["too long", "too short", "characters", "words", "length"]
        )

    @staticmethod
    def _filter_content_expansion_suggestions(suggestions: List[str]) -> List[str]:
        """Filter suggestions that would expand content when length reduction is needed."""
        CONTENT_EXPANSION_PHRASES = [
            "provide more",
            "include more",
            "add more",
            "incorporate",
            "enhance",
            "expand",
            "elaborate",
            "examples",
            "case studies",
            "provide additional",
            "include additional",
            "add details",
            "give more",
            "offer more",
            "present more",
            "show more",
        ]

        filtered_suggestions = []
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if not any(phrase in suggestion_lower for phrase in CONTENT_EXPANSION_PHRASES):
                filtered_suggestions.append(suggestion)

        return filtered_suggestions


def create_validation_context(
    validation_results: Optional[Dict[str, ValidationResult]],
) -> Optional[Dict[str, Any]]:
    """Convenience function to create validation context from validation results.

    Args:
        validation_results: Dictionary of validation results from validators.

    Returns:
        Validation context dictionary, or None if no constraints found.
    """
    if not validation_results:
        return None

    return ValidationContext.extract_constraints(validation_results)
