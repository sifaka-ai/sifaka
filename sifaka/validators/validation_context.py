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

        # Get all failed validations - we don't need to categorize them
        failed_validations = [
            (name, result) for name, result in validation_results.items() if not result.passed
        ]

        if not failed_validations:
            return None

        # Return the raw validation failures - let the improvement prompt handle them intelligently
        return {
            "type": "validation_failures",
            "failed_validations": [
                {
                    "validator_name": name,
                    "message": result.message,
                    "suggestions": result.suggestions or [],
                    "issues": result.issues or [],
                    "score": getattr(result, "score", 0.0),
                }
                for name, result in failed_validations
            ],
            "total_failures": len(failed_validations),
        }

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
        if not constraints or constraints.get("type") != "validation_failures":
            return ""

        failed_validations = constraints.get("failed_validations", [])
        if not failed_validations:
            return ""

        # Check if any validation failure seems critical (like length constraints)
        critical_failures = []
        for validation in failed_validations:
            message = validation.get("message", "").lower()
            if any(
                keyword in message
                for keyword in ["too long", "maximum", "limit", "characters", "words"]
            ):
                critical_failures.append(validation)

        if critical_failures:
            # Handle critical failures with strong language
            return (
                "🚨 CRITICAL VALIDATION FAILURES: The following requirements MUST be met and take "
                "ABSOLUTE PRIORITY over all other suggestions:\n\n"
            )
        else:
            # Handle regular validation failures
            return "⚠️ VALIDATION REQUIREMENTS: The text must be corrected to meet the following requirements:\n\n"

    @staticmethod
    def categorize_feedback(constraints: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Categorize feedback based on validation constraints.

        Args:
            constraints: Constraint information from extract_constraints().

        Returns:
            Dictionary with categorized feedback templates.
        """
        if not constraints or constraints.get("type") != "validation_failures":
            return {
                "validation_header": "Validation Issues:",
                "critic_header": "Critic Feedback:",
                "priority_instructions": (
                    "addresses the issues identified in the feedback while maintaining the core message "
                    "and staying true to the original task"
                ),
            }

        # Check if any validation failure seems critical
        failed_validations = constraints.get("failed_validations", [])
        has_critical_failures = any(
            any(
                keyword in validation.get("message", "").lower()
                for keyword in ["too long", "maximum", "limit", "characters", "words"]
            )
            for validation in failed_validations
        )

        if has_critical_failures:
            return {
                "validation_header": "🚨 CRITICAL VALIDATION REQUIREMENTS (MUST BE ADDRESSED FIRST):",
                "critic_header": "Secondary Feedback (IGNORE if it conflicts with validation requirements):",
                "priority_instructions": (
                    "IMMEDIATELY addresses the validation requirements listed above. "
                    "These requirements take ABSOLUTE PRIORITY over all other feedback. "
                    "Only consider other suggestions if they don't conflict with validation requirements"
                ),
            }
        else:
            return {
                "validation_header": "⚠️ VALIDATION REQUIREMENTS:",
                "critic_header": "Additional Feedback:",
                "priority_instructions": (
                    "first addresses the validation requirements, then incorporates other feedback "
                    "while maintaining the core message and staying true to the original task"
                ),
            }

    # Legacy methods - no longer needed with generic validation handling

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
    def _detect_language_constraints(
        validation_results: Dict[str, ValidationResult],
    ) -> Optional[Dict[str, Any]]:
        """Detect language constraints from validation results.

        Args:
            validation_results: Dictionary of validation results from validators.

        Returns:
            Dictionary containing language constraint information, or None if no language constraints found.
        """
        for validator_name, result in validation_results.items():
            if not result.passed:
                # Check if this is a language validation failure
                if (
                    "language" in validator_name.lower()
                    or "classified as invalid label" in result.message
                    or "text classified as" in result.message.lower()
                ):
                    # Extract the detected language and expected language
                    detected_lang = None
                    expected_lang = None

                    # Parse the message to extract language info
                    if "classified as invalid label" in result.message:
                        # Format: "Text classified as invalid label 'es'"
                        try:
                            detected_lang = result.message.split("'")[1]
                        except (IndexError, AttributeError):
                            detected_lang = "unknown"

                    # Try to get expected language from suggestions
                    if result.suggestions:
                        for suggestion in result.suggestions:
                            if "valid categories:" in suggestion:
                                try:
                                    # Extract expected languages from suggestions
                                    categories_part = suggestion.split("valid categories:")[
                                        1
                                    ].strip()
                                    if categories_part.startswith("[") and categories_part.endswith(
                                        "]"
                                    ):
                                        expected_lang = (
                                            categories_part.strip("[]")
                                            .replace("'", "")
                                            .replace('"', "")
                                        )
                                except (IndexError, AttributeError):
                                    pass

                    return {
                        "type": "language",
                        "validator_name": validator_name,
                        "message": result.message,
                        "detected_language": detected_lang,
                        "expected_language": expected_lang or "en",  # Default to English
                        "confidence": getattr(result, "score", 0.0),
                        "suggestions": result.suggestions or [],
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
