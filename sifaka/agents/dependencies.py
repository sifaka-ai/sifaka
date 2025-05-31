"""PydanticAI dependency adapters for Sifaka components.

This module provides wrapper classes that adapt Sifaka validators and critics
to work as PydanticAI dependencies, enabling generation-time guidance and
real-time feedback during agent execution.

The dependency system allows validators and critics to be injected into:
- System prompts (for context-aware generation)
- Tools (for real-time validation/criticism)
- Output validators (for generation-time validation)

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents.dependencies import SifakaDependencies
    from sifaka.validators import LengthValidator
    from sifaka.critics import ReflexionCritic

    # Create Sifaka components
    validator = LengthValidator(min_length=100, max_length=500)
    critic = ReflexionCritic(model=create_model("openai:gpt-4"))

    # Create dependency container
    deps = SifakaDependencies(validators=[validator], critics=[critic])

    # Create agent with dependencies
    agent = Agent("openai:gpt-4", deps_type=SifakaDependencies)

    # Use in system prompt
    @agent.system_prompt
    async def get_system_prompt(ctx: RunContext[SifakaDependencies]) -> str:
        # Access validators/critics through ctx.deps
        return "You are a helpful assistant."
    ```
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import CriticFeedback, Thought, ValidationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SifakaDependencies:
    """Container for Sifaka components as PydanticAI dependencies.

    This class serves as a dependency container that holds Sifaka validators
    and critics, making them available to PydanticAI system prompts, tools,
    and output validators.

    Supports context manager protocol for proper resource lifecycle management.

    Attributes:
        validators: List of Sifaka validators for generation-time validation.
        critics: List of Sifaka critics for generation-time feedback.
        enable_generation_time_validation: Whether to enable real-time validation.
        enable_generation_time_criticism: Whether to enable real-time criticism.
    """

    validators: List[Validator]
    critics: List[Critic]
    enable_generation_time_validation: bool = True
    enable_generation_time_criticism: bool = True
    _is_active: bool = True

    def __enter__(self):
        """Enter context manager - initialize dependencies."""
        self._is_active = True
        logger.debug(
            f"Activated SifakaDependencies with {len(self.validators)} validators and {len(self.critics)} critics"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup dependencies."""
        self._cleanup()
        return False

    def _cleanup(self):
        """Clean up dependency resources."""
        if not self._is_active:
            return

        logger.debug("Cleaning up SifakaDependencies resources")

        # Cleanup validators that support it
        for validator in self.validators:
            if hasattr(validator, "cleanup"):
                try:
                    validator.cleanup()  # type: ignore
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up validator {validator.__class__.__name__}: {e}"
                    )

        # Cleanup critics that support it
        for critic in self.critics:
            if hasattr(critic, "cleanup"):
                try:
                    critic.cleanup()  # type: ignore
                except Exception as e:
                    logger.warning(f"Error cleaning up critic {critic.__class__.__name__}: {e}")

        self._is_active = False

    def is_healthy(self) -> bool:
        """Check if all dependencies are in a healthy state."""
        if not self._is_active:
            return False

        # Check validators
        for validator in self.validators:
            if hasattr(validator, "is_healthy"):
                try:
                    if not validator.is_healthy():  # type: ignore
                        return False
                except Exception:
                    return False

        # Check critics
        for critic in self.critics:
            if hasattr(critic, "is_healthy"):
                try:
                    if not critic.is_healthy():  # type: ignore
                        return False
                except Exception:
                    return False

        return True

    async def validate_text_async(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate text using configured validators.

        This method provides generation-time validation by running all configured
        validators against the provided text.

        Args:
            text: The text to validate.
            context: Optional context information for validation.

        Returns:
            List of validation results from all validators.
        """
        if not self.enable_generation_time_validation or not self.validators:
            return []

        logger.debug(f"Running generation-time validation with {len(self.validators)} validators")

        # Create a temporary thought for validation
        thought = Thought(prompt="", text=text, chain_id="dependency-validation")

        results = []
        for validator in self.validators:
            try:
                # Use standard async validation (all validators support this)
                result = await validator.validate_async(thought)
                results.append(result)
                logger.debug(
                    f"Validation by {validator.__class__.__name__}: {'PASSED' if result.passed else 'FAILED'}"
                )
            except Exception as e:
                logger.error(
                    f"Generation-time validation failed for {validator.__class__.__name__}: {e}"
                )
                # Create error result
                error_result = ValidationResult(
                    validator_name=validator.__class__.__name__,
                    passed=False,
                    issues=[str(e)],
                    suggestions=["Please check the validator configuration"],
                )
                results.append(error_result)

        return results

    async def critique_text_async(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[CriticFeedback]:
        """Critique text using configured critics.

        This method provides generation-time criticism by running all configured
        critics against the provided text.

        Args:
            text: The text to critique.
            context: Optional context information for criticism.

        Returns:
            List of critic feedback from all critics.
        """
        if not self.enable_generation_time_criticism or not self.critics:
            return []

        logger.debug(f"Running generation-time criticism with {len(self.critics)} critics")

        # Create a temporary thought for criticism
        thought = Thought(prompt="", text=text, chain_id="dependency-criticism")

        feedback_list = []
        for critic in self.critics:
            try:
                # Use async criticism if available, otherwise fall back to sync
                if hasattr(critic, "_critique_async"):
                    result = await critic._critique_async(thought)  # type: ignore
                else:
                    result = critic.critique(thought)

                # Convert dict result to CriticFeedback if needed
                if isinstance(result, dict):
                    feedback = CriticFeedback(
                        critic_name=critic.__class__.__name__,
                        feedback=result.get("feedback", ""),
                        confidence=result.get("confidence", 0.0),
                        issues=result.get("issues", []),
                        suggestions=result.get("suggestions", []),
                        needs_improvement=result.get("needs_improvement", False),
                    )
                else:
                    feedback = result

                feedback_list.append(feedback)
                logger.debug(
                    f"Criticism by {critic.__class__.__name__}: confidence={feedback.confidence}"
                )
            except Exception as e:
                logger.error(
                    f"Generation-time criticism failed for {critic.__class__.__name__}: {e}"
                )
                # Create error feedback
                error_feedback = CriticFeedback(
                    critic_name=critic.__class__.__name__,
                    feedback="Criticism failed due to an error",
                    confidence=0.0,
                    issues=[str(e)],
                    suggestions=["Please check the critic configuration"],
                    needs_improvement=False,
                )
                feedback_list.append(error_feedback)

        return feedback_list

    def get_validation_summary(self, validation_results: List[ValidationResult]) -> str:
        """Get a summary of validation results for use in prompts.

        Args:
            validation_results: List of validation results.

        Returns:
            A formatted summary string.
        """
        if not validation_results:
            return "No validation performed."

        passed_count = sum(1 for result in validation_results if result.passed)
        total_count = len(validation_results)

        summary = f"Validation: {passed_count}/{total_count} passed"

        # Add details for failed validations
        failed_results = [result for result in validation_results if not result.passed]
        if failed_results:
            summary += "\nFailed validations:"
            for result in failed_results:
                summary += f"\n- {result.validator_name}: {', '.join(result.issues)}"

        return summary

    def get_criticism_summary(self, feedback_list: List[CriticFeedback]) -> str:
        """Get a summary of criticism feedback for use in prompts.

        Args:
            feedback_list: List of critic feedback.

        Returns:
            A formatted summary string.
        """
        if not feedback_list:
            return "No criticism performed."

        avg_confidence = sum(feedback.confidence for feedback in feedback_list) / len(feedback_list)
        needs_improvement_count = sum(1 for feedback in feedback_list if feedback.needs_improvement)

        summary = f"Criticism: {len(feedback_list)} critics, avg confidence={avg_confidence:.2f}"

        if needs_improvement_count > 0:
            summary += f", {needs_improvement_count} suggest improvements"

            # Add improvement suggestions
            suggestions = []
            for feedback in feedback_list:
                if feedback.needs_improvement and feedback.suggestions:
                    suggestions.extend(feedback.suggestions)

            if suggestions:
                summary += "\nSuggestions:"
                for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
                    summary += f"\n- {suggestion}"

        return summary
