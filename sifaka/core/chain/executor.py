"""Chain execution logic for Sifaka.

This module contains the ChainExecutor class which handles the low-level
execution logic for chain operations including generation, validation, and criticism.
"""

from typing import Dict, List, Tuple

from sifaka.core.chain.config import ChainConfig
from sifaka.core.chain.orchestrator import ChainOrchestrator
from sifaka.core.thought import CriticFeedback, Thought, ValidationResult
from sifaka.utils.performance import time_operation
from sifaka.utils.error_handling import chain_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ChainExecutor:
    """Low-level execution engine for chain operations.

    This class handles the actual execution of generation, validation,
    and criticism operations within a chain.
    """

    def __init__(self, config: ChainConfig, orchestrator: ChainOrchestrator):
        """Initialize the executor with configuration and orchestrator.

        Args:
            config: The chain configuration.
            orchestrator: The chain orchestrator for workflow coordination.
        """
        self.config = config
        self.orchestrator = orchestrator

    def execute_generation(self, thought: Thought) -> Thought:
        """Execute text generation using the configured model.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with generated text.
        """
        with time_operation("text_generation"):
            with chain_context(
                operation="generation",
                message_prefix="Failed to generate text",
            ):
                logger.debug("Generating text using model...")
                text, model_prompt = self.config.model.generate_with_thought(thought)
                thought = thought.set_text(text).set_model_prompt(model_prompt)
                logger.debug(f"Generated text: {len(text)} characters")

                return thought

    def execute_validation(self, thought: Thought) -> Tuple[Thought, bool]:
        """Execute validation using all configured validators.

        Args:
            thought: The current thought state.

        Returns:
            Tuple of (updated thought with validation results, overall validation passed).
        """
        if not self.config.validators:
            logger.debug("No validators configured, skipping validation")
            return thought, True

        with time_operation("validation"):
            logger.debug(f"Running validation with {len(self.config.validators)} validators")

            all_passed = True
            for validator in self.config.validators:
                try:
                    with chain_context(
                        operation="validation",
                        message_prefix=f"Validation failed for {validator.__class__.__name__}",
                        metadata={"validator_name": validator.__class__.__name__},
                    ):
                        result = validator.validate(thought)
                        thought = thought.add_validation_result(
                            validator.__class__.__name__, result
                        )

                        if not result.passed:
                            all_passed = False
                            logger.debug(
                                f"Validation failed for {validator.__class__.__name__}: {result.message}"
                            )
                        else:
                            logger.debug(f"Validation passed for {validator.__class__.__name__}")

                except Exception as e:
                    logger.error(f"Validation error for {validator.__class__.__name__}: {e}")
                    # Create a failed validation result for the error
                    error_result = ValidationResult(
                        passed=False, message=f"Validation error: {str(e)}", score=0.0
                    )
                    thought = thought.add_validation_result(
                        validator.__class__.__name__, error_result
                    )
                    all_passed = False

            logger.debug(f"Overall validation result: {'PASSED' if all_passed else 'FAILED'}")
            return thought, all_passed

    def execute_criticism(self, thought: Thought) -> Thought:
        """Execute criticism using all configured critics.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic feedback.
        """
        if not self.config.critics:
            logger.debug("No critics configured, skipping criticism")
            return thought

        with time_operation("criticism"):
            logger.debug(f"Running criticism with {len(self.config.critics)} critics")

            for critic in self.config.critics:
                try:
                    with chain_context(
                        operation="criticism",
                        message_prefix=f"Criticism failed for {critic.__class__.__name__}",
                        metadata={"critic_name": critic.__class__.__name__},
                    ):
                        feedback_dict = critic.critique(thought)

                        # Convert the dictionary to a CriticFeedback object
                        feedback = CriticFeedback(
                            critic_name=critic.__class__.__name__,
                            confidence=feedback_dict.get("confidence", 0.8),
                            violations=feedback_dict.get("issues", []),
                            suggestions=feedback_dict.get("suggestions", []),
                            feedback=feedback_dict,  # Store the full feedback dict
                        )

                        thought = thought.add_critic_feedback(feedback)
                        logger.debug(f"Added feedback from {critic.__class__.__name__}")

                except Exception as e:
                    logger.error(f"Criticism error for {critic.__class__.__name__}: {e}")
                    # Create error feedback
                    error_feedback = CriticFeedback(
                        critic_name=critic.__class__.__name__,
                        confidence=0.0,
                        violations=[f"Criticism error: {str(e)}"],
                        suggestions=["Please try again or check the critic configuration."],
                        feedback={"error": str(e)},
                    )
                    thought = thought.add_critic_feedback(error_feedback)

            return thought

    def execute_improvement_iteration(self, thought: Thought) -> Thought:
        """Execute a single improvement iteration with criticism and regeneration.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought after improvement iteration.
        """
        logger.debug(f"Starting improvement iteration {thought.iteration}")

        # Apply critic retrieval if configured
        thought = self.orchestrator.orchestrate_retrieval(thought, "critic")

        # Apply critics to get feedback
        thought = self.execute_criticism(thought)

        # Create next iteration with feedback for the model to see
        thought = thought.next_iteration()

        # Apply pre-generation retrieval for the new iteration
        thought = self.orchestrator.orchestrate_retrieval(thought, "pre_generation")

        # Generate improved text
        thought = self.execute_generation(thought)

        # Apply post-generation retrieval
        thought = self.orchestrator.orchestrate_retrieval(thought, "post_generation")

        # Save the improved thought to storage
        self._save_thought_to_storage(thought)

        logger.debug(f"Completed improvement iteration {thought.iteration}")
        return thought

    def execute_improvement_loop(self, thought: Thought, validation_passed: bool) -> Thought:
        """Execute the complete improvement loop with multiple iterations.

        Args:
            thought: The current thought state.
            validation_passed: Whether initial validation passed.

        Returns:
            Final thought after all improvement iterations.
        """
        if not self.orchestrator.should_apply_critics(validation_passed):
            logger.debug("Skipping improvement loop based on configuration and validation results")
            return thought

        max_iterations = self.orchestrator.get_max_iterations()
        current_iteration = 0

        logger.debug(f"Starting improvement loop (max {max_iterations} iterations)")

        while current_iteration < max_iterations:
            current_iteration += 1
            logger.debug(f"Improvement iteration {current_iteration}/{max_iterations}")

            # Execute improvement iteration
            thought = self.execute_improvement_iteration(thought)

            # Re-validate after improvement
            thought, validation_passed = self.execute_validation(thought)

            # If validation passes and we're not always applying critics, we can stop
            if validation_passed and not self.config.get_option("always_apply_critics", False):
                logger.debug(
                    f"Validation passed after iteration {current_iteration}, stopping improvement loop"
                )
                break

        if current_iteration >= max_iterations:
            logger.debug(
                f"Reached maximum iterations ({max_iterations}), stopping improvement loop"
            )

        return thought

    def _save_thought_to_storage(self, thought: Thought) -> None:
        """Save a thought to the configured storage.

        Args:
            thought: The thought to save.
        """
        try:
            thought_key = f"thought_{self.config.chain_id}_{thought.iteration}"
            self.config.storage.set(thought_key, thought.model_dump())
            logger.debug(f"Saved thought (iteration {thought.iteration}) to storage")
        except Exception as e:
            logger.warning(f"Failed to save thought to storage: {e}")
