"""Chain execution logic for Sifaka.

This module contains the ChainExecutor class which handles the low-level
execution logic for chain operations including generation, validation, and criticism.

The executor supports both sync and async execution internally, with sync methods
wrapping async implementations using asyncio.run() for backward compatibility.
"""

import asyncio
from typing import Any, Dict, Tuple

from sifaka.core.chain.config import ChainConfig
from sifaka.core.chain.orchestrator import ChainOrchestrator
from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import CriticFeedback, Thought, ValidationResult
from sifaka.utils.error_handling import chain_context
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

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
                if self.config.model is None:
                    raise ValueError("No model configured for text generation")

                logger.debug("Generating text using model...")
                text, model_prompt = self.config.model.generate_with_thought(thought)
                thought = thought.set_text(text).set_model_prompt(model_prompt)
                thought = thought.model_copy(update={"model_name": self.config.model.model_name})
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
                        validator_name = getattr(validator, "name", validator.__class__.__name__)
                        thought = thought.add_validation_result(validator_name, result)

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
                    validator_name = getattr(validator, "name", validator.__class__.__name__)
                    thought = thought.add_validation_result(validator_name, error_result)
                    all_passed = False

            logger.debug(f"Overall validation result: {'PASSED' if all_passed else 'FAILED'}")
            return thought, all_passed

    async def _execute_validation_async(self, thought: Thought) -> Tuple[Thought, bool]:
        """Execute validation using all configured validators asynchronously.

        Args:
            thought: The current thought state.

        Returns:
            Tuple of (updated thought with validation results, overall validation passed).
        """
        if not self.config.validators:
            logger.debug("No validators configured, skipping validation")
            return thought, True

        with time_operation("validation_async"):
            logger.debug(f"Running async validation with {len(self.config.validators)} validators")

            # Run all validators concurrently
            validation_tasks = []
            for validator in self.config.validators:
                validation_tasks.append(self._validate_with_validator_async(validator, thought))

            # Wait for all validations to complete
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            all_passed = True
            for i, result in enumerate(validation_results):
                validator = self.config.validators[i]
                validator_name = getattr(validator, "name", validator.__class__.__name__)

                if isinstance(result, Exception):
                    logger.error(f"Validation error for {validator_name}: {result}")
                    # Create a failed validation result for the error
                    error_result = ValidationResult(
                        passed=False, message=f"Validation error: {str(result)}", score=0.0
                    )
                    thought = thought.add_validation_result(validator_name, error_result)
                    all_passed = False
                elif isinstance(result, ValidationResult):
                    thought = thought.add_validation_result(validator_name, result)
                    if not result.passed:
                        all_passed = False
                        logger.debug(f"Validation failed for {validator_name}: {result.message}")
                    else:
                        logger.debug(f"Validation passed for {validator_name}")

            logger.debug(f"Overall async validation result: {'PASSED' if all_passed else 'FAILED'}")
            return thought, all_passed

    async def _validate_with_validator_async(
        self, validator: Validator, thought: Thought
    ) -> ValidationResult:
        """Run a single validator asynchronously with error handling."""
        try:
            with chain_context(
                operation="validation_async",
                message_prefix=f"Async validation failed for {validator.__class__.__name__}",
                metadata={"validator_name": validator.__class__.__name__},
            ):
                return await validator._validate_async(thought)  # type: ignore
        except Exception as e:
            # Return a failed validation result for the error
            return ValidationResult(passed=False, message=f"Validation error: {str(e)}", score=0.0)

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
                        # Extract the main feedback text (try multiple possible fields)
                        main_feedback = (
                            feedback_dict.get("message")
                            or feedback_dict.get("critique")
                            or feedback_dict.get("feedback", "")
                        )

                        # Add critic model information to metadata
                        enhanced_metadata = dict(feedback_dict)
                        if hasattr(critic, "model") and hasattr(critic.model, "model_name"):
                            enhanced_metadata["critic_model_name"] = critic.model.model_name

                        feedback = CriticFeedback(
                            critic_name=critic.__class__.__name__,
                            feedback=main_feedback,  # Store main feedback as string
                            needs_improvement=feedback_dict.get("needs_improvement", False),
                            confidence=feedback_dict.get("confidence", 0.8),
                            violations=feedback_dict.get("issues", []),
                            suggestions=feedback_dict.get("suggestions", []),
                            metadata=enhanced_metadata,  # Store the full feedback dict with model info in metadata
                        )

                        thought = thought.add_critic_feedback(feedback)
                        logger.debug(f"Added feedback from {critic.__class__.__name__}")

                except Exception as e:
                    logger.error(f"Criticism error for {critic.__class__.__name__}: {e}")
                    # Create error feedback
                    error_feedback = CriticFeedback(
                        critic_name=critic.__class__.__name__,
                        feedback=f"Criticism error: {str(e)}",
                        confidence=0.0,
                        violations=[f"Criticism error: {str(e)}"],
                        suggestions=["Please try again or check the critic configuration."],
                        metadata={"error": str(e)},
                    )
                    thought = thought.add_critic_feedback(error_feedback)

            return thought

    async def _execute_criticism_async(self, thought: Thought) -> Thought:
        """Execute criticism using all configured critics asynchronously.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic feedback.
        """
        if not self.config.critics:
            logger.debug("No critics configured, skipping criticism")
            return thought

        with time_operation("criticism_async"):
            logger.debug(f"Running async criticism with {len(self.config.critics)} critics")

            # Run all critics concurrently
            criticism_tasks = []
            for critic in self.config.critics:
                criticism_tasks.append(self._critique_with_critic_async(critic, thought))

            # Wait for all criticisms to complete
            criticism_results = await asyncio.gather(*criticism_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(criticism_results):
                critic = self.config.critics[i]
                critic_name = critic.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Criticism error for {critic_name}: {result}")
                    # Create error feedback
                    error_feedback = CriticFeedback(
                        critic_name=critic_name,
                        feedback=f"Criticism error: {str(result)}",
                        needs_improvement=True,
                        confidence=0.0,
                        violations=[f"Criticism error: {str(result)}"],
                        suggestions=["Please try again or check the critic configuration."],
                        metadata={"error": str(result)},
                    )
                    thought = thought.add_critic_feedback(error_feedback)
                elif isinstance(result, dict):
                    # Convert the dictionary to a CriticFeedback object
                    # Extract the main feedback text (try multiple possible fields)
                    main_feedback = (
                        result.get("message")
                        or result.get("critique")
                        or result.get("feedback", "")
                    )

                    # Add critic model information to metadata
                    enhanced_metadata = dict(result)
                    if hasattr(critic, "model") and hasattr(critic.model, "model_name"):
                        enhanced_metadata["critic_model_name"] = critic.model.model_name

                    feedback = CriticFeedback(
                        critic_name=critic_name,
                        feedback=main_feedback,  # Store main feedback as string
                        needs_improvement=result.get("needs_improvement", False),
                        confidence=result.get("confidence", 0.8),
                        violations=result.get("issues", []),
                        suggestions=result.get("suggestions", []),
                        metadata=enhanced_metadata,  # Store the full feedback dict with model info in metadata
                    )
                    thought = thought.add_critic_feedback(feedback)
                    logger.debug(f"Added async feedback from {critic_name}")

            return thought

    async def _critique_with_critic_async(self, critic: Critic, thought: Thought) -> Dict[str, Any]:
        """Run a single critic asynchronously with error handling."""
        try:
            with chain_context(
                operation="criticism_async",
                message_prefix=f"Async criticism failed for {critic.__class__.__name__}",
                metadata={"critic_name": critic.__class__.__name__},
            ):
                return await critic._critique_async(thought)  # type: ignore
        except Exception as e:
            # Return error feedback dict
            return {"error": str(e), "confidence": 0.0, "issues": [str(e)], "suggestions": []}

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

        # Save the thought with critic feedback BEFORE creating next iteration
        self._save_thought_to_storage(thought)

        # Create next iteration with feedback for the model to see
        thought = thought.next_iteration()

        # Apply pre-generation retrieval for the new iteration
        thought = self.orchestrator.orchestrate_retrieval(thought, "pre_generation")

        # Generate improved text using critics if feedback is available
        thought = self._execute_improvement_generation(thought)

        # Apply post-generation retrieval
        thought = self.orchestrator.orchestrate_retrieval(thought, "post_generation")

        # Save the improved thought to storage
        self._save_thought_to_storage(thought)

        logger.debug(f"Completed improvement iteration {thought.iteration}")
        return thought

    def _execute_improvement_generation(self, thought: Thought) -> Thought:
        """Execute text generation using critics' improve methods when feedback is available.

        Args:
            thought: The current thought state with potential critic feedback.

        Returns:
            Updated thought with improved text.
        """
        # Check if we have critic feedback from the previous iteration
        if thought.critic_feedback and len(thought.critic_feedback) > 0:
            logger.debug(
                f"Using critic improvement methods for {len(thought.critic_feedback)} critics"
            )

            # Use the first critic that has feedback and can improve
            for feedback in thought.critic_feedback:
                if feedback.needs_improvement:
                    # Find the corresponding critic
                    for critic in self.config.critics:
                        if critic.__class__.__name__ == feedback.critic_name:
                            try:
                                logger.debug(f"Using {feedback.critic_name} to improve text")

                                # Get the original text from the previous iteration (from history)
                                original_text = None
                                if thought.history and len(thought.history) > 0:
                                    # Try to get the original text from the most recent history entry
                                    previous_thought_id = thought.history[0].thought_id
                                    try:
                                        # Try to get the previous thought from storage to get its text
                                        previous_thought = self.config.storage.get(
                                            previous_thought_id
                                        )
                                        if (
                                            previous_thought
                                            and hasattr(previous_thought, "text")
                                            and previous_thought.text
                                        ):
                                            original_text = previous_thought.text
                                            logger.debug(
                                                f"Retrieved original text from previous iteration: {len(original_text)} characters"
                                            )
                                    except Exception as e:
                                        logger.debug(
                                            f"Could not retrieve previous thought text: {e}"
                                        )

                                # Create a temporary thought with the original text for the critic to improve
                                if original_text:
                                    temp_thought = thought.model_copy(
                                        update={"text": original_text}
                                    )
                                    improved_text = critic.improve(temp_thought)
                                else:
                                    # Fall back to regular generation if no original text available
                                    logger.debug(
                                        "No original text available for critic improvement, falling back to regular generation"
                                    )
                                    break

                                # Get the actual improvement prompt that was sent to the model
                                if (
                                    hasattr(critic, "last_improvement_prompt")
                                    and critic.last_improvement_prompt
                                ):
                                    model_prompt = critic.last_improvement_prompt
                                else:
                                    model_prompt = (
                                        f"Improved by {feedback.critic_name} (prompt not available)"
                                    )

                                # Update thought with improved text
                                thought = thought.set_text(improved_text).set_model_prompt(
                                    model_prompt
                                )
                                thought = thought.model_copy(
                                    update={"model_name": self.config.model.model_name}
                                )

                                logger.debug(
                                    f"Generated improved text using {feedback.critic_name}: {len(improved_text)} characters"
                                )
                                return thought

                            except Exception as e:
                                logger.error(
                                    f"Failed to improve text using {feedback.critic_name}: {e}"
                                )
                                # Fall back to regular generation
                                break

            # If no critic could improve or all failed, fall back to regular generation
            logger.debug("No critic could improve text, falling back to regular generation")

        # Fall back to regular generation when no critic feedback or improvement needed
        return self.execute_generation(thought)

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

    async def _save_thought_to_storage_async(self, thought: Thought) -> None:
        """Save a thought to the configured storage asynchronously.

        Args:
            thought: The thought to save.
        """
        try:
            # Save by thought ID only - let storage backends handle their own key strategies
            await self.config.storage._set_async(thought.id, thought)
            logger.debug(f"Saved thought (iteration {thought.iteration}) to storage")
        except Exception as e:
            logger.warning(f"Failed to save thought to storage: {e}")

    def _save_thought_to_storage(self, thought: Thought) -> None:
        """Save a thought to the configured storage (sync wrapper).

        Args:
            thought: The thought to save.
        """
        try:
            # Save by thought ID only - let storage backends handle their own key strategies
            self.config.storage.set(thought.id, thought)
            logger.debug(f"Saved thought (iteration {thought.iteration}) to storage")
        except Exception as e:
            logger.warning(f"Failed to save thought to storage: {e}")
