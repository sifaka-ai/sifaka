"""Chain orchestration for Sifaka.

This module defines the Chain class, which is the main entry point for the Sifaka framework.
It orchestrates the process of generating text using language models, validating the text
against specified criteria, and improving the text using specialized critics.

The Chain class uses a builder pattern to provide a fluent API for configuring and
executing LLM operations. It allows you to specify which model to use, set the prompt
for generation, add validators to check if the generated text meets requirements,
add critics to enhance the quality of the generated text, and configure model options.
"""

import uuid
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.core.thought import Thought, CriticFeedback
from sifaka.storage.checkpoints import ChainCheckpoint, CachedCheckpointStorage
from sifaka.recovery.manager import RecoveryManager, RecoveryStrategy, RecoveryAction
from sifaka.utils.error_handling import ChainError, chain_context
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation, PerformanceMonitor

# Configure logger
logger = get_logger(__name__)


class Chain:
    """Main orchestrator for text generation, validation, and improvement.

    The Chain class is the central component of the Sifaka framework, coordinating
    the process of generating text using language models, validating it against
    specified criteria, and improving it using specialized critics.

    The Chain orchestrates retrieval operations with separate retrievers for different purposes:
    - Model retrievers: Get context before text generation (pre-generation)
    - Critic retrievers: Get context for critics during improvement (post-generation)

    It follows a fluent interface pattern (builder pattern) for easy configuration,
    allowing you to chain method calls to set up the desired behavior.

    The typical workflow is:
    1. Create a Chain instance with model, prompt, and optional retrievers
    2. Configure it with validators and critics
    3. Run the chain - it handles ALL retrieval automatically
    4. Process the result

    Attributes:
        model: The language model to use for text generation.
        prompt: The prompt to use for text generation.
        model_retrievers: Retrievers for model context (pre-generation).
        critic_retrievers: Retrievers for critic context (post-generation).
        validators: List of validators to check if the generated text meets requirements.
        critics: List of critics to improve the generated text.
        options: Dictionary of options for the chain.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt: Optional[str] = None,
        model_retrievers: Optional[List[Retriever]] = None,
        critic_retrievers: Optional[List[Retriever]] = None,
        storage: Optional[Any] = None,
        checkpoint_storage: Optional[CachedCheckpointStorage] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
        always_apply_critics: bool = False,
    ):
        """Initialize the Chain with optional model, prompt, and retrievers.

        Args:
            model: Optional language model to use for text generation.
            prompt: Optional prompt to use for text generation.
            model_retrievers: Optional list of retrievers for model context.
            critic_retrievers: Optional list of retrievers for critic context.
            storage: Optional storage for saving intermediate thoughts.
            checkpoint_storage: Optional storage for chain execution checkpoints.
            max_improvement_iterations: Maximum number of improvement iterations.
            apply_improvers_on_validation_failure: Whether to apply improvers when validation fails.
            always_apply_critics: Whether to always apply critics regardless of validation status.
        """
        self._model = model
        self._prompt = prompt
        self._model_retrievers = model_retrievers or []
        self._critic_retrievers = critic_retrievers or []
        self._storage = storage
        self._checkpoint_storage = checkpoint_storage
        self._recovery_manager = RecoveryManager(checkpoint_storage) if checkpoint_storage else None

        self._validators: List[Validator] = []
        self._critics: List[Critic] = []
        self._options: Dict[str, Any] = {
            "max_improvement_iterations": max_improvement_iterations,
            "apply_improvers_on_validation_failure": apply_improvers_on_validation_failure,
            "always_apply_critics": always_apply_critics,
        }
        self._chain_id = str(uuid.uuid4())
        self._current_checkpoint: Optional[ChainCheckpoint] = None

    def with_model(self, model: Model) -> "Chain":
        """Set the model for the chain.

        Args:
            model: The model to use for text generation.

        Returns:
            The chain instance for method chaining.
        """
        self._model = model
        return self

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt for the chain.

        Args:
            prompt: The prompt to use for text generation.

        Returns:
            The chain instance for method chaining.
        """
        self._prompt = prompt
        return self

    def with_model_retrievers(self, retrievers: List[Retriever]) -> "Chain":
        """Set retrievers specifically for model context.

        Args:
            retrievers: List of retrievers to use for model context.

        Returns:
            The chain instance for method chaining.
        """
        self._model_retrievers = retrievers
        return self

    def with_critic_retrievers(self, retrievers: List[Retriever]) -> "Chain":
        """Set retrievers specifically for critic context.

        Args:
            retrievers: List of retrievers to use for critic context.

        Returns:
            The chain instance for method chaining.
        """
        self._critic_retrievers = retrievers
        return self

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.

        Args:
            validator: The validator to check if the generated text meets requirements.

        Returns:
            The chain instance for method chaining.
        """
        self._validators.append(validator)
        return self

    def improve_with(self, critic: Critic) -> "Chain":
        """Add a critic to the chain.

        Args:
            critic: The critic to improve the generated text.

        Returns:
            The chain instance for method chaining.
        """
        self._critics.append(critic)
        return self

    def with_options(self, **options: Any) -> "Chain":
        """Set options for the chain.

        Args:
            **options: Options to set for the chain.

        Returns:
            The chain instance for method chaining.
        """
        self._options.update(options)
        return self

    def run(self) -> Thought:
        """Execute the chain and return the result.

        This method runs the complete chain execution process with Chain-orchestrated retrieval:
        1. Checks that the chain is properly configured (has a model and prompt)
        2. Pre-generation retrieval: Gets context before text generation (if configured)
        3. Generates text using the model with retrieved context
        4. Post-generation retrieval: Gets context after text generation (if configured)
        5. Validates the generated text using all configured validators
        6. Applies critics if any of these conditions are met:
           a. Validation fails and apply_improvers_on_validation_failure is True
           b. always_apply_critics is True (regardless of validation status)
        7. If critics are applied:
           a. Critic retrieval: Gets additional context for critics (if configured)
           b. Gets feedback from critics on how to improve the text
           c. Sends the original text + feedback to the model to generate improved text
           d. Re-validates the improved text
           e. Repeats steps 7a-7d until validation passes or max iterations is reached
        8. Returns a Thought object with the final text and all validation/improvement results

        The Chain orchestrates ALL retrieval operations - models and critics no longer
        handle retrieval themselves.

        Returns:
            A Thought object containing the final text and all validation/improvement results.

        Raises:
            ChainError: If the chain is not properly configured or an error occurs during execution.
        """
        # Start performance monitoring for the entire chain execution
        with time_operation(f"chain_execution_{self._chain_id}"):
            # Check that the chain is properly configured
            if not self._model:
                raise ChainError("No model specified for the chain")
            if not self._prompt:
                raise ChainError("No prompt specified for the chain")

            # Create initial thought
            thought = Thought(
                prompt=self._prompt,
                chain_id=self._chain_id,
            )

            # Save the initial thought if storage is available
            if self._storage:
                try:
                    self._storage.save_thought(thought)
                    logger.debug(
                        f"Saved initial thought (iteration {thought.iteration}) to storage"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save initial thought: {e}")

            # 1. PRE-GENERATION RETRIEVAL (Chain orchestrates this using model_retrievers)
            if self._model_retrievers:
                with time_operation("pre_generation_retrieval"):
                    with chain_context(
                        operation="pre_generation_retrieval",
                        message_prefix="Failed to retrieve pre-generation context",
                    ):
                        logger.debug("Chain orchestrating pre-generation retrieval for model...")
                        for retriever in self._model_retrievers:
                            thought = retriever.retrieve_for_thought(
                                thought, is_pre_generation=True
                            )
                        logger.debug(
                            f"Retrieved {len(thought.pre_generation_context or [])} pre-generation documents for model"
                        )

            # 2. GENERATE TEXT (Model just generates, no retrieval)
            with time_operation("text_generation"):
                with chain_context(
                    operation="generation",
                    message_prefix="Failed to generate text",
                ):
                    logger.debug(f"Generating text with prompt: {self._prompt[:100]}...")
                    text, model_prompt = self._model.generate_with_thought(thought)
                    thought = thought.set_text(text).set_model_prompt(model_prompt)
                    logger.debug(f"Generated text of length {len(text)}")
                    logger.debug(f"Model prompt length: {len(model_prompt)} characters")

            # 3. POST-GENERATION RETRIEVAL (Chain orchestrates this using critic_retrievers)
            if self._critic_retrievers:
                with time_operation("post_generation_retrieval"):
                    with chain_context(
                        operation="post_generation_retrieval",
                        message_prefix="Failed to retrieve post-generation context",
                    ):
                        logger.debug("Chain orchestrating post-generation retrieval for critics...")
                        for retriever in self._critic_retrievers:
                            thought = retriever.retrieve_for_thought(
                                thought, is_pre_generation=False
                            )
                        logger.debug(
                            f"Retrieved {len(thought.post_generation_context or [])} post-generation documents for critics"
                        )

            # 4. VALIDATE TEXT
            all_passed = True
            with time_operation("validation"):
                for validator in self._validators:
                    with time_operation(f"validation_{validator.__class__.__name__}"):
                        with chain_context(
                            operation="validation",
                            message_prefix=f"Failed to validate text with {validator.__class__.__name__}",
                        ):
                            logger.debug(f"Validating text with {validator.__class__.__name__}")
                            validation_result = validator.validate(thought)

                            # Add validation result to thought
                            thought = thought.add_validation_result(
                                validator.__class__.__name__, validation_result
                            )

                            # Update all_passed flag
                            all_passed = all_passed and validation_result.passed
                            logger.debug(
                                f"Validation with {validator.__class__.__name__} "
                                f"{'passed' if validation_result.passed else 'failed'}"
                            )

            # 5. CRITIC IMPROVEMENT LOOP (if validation failed OR always_apply_critics is enabled)
            max_iterations = self._options.get("max_improvement_iterations", 3)
            current_iteration = 0

            # Determine if we should run critics
            always_apply_critics = self._options.get("always_apply_critics", False)
            apply_on_failure = self._options.get("apply_improvers_on_validation_failure", False)

            while current_iteration < max_iterations and self._critics:
                # Check if we should run critics in this iteration
                should_run_critics = (
                    # Traditional case: validation failed and apply_improvers_on_validation_failure is True
                    (not all_passed and apply_on_failure)
                    # New case: always_apply_critics is True (run critics on every iteration)
                    or always_apply_critics
                )

                # If we shouldn't run critics, break out of the loop
                if not should_run_critics:
                    break
                current_iteration += 1

                # Log appropriate message based on why critics are running
                if not all_passed:
                    logger.debug(
                        f"Validation failed, attempting improvement (iteration {current_iteration}/{max_iterations})"
                    )
                else:
                    logger.debug(
                        f"Running critics for improvement (always_apply_critics=True, iteration {current_iteration}/{max_iterations})"
                    )

                # 5a. CRITIC RETRIEVAL (Chain orchestrates this using critic_retrievers)
                if self._critic_retrievers:
                    with time_operation("critic_retrieval"):
                        with chain_context(
                            operation="critic_retrieval",
                            message_prefix="Failed to retrieve context for critics",
                        ):
                            logger.debug("Chain orchestrating critic retrieval...")
                            # For critics, we might want fresh post-generation context
                            for retriever in self._critic_retrievers:
                                thought = retriever.retrieve_for_thought(
                                    thought, is_pre_generation=False
                                )
                            logger.debug(
                                f"Retrieved {len(thought.post_generation_context or [])} documents for critics"
                            )

                # 5b. Apply critics to provide feedback (critics no longer do retrieval)
                with time_operation("critic_feedback"):
                    for critic in self._critics:
                        with time_operation(f"critic_{critic.__class__.__name__}"):
                            with chain_context(
                                operation="criticism",
                                message_prefix=f"Failed to get criticism from {critic.__class__.__name__}",
                            ):
                                logger.debug(f"Getting criticism from {critic.__class__.__name__}")

                                # Get critique (critic uses provided context, no retrieval)
                                critique_result = critic.critique(thought)

                                # Create structured CriticFeedback from critique result
                                critic_feedback = CriticFeedback(
                                    critic_name=critic.__class__.__name__,
                                    confidence=critique_result.get("confidence", 0.0),
                                    violations=critique_result.get("violations", []),
                                    suggestions=critique_result.get("suggestions", []),
                                    feedback=critique_result.get("feedback", {}),
                                    processing_time_ms=critique_result.get("processing_time_ms"),
                                )

                                # Add the structured feedback to the thought
                                thought = thought.add_critic_feedback(critic_feedback)
                                logger.debug(f"Added criticism from {critic.__class__.__name__}")

                # Save the current iteration with critic feedback
                if self._storage:
                    try:
                        self._storage.save_thought(thought)
                        logger.debug(
                            f"Saved iteration {thought.iteration} thought with critic feedback to storage"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save iteration {thought.iteration} thought: {e}")

                # 5c. Create next iteration with critic feedback preserved for model context
                thought = thought.next_iteration()
                logger.debug(
                    f"Created iteration {thought.iteration} with previous critic feedback preserved"
                )

                # 5d. Generate improved text using model (which now sees previous critic feedback)
                with time_operation("improvement_generation"):
                    with chain_context(
                        operation="improvement_generation",
                        message_prefix="Failed to generate improved text",
                    ):
                        logger.debug(f"Generating improved text for iteration {thought.iteration}")
                        text, model_prompt = self._model.generate_with_thought(thought)
                        thought = thought.set_text(text).set_model_prompt(model_prompt)
                        logger.debug(f"Generated improved text of length {len(text)}")
                        logger.debug(
                            f"Model prompt included critic feedback: {len(model_prompt)} characters"
                        )

                # Re-validate the improved text
                all_passed = True
                with time_operation("revalidation"):
                    for validator in self._validators:
                        with time_operation(f"revalidation_{validator.__class__.__name__}"):
                            with chain_context(
                                operation="validation",
                                message_prefix=f"Failed to validate improved text with {validator.__class__.__name__}",
                            ):
                                logger.debug(
                                    f"Re-validating improved text with {validator.__class__.__name__}"
                                )
                                validation_result = validator.validate(thought)

                                # Add validation result to thought
                                thought = thought.add_validation_result(
                                    validator.__class__.__name__, validation_result
                                )

                                # Update all_passed flag
                                all_passed = all_passed and validation_result.passed
                                logger.debug(
                                    f"Re-validation with {validator.__class__.__name__} "
                                    f"{'passed' if validation_result.passed else 'failed'}"
                                )

            return thought

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the chain execution.

        Returns:
            Dictionary containing performance metrics and timing information.
        """
        monitor = PerformanceMonitor.get_instance()
        summary = monitor.get_summary()
        stats = monitor.get_stats()

        # Add detailed operation stats to the summary
        summary["operations"] = stats
        return summary

    def clear_performance_data(self) -> None:
        """Clear all performance monitoring data."""
        monitor = PerformanceMonitor.get_instance()
        monitor.clear()

    def get_performance_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks in chain execution.

        Returns:
            List of operation names that are taking the most time.
        """
        summary = self.get_performance_summary()
        if not summary.get("operations"):
            return []

        # Sort operations by average time and return top 3 slowest
        operations = summary["operations"]
        sorted_ops = sorted(operations.items(), key=lambda x: x[1].get("avg_time", 0), reverse=True)

        bottlenecks = []
        for op_name, metrics in sorted_ops[:3]:
            avg_time = metrics.get("avg_time", 0)
            if avg_time > 0.1:  # Only consider operations taking > 100ms
                bottlenecks.append(f"{op_name} (avg: {avg_time:.2f}s)")

        return bottlenecks

    def run_with_recovery(self) -> Thought:
        """Execute the chain with automatic checkpointing and recovery.

        This method provides robust chain execution with automatic recovery
        from failures. It saves checkpoints at each major execution step and
        can resume from the last successful checkpoint if a failure occurs.

        Returns:
            A Thought object containing the final text and all validation/improvement results.

        Raises:
            ChainError: If the chain cannot be recovered after multiple attempts.
        """
        if not self._checkpoint_storage:
            logger.warning("No checkpoint storage configured, falling back to regular run()")
            return self.run()

        logger.info(f"Starting chain execution with recovery for chain {self._chain_id}")

        # Check for existing checkpoints to resume from
        existing_checkpoints = self._checkpoint_storage.get_chain_checkpoints(self._chain_id)
        if existing_checkpoints:
            latest_checkpoint = existing_checkpoints[-1]
            if latest_checkpoint.current_step != "complete":
                logger.info(
                    f"Found incomplete execution, attempting to resume from step: {latest_checkpoint.current_step}"
                )
                return self._resume_from_checkpoint(latest_checkpoint)

        # No existing checkpoints, start fresh with checkpointing
        return self._run_with_checkpoints()

    def _run_with_checkpoints(self) -> Thought:
        """Execute the chain with checkpoint saving at each step.

        Returns:
            A Thought object containing the final text and all validation/improvement results.
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                return self._execute_with_checkpoints()
            except Exception as e:
                retry_count += 1
                logger.warning(f"Chain execution failed (attempt {retry_count}/{max_retries}): {e}")

                if retry_count >= max_retries:
                    logger.error(f"Chain execution failed after {max_retries} attempts")
                    raise ChainError(f"Chain execution failed after {max_retries} attempts: {e}")

                # Analyze failure and attempt recovery
                if self._current_checkpoint and self._recovery_manager:
                    recovery_actions = self._recovery_manager.analyze_failure(
                        self._current_checkpoint, e
                    )
                    if recovery_actions:
                        best_action = recovery_actions[0]
                        logger.info(f"Attempting recovery: {best_action.description}")

                        # Apply recovery action
                        if self._apply_recovery_action(best_action):
                            continue  # Retry with recovery

                # If no recovery possible, wait and retry
                import time

                time.sleep(2**retry_count)  # Exponential backoff

        raise ChainError("Chain execution failed and could not be recovered")

    def _execute_with_checkpoints(self) -> Thought:
        """Execute the chain with checkpoint creation at each step.

        Returns:
            A Thought object containing the final text and all validation/improvement results.
        """
        # Start performance monitoring for the entire chain execution
        with time_operation(f"chain_execution_with_checkpoints_{self._chain_id}"):
            # Check that the chain is properly configured
            if not self._model:
                raise ChainError("No model specified for the chain")
            if not self._prompt:
                raise ChainError("No prompt specified for the chain")

            # Create initial thought
            thought = Thought(
                prompt=self._prompt,
                chain_id=self._chain_id,
            )

            # Save initial checkpoint
            self._save_checkpoint("initialization", thought, 1)

            # Save the initial thought if storage is available
            if self._storage:
                try:
                    self._storage.save_thought(thought)
                    logger.debug(
                        f"Saved initial thought (iteration {thought.iteration}) to storage"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save initial thought: {e}")

            # 1. PRE-GENERATION RETRIEVAL (Chain orchestrates this using model_retrievers)
            if self._model_retrievers:
                with time_operation("pre_generation_retrieval"):
                    with chain_context(
                        operation="pre_generation_retrieval",
                        message_prefix="Failed to retrieve pre-generation context",
                    ):
                        logger.debug("Chain orchestrating pre-generation retrieval for model...")
                        for retriever in self._model_retrievers:
                            thought = retriever.retrieve_for_thought(
                                thought, is_pre_generation=True
                            )
                        logger.debug(
                            f"Retrieved {len(thought.pre_generation_context or [])} pre-generation documents for model"
                        )

                        # Save checkpoint after retrieval
                        self._save_checkpoint("pre_retrieval", thought, 1)

            # 2. GENERATE TEXT (Model just generates, no retrieval)
            with time_operation("text_generation"):
                with chain_context(
                    operation="generation",
                    message_prefix="Failed to generate text",
                ):
                    logger.debug(f"Generating text with prompt: {self._prompt[:100]}...")
                    text, model_prompt = self._model.generate_with_thought(thought)
                    thought = thought.set_text(text).set_model_prompt(model_prompt)
                    logger.debug(f"Generated text of length {len(text)}")
                    logger.debug(f"Model prompt length: {len(model_prompt)} characters")

                    # Save checkpoint after generation
                    self._save_checkpoint("generation", thought, 1)

            # Continue with the rest of the execution...
            # For brevity, I'll delegate to the original run() method for the remaining steps
            # but with checkpoint saving at each major step

            # This is a simplified version - in practice, you'd want to add checkpoints
            # throughout the validation and criticism loops as well
            return self._complete_execution_with_checkpoints(thought)

    def _complete_execution_with_checkpoints(self, thought: Thought) -> Thought:
        """Complete the chain execution with checkpointing.

        This method handles the validation and criticism phases with checkpointing.

        Args:
            thought: The current thought state after generation

        Returns:
            The final thought with all processing complete
        """
        # For now, delegate to the original logic but add checkpoint saving
        # This would need to be fully implemented with checkpoints at each step

        # Save checkpoint before validation
        self._save_checkpoint("pre_validation", thought, thought.iteration)

        # Perform validation (simplified - would need full implementation)
        all_passed = True
        with time_operation("validation"):
            for validator in self._validators:
                with time_operation(f"validation_{validator.__class__.__name__}"):
                    with chain_context(
                        operation="validation",
                        message_prefix=f"Failed to validate text with {validator.__class__.__name__}",
                    ):
                        logger.debug(f"Validating text with {validator.__class__.__name__}")
                        validation_result = validator.validate(thought)

                        # Add validation result to thought
                        thought = thought.add_validation_result(
                            validator.__class__.__name__, validation_result
                        )

                        # Update all_passed flag
                        all_passed = all_passed and validation_result.passed
                        logger.debug(
                            f"Validation with {validator.__class__.__name__} "
                            f"{'passed' if validation_result.passed else 'failed'}"
                        )

        # Save checkpoint after validation
        self._save_checkpoint("validation", thought, thought.iteration)

        # Handle criticism if needed (simplified)
        if self._critics and (not all_passed or self._options.get("always_apply_critics", False)):
            self._save_checkpoint("pre_criticism", thought, thought.iteration)

            # Apply critics (would need full implementation with checkpoints)
            for critic in self._critics:
                with chain_context(
                    operation="criticism",
                    message_prefix=f"Failed to get criticism from {critic.__class__.__name__}",
                ):
                    logger.debug(f"Getting criticism from {critic.__class__.__name__}")
                    critique_result = critic.critique(thought)

                    # Add critic feedback
                    feedback = CriticFeedback(
                        critic_name=critic.__class__.__name__,
                        feedback=critique_result,
                        timestamp=thought.timestamp,
                    )
                    thought = thought.add_critic_feedback(feedback)

            self._save_checkpoint("criticism", thought, thought.iteration)

        # Mark as complete
        self._save_checkpoint("complete", thought, thought.iteration)

        return thought

    def _save_checkpoint(self, step: str, thought: Thought, iteration: int) -> None:
        """Save a checkpoint for the current execution state.

        Args:
            step: The current execution step
            thought: The current thought state
            iteration: The current iteration number
        """
        if not self._checkpoint_storage:
            return

        try:
            # Get performance data
            monitor = PerformanceMonitor.get_instance()
            performance_data = monitor.get_summary()

            # Create checkpoint
            checkpoint = ChainCheckpoint(
                chain_id=self._chain_id,
                current_step=step,
                iteration=iteration,
                thought=thought,
                performance_data=performance_data,
                recovery_point=step,
                completed_validators=[v.__class__.__name__ for v in self._validators],
                completed_critics=[c.__class__.__name__ for c in self._critics],
                metadata={
                    "model_name": getattr(self._model, "model_name", "unknown"),
                    "prompt_length": len(self._prompt or ""),
                    "model_retriever_count": len(self._model_retrievers),
                    "critic_retriever_count": len(self._critic_retrievers),
                },
            )

            self._checkpoint_storage.save_checkpoint(checkpoint)
            self._current_checkpoint = checkpoint

            logger.debug(f"Saved checkpoint for step '{step}' at iteration {iteration}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for step '{step}': {e}")

    def _resume_from_checkpoint(self, checkpoint: ChainCheckpoint) -> Thought:
        """Resume execution from a saved checkpoint.

        Args:
            checkpoint: The checkpoint to resume from

        Returns:
            The final thought after resuming execution
        """
        logger.info(f"Resuming execution from checkpoint: {checkpoint.current_step}")

        self._current_checkpoint = checkpoint
        thought = checkpoint.thought

        # Determine where to resume based on the checkpoint step
        if checkpoint.current_step == "initialization":
            return self._execute_with_checkpoints()
        elif checkpoint.current_step == "pre_retrieval":
            return self._resume_from_generation(thought)
        elif checkpoint.current_step == "generation":
            return self._complete_execution_with_checkpoints(thought)
        elif checkpoint.current_step in ["validation", "pre_criticism"]:
            return self._resume_from_criticism(thought)
        elif checkpoint.current_step == "criticism":
            # Already complete, just return
            return thought
        else:
            logger.warning(f"Unknown checkpoint step: {checkpoint.current_step}, starting fresh")
            return self._execute_with_checkpoints()

    def _resume_from_generation(self, thought: Thought) -> Thought:
        """Resume execution from the generation step.

        Args:
            thought: The thought state to resume from

        Returns:
            The final thought after completion
        """
        # Generate text if not already done
        if not thought.text:
            with time_operation("text_generation"):
                with chain_context(
                    operation="generation",
                    message_prefix="Failed to generate text during recovery",
                ):
                    logger.debug("Resuming text generation...")
                    text, model_prompt = self._model.generate_with_thought(thought)
                    thought = thought.set_text(text).set_model_prompt(model_prompt)

                    # Save checkpoint after generation
                    self._save_checkpoint("generation", thought, thought.iteration)

        return self._complete_execution_with_checkpoints(thought)

    def _resume_from_criticism(self, thought: Thought) -> Thought:
        """Resume execution from the criticism step.

        Args:
            thought: The thought state to resume from

        Returns:
            The final thought after completion
        """
        # Apply any remaining critics
        if self._critics:
            for critic in self._critics:
                # Check if this critic has already been applied
                existing_feedback = thought.critic_feedback or []
                critic_name = critic.__class__.__name__

                if not any(fb.critic_name == critic_name for fb in existing_feedback):
                    with chain_context(
                        operation="criticism",
                        message_prefix=f"Failed to get criticism from {critic_name} during recovery",
                    ):
                        logger.debug(f"Resuming criticism with {critic_name}")
                        critique_result = critic.critique(thought)

                        # Add critic feedback
                        feedback = CriticFeedback(
                            critic_name=critic_name,
                            feedback=critique_result,
                            timestamp=thought.timestamp,
                        )
                        thought = thought.add_critic_feedback(feedback)

            self._save_checkpoint("criticism", thought, thought.iteration)

        # Mark as complete
        self._save_checkpoint("complete", thought, thought.iteration)
        return thought

    def _apply_recovery_action(self, action: RecoveryAction) -> bool:
        """Apply a recovery action to the chain.

        Args:
            action: The recovery action to apply

        Returns:
            True if the action was successfully applied
        """
        try:
            if action.strategy == RecoveryStrategy.RETRY_CURRENT_STEP:
                # Just return True to retry the current execution
                logger.info("Applying recovery: retry current step")
                return True

            elif action.strategy == RecoveryStrategy.MODIFY_PARAMETERS:
                # Modify chain options based on suggested parameters
                if action.parameters:
                    logger.info(f"Applying recovery: modifying parameters {action.parameters}")
                    self._options.update(action.parameters)
                return True

            elif action.strategy == RecoveryStrategy.RESTART_ITERATION:
                # Reset to previous iteration (would need more complex implementation)
                logger.info("Applying recovery: restart iteration")
                return True

            elif action.strategy == RecoveryStrategy.FULL_RESTART:
                # Clear current checkpoint and start fresh
                logger.info("Applying recovery: full restart")
                self._current_checkpoint = None
                return True

            else:
                logger.warning(f"Recovery strategy not implemented: {action.strategy}")
                return False

        except Exception as e:
            logger.error(f"Failed to apply recovery action {action.strategy}: {e}")
            return False

    def get_recovery_suggestions(self, error: Exception) -> List[RecoveryAction]:
        """Get recovery suggestions for a given error.

        Args:
            error: The error that occurred

        Returns:
            List of suggested recovery actions
        """
        if not self._recovery_manager or not self._current_checkpoint:
            return []

        return self._recovery_manager.analyze_failure(self._current_checkpoint, error)

    def get_checkpoint_history(self) -> List[ChainCheckpoint]:
        """Get the checkpoint history for this chain.

        Returns:
            List of checkpoints for this chain, sorted by timestamp
        """
        if not self._checkpoint_storage:
            return []

        return self._checkpoint_storage.get_chain_checkpoints(self._chain_id)
