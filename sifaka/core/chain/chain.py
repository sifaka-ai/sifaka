"""Simplified Chain class using modular architecture.

This module contains the refactored Chain class that uses the new modular
architecture with separated concerns for configuration, orchestration,
execution, and recovery.

The Chain supports both sync and async execution internally, with sync methods
wrapping async implementations using asyncio.run() for backward compatibility.
"""

import asyncio
from typing import Any, List, Optional

from sifaka.core.chain.config import ChainConfig
from sifaka.core.chain.executor import ChainExecutor
from sifaka.core.chain.orchestrator import ChainOrchestrator
from sifaka.core.chain.recovery import RecoveryManager
from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.core.thought import Thought
from sifaka.storage.checkpoints import CachedCheckpointStorage, ChainCheckpoint
from sifaka.storage.protocol import Storage
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class Chain:
    """Main orchestrator for text generation, validation, and improvement.

    This is the simplified Chain class that provides a fluent API interface
    while delegating actual work to specialized components:
    - ChainConfig: Manages configuration state
    - ChainOrchestrator: Coordinates high-level workflow
    - ChainExecutor: Handles low-level execution
    - RecoveryManager: Manages checkpointing and recovery

    The Chain maintains the same fluent interface as before but with
    much cleaner separation of concerns.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt: Optional[str] = None,
        model_retrievers: Optional[List[Retriever]] = None,
        critic_retrievers: Optional[List[Retriever]] = None,
        storage: Optional[Storage] = None,
        checkpoint_storage: Optional[CachedCheckpointStorage] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
        always_apply_critics: bool = False,
        # Additional options that tests expect
        max_iterations: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        retriever: Optional[Retriever] = None,  # For backward compatibility
        **kwargs: Any,
    ):
        """Initialize the Chain with configuration.

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
            max_iterations: Alternative name for max_improvement_iterations (for backward compatibility).
            temperature: Temperature for text generation.
            max_tokens: Maximum tokens for text generation.
            top_p: Top-p for text generation.
            retriever: Single retriever (for backward compatibility).
            **kwargs: Additional options.
        """
        # Handle backward compatibility for max_iterations
        if max_iterations is not None:
            max_improvement_iterations = max_iterations

        # Handle backward compatibility for single retriever
        if retriever is not None:
            if model_retrievers is None:
                model_retrievers = [retriever]
            else:
                model_retrievers = list(model_retrievers) + [retriever]

        # Create configuration
        self._config = ChainConfig(
            model=model,
            prompt=prompt,
            model_retrievers=model_retrievers,
            critic_retrievers=critic_retrievers,
            storage=storage,
            checkpoint_storage=checkpoint_storage,
            max_improvement_iterations=max_improvement_iterations,
            apply_improvers_on_validation_failure=apply_improvers_on_validation_failure,
            always_apply_critics=always_apply_critics,
        )

        # Add additional options
        additional_options = {}
        if temperature is not None:
            additional_options["temperature"] = temperature
        if max_tokens is not None:
            additional_options["max_tokens"] = max_tokens
        if top_p is not None:
            additional_options["top_p"] = top_p

        # Add any other kwargs as options
        additional_options.update(kwargs)

        if additional_options:
            self._config.update_options(**additional_options)

        # Create specialized components
        self._orchestrator = ChainOrchestrator(self._config)
        self._executor = ChainExecutor(self._config, self._orchestrator)
        self._recovery_manager = RecoveryManager(self._config) if checkpoint_storage else None

    # Fluent API methods for configuration

    def with_model(self, model: Model) -> "Chain":
        """Set the model for the chain.

        Args:
            model: The model to use for text generation.

        Returns:
            The chain instance for method chaining.
        """
        self._config.set_model(model)
        return self

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt for the chain.

        Args:
            prompt: The prompt to use for text generation.

        Returns:
            The chain instance for method chaining.
        """
        self._config.set_prompt(prompt)
        return self

    def with_model_retrievers(self, retrievers: List[Retriever]) -> "Chain":
        """Set retrievers specifically for model context.

        Args:
            retrievers: List of retrievers to use for model context.

        Returns:
            The chain instance for method chaining.
        """
        self._config.set_model_retrievers(retrievers)
        return self

    def with_critic_retrievers(self, retrievers: List[Retriever]) -> "Chain":
        """Set retrievers specifically for critic context.

        Args:
            retrievers: List of retrievers to use for critic context.

        Returns:
            The chain instance for method chaining.
        """
        self._config.set_critic_retrievers(retrievers)
        return self

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain.

        Args:
            validator: The validator to check if the generated text meets requirements.

        Returns:
            A new chain instance with the validator added.
        """
        # Create a copy of the configuration
        new_config = self._config.copy()
        new_config.add_validator(validator)

        # Create a new chain with the copied configuration
        new_chain = Chain.__new__(Chain)
        new_chain._config = new_config
        new_chain._orchestrator = ChainOrchestrator(new_config)
        new_chain._executor = ChainExecutor(new_config, new_chain._orchestrator)
        new_chain._recovery_manager = (
            RecoveryManager(new_config) if new_config.checkpoint_storage else None
        )

        return new_chain

    def improve_with(self, critic: Critic) -> "Chain":
        """Add a critic to the chain.

        Args:
            critic: The critic to improve the generated text.

        Returns:
            A new chain instance with the critic added.
        """
        # Create a copy of the configuration
        new_config = self._config.copy()
        new_config.add_critic(critic)

        # Create a new chain with the copied configuration
        new_chain = Chain.__new__(Chain)
        new_chain._config = new_config
        new_chain._orchestrator = ChainOrchestrator(new_config)
        new_chain._executor = ChainExecutor(new_config, new_chain._orchestrator)
        new_chain._recovery_manager = (
            RecoveryManager(new_config) if new_config.checkpoint_storage else None
        )

        return new_chain

    def with_options(self, **options: Any) -> "Chain":
        """Set options for the chain.

        Args:
            **options: Options to set for the chain.

        Returns:
            A new chain instance with the options updated.
        """
        # Create a copy of the configuration
        new_config = self._config.copy()
        new_config.update_options(**options)

        # Create a new chain with the copied configuration
        new_chain = Chain.__new__(Chain)
        new_chain._config = new_config
        new_chain._orchestrator = ChainOrchestrator(new_config)
        new_chain._executor = ChainExecutor(new_config, new_chain._orchestrator)
        new_chain._recovery_manager = (
            RecoveryManager(new_config) if new_config.checkpoint_storage else None
        )

        return new_chain

    # Properties for backward compatibility
    @property
    def config(self) -> "ChainConfig":
        """Get the chain configuration for backward compatibility."""
        return self._config

    # Execution methods

    def run(self) -> Thought:
        """Execute the chain and return the result.

        This method runs the complete chain execution process:
        1. Validates configuration
        2. Pre-generation retrieval
        3. Text generation
        4. Post-generation retrieval
        5. Validation
        6. Improvement loop (if needed)

        The method uses async implementation internally for better performance
        while maintaining the same synchronous API for backward compatibility.

        Returns:
            A Thought object containing the final text and all results.

        Raises:
            ChainError: If the chain is not properly configured or execution fails.
        """
        # Check if we're already in an async context
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context, run in thread pool
            return asyncio.run_coroutine_threadsafe(self._run_async(), loop).result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self._run_async())

    async def run_async(self) -> Thought:
        """Execute the chain asynchronously (public async API).

        This method provides a public async interface for chain execution.

        Returns:
            A Thought object containing the final text and all results.

        Raises:
            ChainError: If the chain is not properly configured or execution fails.
        """
        return await self._run_async()

    async def _run_async(self) -> Thought:
        """Execute the chain asynchronously and return the result.

        This is the internal async implementation that provides the same functionality
        as the sync run method but with concurrent validation and criticism.

        Returns:
            A Thought object containing the final text and all results.

        Raises:
            ChainError: If the chain is not properly configured or execution fails.
        """
        with time_operation(f"chain_execution_async_{self._config.chain_id}"):
            # Validate configuration
            self._config.validate()

            # Create initial thought
            thought = Thought(
                prompt=self._config.prompt,
                chain_id=self._config.chain_id,
            )

            # Skip saving the initial empty thought - we'll save after generation

            # Pre-generation retrieval
            thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

            # Generate text (async)
            thought = await self._execute_generation_async(thought)

            # Set iteration to 0 for the first meaningful thought (with text and model_prompt)
            thought = thought.model_copy(update={"iteration": 0})

            # Save the first meaningful thought
            await self._executor._save_thought_to_storage_async(thought)

            # Post-generation retrieval
            thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")

            # Validation (async with concurrent validators)
            thought, validation_passed = await self._executor._execute_validation_async(thought)

            # Retry loop for validation failures (without critics)
            if not validation_passed and not self._config.critics:
                thought = await self._execute_retry_loop_async(thought)
            else:
                # Improvement loop (async with concurrent critics)
                thought = await self._execute_improvement_loop_async(thought, validation_passed)

            # Save final thought
            await self._executor._save_thought_to_storage_async(thought)

            return thought

    async def _execute_generation_async(self, thought: Thought) -> Thought:
        """Execute text generation asynchronously."""
        if not self._config.model:
            raise ChainError("No model configured for generation")

        # Generate text using the model's async method (no options like sync version)
        generated_text, actual_prompt = await self._config.model._generate_with_thought_async(
            thought
        )

        # Update thought with generated text, prompt, and model name
        thought = thought.set_text(generated_text)
        thought = thought.set_model_prompt(actual_prompt)
        thought = thought.model_copy(update={"model_name": self._config.model.model_name})

        logger.debug(f"Generated text (async): {len(generated_text)} characters")
        return thought

    async def _execute_improvement_loop_async(
        self, thought: Thought, validation_passed: bool
    ) -> Thought:
        """Execute the complete improvement loop with multiple iterations asynchronously."""
        if not self._orchestrator.should_apply_critics(validation_passed):
            logger.debug("Skipping improvement loop based on configuration and validation results")
            return thought

        max_iterations = self._orchestrator.get_max_iterations()
        current_iteration = 0

        logger.debug(f"Starting async improvement loop (max {max_iterations} iterations)")

        while current_iteration < max_iterations:
            current_iteration += 1
            logger.debug(f"Async improvement iteration {current_iteration}/{max_iterations}")

            # Execute improvement iteration asynchronously
            thought = await self._execute_improvement_iteration_async(thought)

            # Re-validate after improvement (async)
            thought, validation_passed = await self._executor._execute_validation_async(thought)

            # If validation passes and we're not always applying critics, we can stop
            if validation_passed and not self._config.get_option("always_apply_critics", False):
                logger.debug(
                    f"Validation passed after async iteration {current_iteration}, stopping improvement loop"
                )
                break

        logger.debug(f"Completed async improvement loop after {current_iteration} iterations")
        return thought

    async def _execute_retry_loop_async(self, thought: Thought) -> Thought:
        """Execute retry loop for validation failures without critics."""
        max_iterations = self._orchestrator.get_max_iterations()
        current_iteration = thought.iteration

        logger.debug(
            f"Starting retry loop for validation failure (max {max_iterations} iterations)"
        )

        while current_iteration < max_iterations:
            current_iteration += 1
            logger.debug(f"Retry iteration {current_iteration}/{max_iterations}")

            # Create next iteration for retry
            thought = thought.model_copy(update={"iteration": current_iteration})

            # Apply pre-generation retrieval for the retry
            thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

            # Generate text again
            thought = await self._execute_generation_async(thought)

            # Apply post-generation retrieval
            thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")

            # Re-validate
            thought, validation_passed = await self._executor._execute_validation_async(thought)

            # If validation passes, we can stop retrying
            if validation_passed:
                logger.debug(
                    f"Validation passed after retry iteration {current_iteration}, stopping retry loop"
                )
                break

        logger.debug(f"Completed retry loop after {current_iteration} iterations")
        return thought

    async def _execute_improvement_iteration_async(self, thought: Thought) -> Thought:
        """Execute a single improvement iteration with criticism and regeneration asynchronously."""
        logger.debug(f"Starting async improvement iteration {thought.iteration}")

        # Apply critic retrieval if configured
        thought = self._orchestrator.orchestrate_retrieval(thought, "critic")

        # Apply critics to get feedback (async with concurrent critics)
        thought = await self._executor._execute_criticism_async(thought)

        # Save the thought with critic feedback BEFORE creating next iteration
        await self._executor._save_thought_to_storage_async(thought)

        # Create next iteration with feedback for the model to see
        thought = thought.next_iteration()

        # Apply pre-generation retrieval for the new iteration
        thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

        # Generate improved text using critics if feedback is available (async)
        thought = await self._execute_improvement_generation_async(thought)

        # Apply post-generation retrieval
        thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")

        # Save the improved thought to storage
        await self._executor._save_thought_to_storage_async(thought)

        logger.debug(f"Completed async improvement iteration {thought.iteration}")
        return thought

    def run_with_recovery(self) -> Thought:
        """Execute the chain with automatic checkpointing and recovery.

        Returns:
            A Thought object containing the final text and all results.

        Raises:
            ChainError: If the chain cannot be recovered after multiple attempts.
        """
        if not self._recovery_manager:
            logger.warning("No recovery manager configured, falling back to regular run()")
            return self.run()

        logger.info(f"Starting chain execution with recovery for chain {self._config.chain_id}")

        # Check for existing checkpoints
        if self._recovery_manager.can_resume():
            latest_checkpoint = self._recovery_manager.get_latest_checkpoint()
            if latest_checkpoint and latest_checkpoint.current_step != "complete":
                logger.info(f"Resuming from checkpoint: {latest_checkpoint.current_step}")
                return self._resume_from_checkpoint(latest_checkpoint)

        # No existing checkpoints, start fresh with checkpointing
        return self._run_with_checkpoints()

    def _run_with_checkpoints(self) -> Thought:
        """Execute the chain with checkpoint saving at each step."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                return self._execute_with_checkpoints()
            except Exception as e:
                retry_count += 1
                logger.warning(f"Chain execution failed (attempt {retry_count}/{max_retries}): {e}")

                if retry_count >= max_retries:
                    raise ChainError(f"Chain execution failed after {max_retries} attempts: {e}")

                # Apply recovery if possible
                if self._recovery_manager and self._recovery_manager.current_checkpoint:
                    recovery_actions = self._recovery_manager.analyze_failure(
                        self._recovery_manager.current_checkpoint, e
                    )
                    if recovery_actions:
                        best_action = recovery_actions[0]
                        logger.info(f"Attempting recovery: {best_action.description}")
                        if self._recovery_manager.apply_recovery_action(best_action):
                            continue

                # Exponential backoff
                import time

                time.sleep(2**retry_count)

        raise ChainError("Chain execution failed and could not be recovered")

    def _execute_with_checkpoints(self) -> Thought:
        """Execute the chain with checkpoint creation at each step."""
        # Validate configuration
        self._config.validate()

        # Create initial thought
        thought = Thought(
            prompt=self._config.prompt,
            chain_id=self._config.chain_id,
        )

        # Save checkpoint
        if self._recovery_manager:
            self._recovery_manager.save_checkpoint("initialization", thought, 1)

        # Execute with checkpointing
        thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

        if self._recovery_manager:
            self._recovery_manager.save_checkpoint("pre_retrieval", thought, thought.iteration)

        thought = self._executor.execute_generation(thought)

        if self._recovery_manager:
            self._recovery_manager.save_checkpoint("generation", thought, thought.iteration)

        thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")
        thought, validation_passed = self._executor.execute_validation(thought)

        if self._recovery_manager:
            self._recovery_manager.save_checkpoint("validation", thought, thought.iteration)

        thought = self._executor.execute_improvement_loop(thought, validation_passed)

        if self._recovery_manager:
            self._recovery_manager.save_checkpoint("complete", thought, thought.iteration)

        return thought

    async def _execute_improvement_generation_async(self, thought: Thought) -> Thought:
        """Execute text generation using critics' improve methods when feedback is available (async).

        Args:
            thought: The current thought state with potential critic feedback.

        Returns:
            Updated thought with improved text.
        """
        # Check if we have critic feedback from the previous iteration
        if thought.critic_feedback and len(thought.critic_feedback) > 0:
            logger.debug(
                f"Using critic improvement methods for {len(thought.critic_feedback)} critics (async)"
            )

            # Use the first critic that has feedback and can improve
            for feedback in thought.critic_feedback:
                if feedback.needs_improvement:
                    # Find the corresponding critic
                    for critic in self._config.critics:
                        if critic.__class__.__name__ == feedback.critic_name:
                            try:
                                logger.debug(
                                    f"Using {feedback.critic_name} to improve text (async)"
                                )

                                # Get the original text from the previous iteration (from history)
                                original_text = None
                                if thought.history and len(thought.history) > 0:
                                    # Try to get the original text from the most recent history entry
                                    previous_thought_id = thought.history[0].thought_id
                                    try:
                                        # Try to get the previous thought from storage to get its text
                                        previous_thought = await self._config.storage._get_async(
                                            previous_thought_id
                                        )
                                        if (
                                            previous_thought
                                            and hasattr(previous_thought, "text")
                                            and previous_thought.text
                                        ):
                                            original_text = previous_thought.text
                                            logger.debug(
                                                f"Retrieved original text from previous iteration (async): {len(original_text)} characters"
                                            )
                                    except Exception as e:
                                        logger.debug(
                                            f"Could not retrieve previous thought text (async): {e}"
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
                                        "No original text available for critic improvement (async), falling back to regular generation"
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
                                    update={"model_name": self._config.model.model_name}
                                )

                                logger.debug(
                                    f"Generated improved text using {feedback.critic_name} (async): {len(improved_text)} characters"
                                )
                                return thought

                            except Exception as e:
                                logger.error(
                                    f"Failed to improve text using {feedback.critic_name} (async): {e}"
                                )
                                # Fall back to regular generation
                                break

            # If no critic could improve or all failed, fall back to regular generation
            logger.debug("No critic could improve text, falling back to regular generation (async)")

        # Fall back to regular generation when no critic feedback or improvement needed
        return await self._execute_generation_async(thought)

    def _resume_from_checkpoint(self, checkpoint: ChainCheckpoint) -> Thought:
        """Resume execution from a saved checkpoint."""
        # This would implement checkpoint resumption logic
        # For now, fall back to regular execution
        logger.warning(
            "Checkpoint resumption not fully implemented, falling back to regular execution"
        )
        return self.run()

    # Properties for clean API access

    @property
    def chain_id(self) -> str:
        """Get the chain ID."""
        return self._config.chain_id

    @property
    def model(self) -> Optional[Model]:
        """Get the model."""
        return self._config.model

    @property
    def prompt(self) -> Optional[str]:
        """Get the prompt."""
        return self._config.prompt

    @property
    def validators(self) -> List[Validator]:
        """Get the list of validators."""
        return self._config.validators

    @property
    def critics(self) -> List[Critic]:
        """Get the list of critics."""
        return self._config.critics

    @property
    def storage(self) -> Storage:
        """Get the storage."""
        return self._config.storage

    @property
    def max_iterations(self) -> int:
        """Get the maximum improvement iterations."""
        return self._config.get_option("max_improvement_iterations", 3)

    @property
    def temperature(self) -> Optional[float]:
        """Get the temperature option."""
        return self._config.get_option("temperature")

    @property
    def max_tokens(self) -> Optional[int]:
        """Get the max_tokens option."""
        return self._config.get_option("max_tokens")

    @property
    def top_p(self) -> Optional[float]:
        """Get the top_p option."""
        return self._config.get_option("top_p")

    @property
    def retriever(self) -> Optional[Retriever]:
        """Get the first model retriever (for backward compatibility)."""
        if self._config.model_retrievers:
            return self._config.model_retrievers[0]
        return None
