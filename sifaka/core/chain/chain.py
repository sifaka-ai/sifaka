"""Simplified Chain class using modular architecture.

This module contains the refactored Chain class that uses the new modular
architecture with separated concerns for configuration, orchestration,
execution, and recovery.

The Chain supports both sync and async execution internally, with sync methods
wrapping async implementations using asyncio.run() for backward compatibility.
"""

import asyncio
from typing import Any, Dict, List, Optional

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
        """
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
            The chain instance for method chaining.
        """
        self._config.add_validator(validator)
        return self

    def improve_with(self, critic: Critic) -> "Chain":
        """Add a critic to the chain.

        Args:
            critic: The critic to improve the generated text.

        Returns:
            The chain instance for method chaining.
        """
        self._config.add_critic(critic)
        return self

    def with_options(self, **options: Any) -> "Chain":
        """Set options for the chain.

        Args:
            **options: Options to set for the chain.

        Returns:
            The chain instance for method chaining.
        """
        self._config.update_options(**options)
        return self

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
            # If we get here, we're in an async context, so we need to run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._run_async())
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self._run_async())

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

            # Save initial thought
            self._executor._save_thought_to_storage(thought)

            # Pre-generation retrieval
            thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

            # Generate text (async)
            thought = await self._execute_generation_async(thought)

            # Post-generation retrieval
            thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")

            # Validation (async with concurrent validators)
            thought, validation_passed = await self._executor._execute_validation_async(thought)

            # Improvement loop (async with concurrent critics)
            thought = await self._execute_improvement_loop_async(thought, validation_passed)

            # Save final thought
            self._executor._save_thought_to_storage(thought)

            return thought

    async def _execute_generation_async(self, thought: Thought) -> Thought:
        """Execute text generation asynchronously."""
        if not self._config.model:
            raise ChainError("No model configured for generation")

        # Generate text using the model's async method (no options like sync version)
        generated_text, actual_prompt = await self._config.model._generate_with_thought_async(
            thought
        )

        # Update thought with generated text and prompt
        thought = thought.set_text(generated_text)
        thought = thought.set_model_prompt(actual_prompt)

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

    async def _execute_improvement_iteration_async(self, thought: Thought) -> Thought:
        """Execute a single improvement iteration with criticism and regeneration asynchronously."""
        logger.debug(f"Starting async improvement iteration {thought.iteration}")

        # Apply critic retrieval if configured
        thought = self._orchestrator.orchestrate_retrieval(thought, "critic")

        # Apply critics to get feedback (async with concurrent critics)
        thought = await self._executor._execute_criticism_async(thought)

        # Create next iteration with feedback for the model to see
        thought = thought.next_iteration()

        # Apply pre-generation retrieval for the new iteration
        thought = self._orchestrator.orchestrate_retrieval(thought, "pre_generation")

        # Generate improved text (async)
        thought = await self._execute_generation_async(thought)

        # Apply post-generation retrieval
        thought = self._orchestrator.orchestrate_retrieval(thought, "post_generation")

        # Save the improved thought to storage
        self._executor._save_thought_to_storage(thought)

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
