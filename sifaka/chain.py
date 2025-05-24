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

    The Chain orchestrates ALL retrieval operations:
    - Pre-generation retrieval: Gets context before text generation
    - Post-generation retrieval: Gets context after text generation for validation/improvement
    - Critic retrieval: Gets additional context during critic operations

    It follows a fluent interface pattern (builder pattern) for easy configuration,
    allowing you to chain method calls to set up the desired behavior.

    The typical workflow is:
    1. Create a Chain instance with model, prompt, and optional retriever
    2. Configure it with validators and critics
    3. Run the chain - it handles ALL retrieval automatically
    4. Process the result

    Attributes:
        model: The language model to use for text generation.
        prompt: The prompt to use for text generation.
        retriever: The retriever to use for context (orchestrated by Chain).
        validators: List of validators to check if the generated text meets requirements.
        critics: List of critics to improve the generated text.
        options: Dictionary of options for the chain.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt: Optional[str] = None,
        retrievers: Optional[List[Retriever]] = None,
        storage: Optional[Any] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
        always_apply_critics: bool = False,
    ):
        """Initialize the Chain with optional model, prompt, and retrievers.

        Args:
            model: Optional language model to use for text generation.
            prompt: Optional prompt to use for text generation.
            retrievers: Optional list of retrievers to use for context retrieval.
            storage: Optional storage for saving intermediate thoughts.
            max_improvement_iterations: Maximum number of improvement iterations.
            apply_improvers_on_validation_failure: Whether to apply improvers when validation fails.
            always_apply_critics: Whether to always apply critics regardless of validation status.
        """
        self._model = model
        self._prompt = prompt
        self._retrievers = retrievers or []
        self._storage = storage

        self._validators: List[Validator] = []
        self._critics: List[Critic] = []
        self._options: Dict[str, Any] = {
            "max_improvement_iterations": max_improvement_iterations,
            "apply_improvers_on_validation_failure": apply_improvers_on_validation_failure,
            "always_apply_critics": always_apply_critics,
        }
        self._chain_id = str(uuid.uuid4())

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

            # 1. PRE-GENERATION RETRIEVAL (Chain orchestrates this using retrievers)
            if self._retrievers:
                with time_operation("pre_generation_retrieval"):
                    with chain_context(
                        operation="pre_generation_retrieval",
                        message_prefix="Failed to retrieve pre-generation context",
                    ):
                        logger.debug("Chain orchestrating pre-generation retrieval...")
                        for retriever in self._retrievers:
                            thought = retriever.retrieve_for_thought(
                                thought, is_pre_generation=True
                            )
                        logger.debug(
                            f"Retrieved {len(thought.pre_generation_context or [])} pre-generation documents"
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

            # 3. POST-GENERATION RETRIEVAL (Chain orchestrates this using retrievers)
            if self._retrievers:
                with time_operation("post_generation_retrieval"):
                    with chain_context(
                        operation="post_generation_retrieval",
                        message_prefix="Failed to retrieve post-generation context",
                    ):
                        logger.debug("Chain orchestrating post-generation retrieval...")
                        for retriever in self._retrievers:
                            thought = retriever.retrieve_for_thought(
                                thought, is_pre_generation=False
                            )
                        logger.debug(
                            f"Retrieved {len(thought.post_generation_context or [])} post-generation documents"
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

                # 5a. CRITIC RETRIEVAL (Chain orchestrates this using retrievers)
                if self._retrievers:
                    with time_operation("critic_retrieval"):
                        with chain_context(
                            operation="critic_retrieval",
                            message_prefix="Failed to retrieve context for critics",
                        ):
                            logger.debug("Chain orchestrating critic retrieval...")
                            # For critics, we might want fresh post-generation context
                            for retriever in self._retrievers:
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
