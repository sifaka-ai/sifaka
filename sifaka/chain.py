"""Chain orchestration for Sifaka.

This module defines the Chain class, which is the main entry point for the Sifaka framework.
It orchestrates the process of generating text using language models, validating the text
against specified criteria, and improving the text using specialized critics.

The Chain class uses a builder pattern to provide a fluent API for configuring and
executing LLM operations. It allows you to specify which model to use, set the prompt
for generation, add validators to check if the generated text meets requirements,
add critics to enhance the quality of the generated text, and configure model options.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from sifaka.core.interfaces import Critic, Model, Retriever, Validator
from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ChainError, chain_context, log_error
from sifaka.utils.logging import get_logger

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
        retriever: Optional[Retriever] = None,
        model_retriever: Optional[Retriever] = None,
        critic_retriever: Optional[Retriever] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
        pre_generation_retrieval: bool = True,
        post_generation_retrieval: bool = True,
        critic_retrieval: bool = True,
    ):
        """Initialize the Chain with optional model, prompt, and retrievers.

        Args:
            model: Optional language model to use for text generation.
            prompt: Optional prompt to use for text generation.
            retriever: Default retriever for all stages (if specific retrievers not provided).
            model_retriever: Specific retriever for model generation (e.g., recent context like Twitter).
            critic_retriever: Specific retriever for critics (e.g., factual database for fact-checking).
            max_improvement_iterations: Maximum number of improvement iterations.
            apply_improvers_on_validation_failure: Whether to apply improvers when validation fails.
            pre_generation_retrieval: Whether to retrieve context before generation.
            post_generation_retrieval: Whether to retrieve context after generation.
            critic_retrieval: Whether to retrieve context during critic operations.
        """
        self._model = model
        self._prompt = prompt

        # Set up retrievers with fallback logic
        self._retriever = retriever  # Default retriever
        self._model_retriever = model_retriever or retriever  # Model-specific or default
        self._critic_retriever = critic_retriever or retriever  # Critic-specific or default

        self._validators: List[Validator] = []
        self._critics: List[Critic] = []
        self._options: Dict[str, Any] = {
            "max_improvement_iterations": max_improvement_iterations,
            "apply_improvers_on_validation_failure": apply_improvers_on_validation_failure,
            "pre_generation_retrieval": pre_generation_retrieval,
            "post_generation_retrieval": post_generation_retrieval,
            "critic_retrieval": critic_retrieval,
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
        6. If validation fails and apply_improvers_on_validation_failure is True:
           a. Critic retrieval: Gets additional context for critics (if configured)
           b. Gets feedback from critics on how to improve the text
           c. Sends the original text + feedback to the model to generate improved text
           d. Re-validates the improved text
           e. Repeats steps 6a-6d until validation passes or max iterations is reached
        7. Returns a Thought object with the final text and all validation/improvement results

        The Chain orchestrates ALL retrieval operations - models and critics no longer
        handle retrieval themselves.

        Returns:
            A Thought object containing the final text and all validation/improvement results.

        Raises:
            ChainError: If the chain is not properly configured or an error occurs during execution.
        """
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

        # 1. PRE-GENERATION RETRIEVAL (Chain orchestrates this using model_retriever)
        if self._model_retriever and self._options.get("pre_generation_retrieval", True):
            with chain_context(
                operation="pre_generation_retrieval",
                message_prefix="Failed to retrieve pre-generation context",
            ):
                logger.debug("Chain orchestrating pre-generation retrieval with model_retriever...")
                thought = self._model_retriever.retrieve_for_thought(
                    thought, is_pre_generation=True
                )
                logger.debug(
                    f"Retrieved {len(thought.pre_generation_context)} pre-generation documents from model_retriever"
                )

        # 2. GENERATE TEXT (Model just generates, no retrieval)
        with chain_context(
            operation="generation",
            message_prefix="Failed to generate text",
        ):
            logger.debug(f"Generating text with prompt: {self._prompt[:100]}...")
            text = self._model.generate_with_thought(thought)
            thought = thought.set_text(text)
            logger.debug(f"Generated text of length {len(text)}")

        # 3. POST-GENERATION RETRIEVAL (Chain orchestrates this using model_retriever)
        if self._model_retriever and self._options.get("post_generation_retrieval", True):
            with chain_context(
                operation="post_generation_retrieval",
                message_prefix="Failed to retrieve post-generation context",
            ):
                logger.debug(
                    "Chain orchestrating post-generation retrieval with model_retriever..."
                )
                thought = self._model_retriever.retrieve_for_thought(
                    thought, is_pre_generation=False
                )
                logger.debug(
                    f"Retrieved {len(thought.post_generation_context)} post-generation documents from model_retriever"
                )

        # 4. VALIDATE TEXT
        all_passed = True
        for validator in self._validators:
            with chain_context(
                operation="validation",
                message_prefix=f"Failed to validate text with {validator.__class__.__name__}",
            ):
                logger.debug(f"Validating text with {validator.__class__.__name__}")
                validation_result = validator.validate(thought)

                # Convert to ValidationResult if needed
                if not isinstance(validation_result, ValidationResult):
                    validation_result = ValidationResult(
                        passed=validation_result.get("passed", False),
                        message=validation_result.get("message", ""),
                        score=validation_result.get("score"),
                        issues=validation_result.get("issues"),
                        suggestions=validation_result.get("suggestions"),
                    )

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

        # 5. CRITIC IMPROVEMENT LOOP (if validation failed)
        max_iterations = self._options.get("max_improvement_iterations", 3)
        current_iteration = 0

        while (
            not all_passed
            and self._options.get("apply_improvers_on_validation_failure", False)
            and current_iteration < max_iterations
            and self._critics
        ):
            current_iteration += 1
            logger.debug(
                f"Validation failed, attempting improvement (iteration {current_iteration}/{max_iterations})"
            )

            # Create a new thought for this iteration
            thought = thought.next_iteration()

            # 5a. CRITIC RETRIEVAL (Chain orchestrates this using critic_retriever)
            if self._critic_retriever and self._options.get("critic_retrieval", True):
                with chain_context(
                    operation="critic_retrieval",
                    message_prefix="Failed to retrieve context for critics",
                ):
                    logger.debug("Chain orchestrating critic retrieval with critic_retriever...")
                    # For critics, we might want fresh post-generation context from the critic_retriever
                    thought = self._critic_retriever.retrieve_for_thought(
                        thought, is_pre_generation=False
                    )
                    logger.debug(
                        f"Retrieved {len(thought.post_generation_context)} documents for critics from critic_retriever"
                    )

            # 5b. Apply critics to improve the text (critics no longer do retrieval)
            for critic in self._critics:
                with chain_context(
                    operation="improvement",
                    message_prefix=f"Failed to improve text with {critic.__class__.__name__}",
                ):
                    logger.debug(f"Improving text with {critic.__class__.__name__}")

                    # Get critique (critic uses provided context, no retrieval)
                    critique = critic.critique(thought)
                    thought = thought.set_critique(critique)

                    # Improve text (critic uses provided context, no retrieval)
                    improved_text = critic.improve(thought)
                    thought = thought.set_text(improved_text)

                    logger.debug(
                        f"Improved text with {critic.__class__.__name__}, "
                        f"new length: {len(improved_text)}"
                    )

            # Re-validate the improved text
            all_passed = True
            for validator in self._validators:
                with chain_context(
                    operation="validation",
                    message_prefix=f"Failed to validate improved text with {validator.__class__.__name__}",
                ):
                    logger.debug(f"Re-validating improved text with {validator.__class__.__name__}")
                    validation_result = validator.validate(thought)

                    # Convert to ValidationResult if needed
                    if not isinstance(validation_result, ValidationResult):
                        validation_result = ValidationResult(
                            passed=validation_result.get("passed", False),
                            message=validation_result.get("message", ""),
                            score=validation_result.get("score"),
                            issues=validation_result.get("issues"),
                            suggestions=validation_result.get("suggestions"),
                        )

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
