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

from sifaka.core.interfaces import Critic, Model, Validator
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

    It follows a fluent interface pattern (builder pattern) for easy configuration,
    allowing you to chain method calls to set up the desired behavior.

    The typical workflow is:
    1. Create a Chain instance
    2. Configure it with a model, prompt, validators, and improvers
    3. Run the chain to generate, validate, and improve text
    4. Process the result

    Attributes:
        model: The language model to use for text generation.
        prompt: The prompt to use for text generation.
        validators: List of validators to check if the generated text meets requirements.
        critics: List of critics to improve the generated text.
        options: Dictionary of options for the chain.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        prompt: Optional[str] = None,
        max_improvement_iterations: int = 3,
        apply_improvers_on_validation_failure: bool = False,
    ):
        """Initialize the Chain with optional model and prompt.

        Args:
            model: Optional language model to use for text generation.
            prompt: Optional prompt to use for text generation.
            max_improvement_iterations: Maximum number of improvement iterations.
            apply_improvers_on_validation_failure: Whether to apply improvers when validation fails.
        """
        self._model = model
        self._prompt = prompt
        self._validators: List[Validator] = []
        self._critics: List[Critic] = []
        self._options: Dict[str, Any] = {
            "max_improvement_iterations": max_improvement_iterations,
            "apply_improvers_on_validation_failure": apply_improvers_on_validation_failure,
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

        This method runs the complete chain execution process with a feedback loop:
        1. Checks that the chain is properly configured (has a model and prompt)
        2. Generates text using the model and prompt
        3. Validates the generated text using all configured validators
        4. If validation fails and apply_improvers_on_validation_failure is True:
           a. Gets feedback from critics on how to improve the text
           b. Sends the original text + feedback to the model to generate improved text
           c. Re-validates the improved text
           d. Repeats steps 4a-4c until validation passes or max iterations is reached
        5. If validation passes and the text needs improvement:
           a. Improves the text using configured critics
        6. Returns a Thought object with the final text and all validation/improvement results

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

        # Generate text
        with chain_context(
            operation="generation",
            message_prefix="Failed to generate text",
        ):
            logger.debug(f"Generating text with prompt: {self._prompt[:100]}...")
            text = self._model.generate_with_thought(thought)
            thought = thought.set_text(text)
            logger.debug(f"Generated text of length {len(text)}")

        # Validate text
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

        # If validation failed and apply_improvers_on_validation_failure is True,
        # try to improve the text
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
            
            # Apply critics to improve the text
            for critic in self._critics:
                with chain_context(
                    operation="improvement",
                    message_prefix=f"Failed to improve text with {critic.__class__.__name__}",
                ):
                    logger.debug(f"Improving text with {critic.__class__.__name__}")
                    
                    # Get critique
                    critique = critic.critique(thought)
                    thought = thought.set_critique(critique)
                    
                    # Improve text
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
