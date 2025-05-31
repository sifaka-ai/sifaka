"""Validation execution for PydanticAI chains.

This module handles the validation phase, running validators concurrently
and collecting results.
"""

import asyncio
from typing import List

from sifaka.core.interfaces import Validator
from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class ValidationExecutor:
    """Handles validation execution for thoughts."""

    def __init__(self, validators: List[Validator]):
        """Initialize the validation executor.

        Args:
            validators: List of validators to apply.
        """
        self.validators = validators

    async def execute(self, thought: Thought) -> Thought:
        """Execute validation on the generated text.

        Args:
            thought: The thought with generated text.

        Returns:
            Updated thought with validation results.
        """
        if not self.validators:
            logger.debug("No validators configured, skipping validation")
            return thought

        logger.debug(f"Running async validation with {len(self.validators)} validators")

        with time_operation("validation"):
            # Run all validators concurrently
            validation_tasks = [
                self._validate_with_validator(validator, thought) for validator in self.validators
            ]

            # Wait for all validations to complete
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(validation_results):
                validator = self.validators[i]
                validator_name = validator.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Validation error for {validator_name}: {result}")
                    # Continue with other validators - don't add failed validation to thought
                else:
                    # Add successful validation result to thought
                    thought = thought.add_validation_result(validator_name, result)
                    logger.debug(
                        f"Async validation by {validator_name}: {'PASSED' if result.passed else 'FAILED'}"
                    )

            return thought

    async def _validate_with_validator(self, validator: Validator, thought: Thought):
        """Run a single validator asynchronously with error handling.

        Args:
            validator: The validator to run.
            thought: The thought to validate.

        Returns:
            The validation result.
        """
        try:
            # All validators must now be async-only
            return await validator.validate_async(thought)
        except Exception as e:
            logger.error(f"Async validation failed for {validator.__class__.__name__}: {e}")
            raise

    def validation_passed(self, thought: Thought) -> bool:
        """Check if all validations passed.

        Args:
            thought: The thought to check.

        Returns:
            True if all validations passed, False otherwise.
        """
        if not hasattr(thought, "validation_results") or not thought.validation_results:
            return True  # No validations means pass

        return all(result.passed for result in thought.validation_results.values())
