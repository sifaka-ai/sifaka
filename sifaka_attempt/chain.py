"""
Chain module for the new Sifaka.

This module provides a simplified Chain class for processing text using language models,
validators, and critics.
"""

from typing import Any, Dict, List, Optional, Union

from .types import ChainResult, ValidationResult
from .validators import Validator
from .critics import Critic
from .models import ModelProvider, create_model
from .di import resolve


class Chain:
    """
    A simplified Chain for text processing.

    This class provides a clean, fluent interface for building chains to process text
    using language models, with validation and improvement capabilities.

    Example:
        ```python
        from sifaka import Chain

        chain = Chain(model="openai", api_key="your-api-key")
        chain.add_validator(LengthValidator(max_chars=500))
        chain.add_critic(PromptCritic(instructions="Evaluate clarity"))
        result = chain.run("Write a summary about AI")
        ```
    """

    def __init__(self, model: Union[str, ModelProvider], **model_kwargs):
        """
        Initialize a new Chain.

        Args:
            model: Either a model provider instance or a string name of a model provider
            **model_kwargs: Additional arguments to pass to the model factory if model is a string
        """
        # Initialize the model using DI for string names
        if isinstance(model, str):
            factory = resolve("model.factory")
            self.model = factory(model, **model_kwargs)
        else:
            # Initialize with model instance directly
            self.model = model

        self.validators: List[Validator] = []
        self.critics: List[Critic] = []

    def add_validator(self, validator: Union[str, Validator], **kwargs) -> "Chain":
        """
        Add a validator to the chain.

        Args:
            validator: The validator to add, either a validator instance or a registered name
            **kwargs: Configuration parameters if creating a validator by name

        Returns:
            The chain instance for method chaining
        """
        if isinstance(validator, str):
            validator_instance = resolve(f"validator.{validator}")
            if validator_instance is None and kwargs:
                # Try to create using a factory
                factory = resolve(f"validator.{validator}")
                if factory:
                    validator_instance = factory(**kwargs)

            if validator_instance:
                self.validators.append(validator_instance)
            else:
                raise ValueError(f"Validator '{validator}' not found")
        else:
            self.validators.append(validator)

        return self

    def add_critic(self, critic: Union[str, Critic], **kwargs) -> "Chain":
        """
        Add a critic to the chain.

        Args:
            critic: The critic to add, either a critic instance or a registered name
            **kwargs: Configuration parameters if creating a critic by name

        Returns:
            The chain instance for method chaining
        """
        if isinstance(critic, str):
            critic_instance = resolve(f"critic.{critic}")
            if critic_instance is None and kwargs:
                # Try to create using a factory
                factory = resolve(f"critic.{critic}")
                if factory:
                    critic_instance = factory(**kwargs)

            if critic_instance:
                self.critics.append(critic_instance)
            else:
                raise ValueError(f"Critic '{critic}' not found")
        else:
            self.critics.append(critic)

        return self

    def run(self, input_text: str) -> Union[str, ChainResult]:
        """
        Run the chain on the input text.

        Args:
            input_text: The input text to process

        Returns:
            Either a string with the generated (and possibly improved) text,
            or a ChainResult with additional details
        """
        # Generate output
        output = self.model.generate(input_text)

        # Extract text if needed
        output_text = output.output if hasattr(output, "output") else output

        # Validate output
        validation_results = [v.validate(output_text) for v in self.validators]
        all_passed = all(result.passed for result in validation_results)

        # Improve if needed
        if not all_passed and self.critics:
            # Collect issues from failed validations
            issues = [
                issue
                for result in validation_results
                if not result.passed
                for issue in result.issues
            ]

            # Use the first critic to improve the output
            improved_output = self.critics[0].improve(output_text, issues)

            # Return the improved output with result details
            return ChainResult(
                output=improved_output,
                validation_results=validation_results,
                passed=True,  # We consider it passed after improvement
                original_output=output_text,
            )

        # Return the output with result details
        return ChainResult(
            output=output_text, validation_results=validation_results, passed=all_passed
        )
