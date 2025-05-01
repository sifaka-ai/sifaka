"""
Chain module for Sifaka.

This module provides the Chain class which orchestrates the validation and improvement
flow between models, rules, and critics.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Dict, Any

from .critics import PromptCritic
from .critics.prompt import CriticMetadata
from .generation import Generator
from .improvement import Improver, ImprovementResult
from .models.base import ModelProvider
from .rules import Rule, RuleResult
from .validation import Validator, ValidationResult

OutputType = TypeVar("OutputType")


@dataclass
class ChainResult(Generic[OutputType]):
    """Result from running a chain, including the output and validation details."""

    output: OutputType
    rule_results: List[RuleResult]
    critique_details: Optional[Dict[str, Any]] = None


class Chain(Generic[OutputType]):
    """
    Chain class that orchestrates the validation and improvement flow.

    This class combines generation, validation, and improvement components to
    create a complete pipeline for generating, validating, and improving outputs.
    """

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[PromptCritic] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a Chain instance.

        Args:
            model: The model provider to use for generation
            rules: List of validation rules to apply
            critic: Optional critic for improving outputs
            max_attempts: Maximum number of improvement attempts
        """
        self.generator = Generator[OutputType](model)
        self.validator = Validator[OutputType](rules)
        self.improver = Improver[OutputType](critic) if critic else None
        self.max_attempts = max_attempts

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the prompt through the chain.

        Args:
            prompt: The input prompt to process

        Returns:
            ChainResult containing the output and validation details

        Raises:
            ValueError: If validation fails after max attempts
        """
        attempts = 0
        current_prompt = prompt
        last_critique_details = None

        while attempts < self.max_attempts:
            # Generate output
            output = self.generator.generate(current_prompt)

            # Validate output
            validation_result = self.validator.validate(output)

            # If validation passed, return result
            if validation_result.all_passed:
                return ChainResult(
                    output=output,
                    rule_results=validation_result.rule_results,
                    critique_details=last_critique_details,
                )

            # If validation failed but we have no improver, raise error
            if not self.improver:
                error_messages = self.validator.get_error_messages(validation_result)
                raise ValueError(f"Validation failed. Errors:\n" + "\n".join(error_messages))

            # If we have an improver and validation failed, try to improve
            if attempts < self.max_attempts - 1:
                improvement_result = self.improver.improve(output, validation_result)
                last_critique_details = improvement_result.critique_details

                if improvement_result.critique_details:
                    feedback = self.improver.get_feedback(improvement_result.critique_details)
                    current_prompt = f"{prompt}\n\nPrevious attempt feedback:\n{feedback}"

                attempts += 1
                continue

            # If we're out of attempts, raise error
            error_messages = self.validator.get_error_messages(validation_result)
            raise ValueError(
                f"Validation failed after {attempts + 1} attempts. Errors:\n"
                + "\n".join(error_messages)
            )

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of chain execution")
