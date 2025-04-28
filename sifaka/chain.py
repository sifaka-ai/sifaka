"""
Chain module for Sifaka.

This module provides the Chain class which orchestrates the validation and improvement
flow between models, rules, and critics.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from .critics import PromptCritic
from .critics.prompt import CriticMetadata
from .models.base import ModelProvider
from .rules import Rule, RuleResult

OutputType = TypeVar("OutputType")

@dataclass
class ChainResult(Generic[OutputType]):
    """Result from running a chain, including the output and validation details."""

    output: OutputType
    rule_results: List[RuleResult]
    critique_details: Optional[dict] = None

class Chain(Generic[OutputType]):
    """
    Chain class that orchestrates the validation and improvement flow.

    This class combines a model provider with validation rules and an optional critic
    to generate, validate, and potentially improve outputs based on prompts.
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
        self.model = model
        self.rules = rules
        self.critic = critic
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
        last_critique = None

        while attempts < self.max_attempts:
            # Generate output
            output = self.model.generate(prompt)

            # Validate output
            rule_results = []
            all_passed = True
            for rule in self.rules:
                result = rule.validate(output)
                rule_results.append(result)
                if not result.passed:
                    all_passed = False

            # If validation passed or no critic, return result
            if all_passed or (not self.critic and attempts == 0):
                return ChainResult(
                    output=output,
                    rule_results=rule_results,
                    critique_details=last_critique.__dict__ if last_critique else None,
                )

            # If we have a critic and validation failed, try to improve
            if self.critic and attempts < self.max_attempts - 1:
                critique = self.critic.critique(output)
                last_critique = critique
                if isinstance(critique, CriticMetadata):
                    feedback = critique.feedback
                else:
                    feedback = critique.get("feedback", "")
                prompt = f"{prompt}\n\nPrevious attempt feedback:\n{feedback}"
                attempts += 1
                continue

            # If we're out of attempts, raise error
            error_messages = [r.message for r in rule_results if not r.passed]
            raise ValueError(
                f"Validation failed after {attempts + 1} attempts. Errors:\n"
                + "\n".join(error_messages)
            )

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of chain execution")
