"""
Reflexion critic for Sifaka.

This module provides a critic that implements the Reflexion technique
(https://arxiv.org/abs/2303.11366) for improving text generation.
"""

import logging
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base_critic import BaseCritic
from sifaka.models.openai_model import OpenAIModel

logger = logging.getLogger(__name__)


class ReflexionCritic(BaseCritic):
    """
    Critic that implements the Reflexion technique.

    Reflexion is a technique for improving text generation by having the model
    reflect on its own output and identify areas for improvement.

    Reference: https://arxiv.org/abs/2303.11366
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        reflection_prompt_template: Optional[str] = None,
        name: str = "reflexion",
        **options: Any,
    ):
        """
        Initialize the Reflexion critic.

        Args:
            model: Model to use for reflection. If None, uses OpenAIModel.
            reflection_prompt_template: Template for the reflection prompt.
            name: Name of the critic.
            **options: Additional options for the critic.
        """
        # Create default model if none provided
        if model is None:
            model = OpenAIModel(
                model_name="gpt-3.5-turbo",
                temperature=0.2,
                system_prompt=(
                    "You are a helpful assistant that analyzes text and provides constructive feedback "
                    "on how to improve it. Focus on identifying issues and suggesting specific improvements."
                ),
            )

        # Initialize base critic
        super().__init__(model=model, name=name, **options)

        # Store reflection prompt template
        self.reflection_prompt_template = reflection_prompt_template or (
            "Please analyze the following text and provide constructive feedback on how to improve it. "
            "Focus on identifying issues and suggesting specific improvements.\n\n"
            "Text to analyze:\n\n{text}\n\n"
            "If there are validation issues, they are:\n{validation_issues}\n\n"
            "Please provide your analysis and suggestions for improvement."
        )

    def _critique(self, thought: Thought) -> Dict[str, Any]:
        """
        Implement specific critique logic for Reflexion.

        Args:
            thought: The thought containing the text to critique.

        Returns:
            A dictionary with critique results.
        """
        # Extract validation issues
        validation_issues = ""
        for result in thought.validation_results:
            if not result.passed:
                validation_issues += f"- {result.validator_name}: {result.message}\n"

        if not validation_issues:
            validation_issues = "No specific validation issues identified."

        # Create reflection prompt
        reflection_prompt = self.reflection_prompt_template.format(
            text=thought.text,
            validation_issues=validation_issues,
        )

        # Generate reflection
        reflection = self._generate_with_model(reflection_prompt)

        # Extract suggestions from reflection
        suggestions = self._extract_suggestions(reflection)

        # Return critique results
        return {
            "feedback": reflection,
            "suggestions": suggestions,
            "details": {
                "validation_issues": validation_issues,
                "prompt_template": self.reflection_prompt_template,
            },
        }

    def _extract_suggestions(self, reflection: str) -> List[str]:
        """
        Extract suggestions from the reflection.

        Args:
            reflection: The reflection text.

        Returns:
            A list of suggestions extracted from the reflection.
        """
        suggestions = []

        # Look for suggestions in the reflection
        lines = reflection.split("\n")
        in_suggestions_section = False

        for line in lines:
            # Check if we're entering a suggestions section
            if any(
                marker in line.lower()
                for marker in ["suggestion", "recommend", "improve", "change"]
            ):
                in_suggestions_section = True
                continue

            # If we're in a suggestions section, extract bullet points
            if in_suggestions_section and (
                line.strip().startswith("-") or line.strip().startswith("*")
            ):
                suggestion = line.strip()[1:].strip()
                if suggestion:
                    suggestions.append(suggestion)

            # Check if we're leaving a suggestions section
            if in_suggestions_section and not line.strip():
                in_suggestions_section = False

        # If no suggestions were found, try to extract sentences that sound like suggestions
        if not suggestions:
            for line in lines:
                if any(
                    marker in line.lower() for marker in ["should", "could", "try to", "consider"]
                ):
                    suggestions.append(line.strip())

        return suggestions
