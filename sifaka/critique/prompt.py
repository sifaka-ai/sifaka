"""
Prompt-based critique for Sifaka.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field
from .base import Critique
from ..models.base import ModelProvider


class PromptCritique(Critique):
    """
    A critique that uses prompts to improve LLM outputs.

    This critique uses the LLM itself to improve outputs based on rule violations.

    Attributes:
        name: The name of the critique
        model: The model provider to use for critique
        system_prompt: The system prompt to use for critique
        user_prompt_template: The template for the user prompt
    """

    system_prompt: str = Field(
        default="You are a helpful AI assistant that improves text based on feedback.",
        description="The system prompt to use for critique",
    )
    user_prompt_template: str = Field(
        default=(
            "Please improve the following text based on the feedback:\n\n"
            "Original text: {output}\n\n"
            "Feedback:\n{feedback}\n\n"
            "Improved text:"
        ),
        description="The template for the user prompt",
    )

    def __init__(
        self,
        model: ModelProvider,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a prompt critique.

        Args:
            model: The model provider to use for critique
            system_prompt: The system prompt to use for critique
            user_prompt_template: The template for the user prompt
            **kwargs: Additional arguments for the critique
        """
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        if user_prompt_template is not None:
            kwargs["user_prompt_template"] = user_prompt_template

        super().__init__(model=model, **kwargs)

    def _format_feedback(self, rule_violations: List[Dict[str, Any]]) -> str:
        """
        Format rule violations into feedback.

        Args:
            rule_violations: List of rule violations

        Returns:
            The formatted feedback
        """
        if not rule_violations:
            return "No specific feedback provided."

        feedback = []
        for violation in rule_violations:
            rule_name = violation.get("rule", "Unknown rule")
            message = violation.get("message", "No message provided")
            feedback.append(f"- {rule_name}: {message}")

        return "\n".join(feedback)

    def improve(
        self, output: str, prompt: str, rule_violations: List[Dict[str, Any]], **kwargs
    ) -> str:
        """
        Improve the output based on rule violations.

        Args:
            output: The output to improve
            prompt: The original prompt
            rule_violations: List of rule violations
            **kwargs: Additional arguments for improvement

        Returns:
            The improved output
        """
        feedback = self._format_feedback(rule_violations)
        user_prompt = self.user_prompt_template.format(output=output, feedback=feedback)

        improved_output = self.model.generate(
            user_prompt, system_prompt=self.system_prompt, **kwargs
        )

        return improved_output.strip()
