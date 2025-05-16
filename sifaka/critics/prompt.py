"""
Prompt-based critic for Sifaka.

This module provides a critic that uses custom prompts to improve text.

This is a general-purpose critic that can be customized with different prompts
to implement various critique and improvement strategies. It serves as a flexible
foundation for more specialized critics.
"""

import json
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError


class PromptCritic(Critic):
    """Critic that uses custom prompts to improve text.

    This critic uses custom prompts to critique and improve text. It allows
    for flexible customization of the critique and improvement process.

    Attributes:
        model: The model to use for critiquing and improving text.
        critique_prompt_template: The template to use for critiquing text.
        improvement_prompt_template: The template to use for improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        critique_prompt_template: Optional[str] = None,
        improvement_prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the prompt critic.

        Args:
            model: The model to use for critiquing and improving text.
            critique_prompt_template: The template to use for critiquing text.
            improvement_prompt_template: The template to use for improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert editor who specializes in improving text. "
                "Your goal is to provide detailed feedback and suggestions for improvement."
            )

        super().__init__(model, system_prompt, temperature, **options)

        # Use default critique prompt template if not provided
        self.critique_prompt_template = critique_prompt_template or (
            "Please analyze the following text and provide a critique:\n\n"
            "```\n{text}\n```\n\n"
            "Provide your analysis in JSON format with the following fields:\n"
            '- "needs_improvement": boolean indicating whether the text needs improvement\n'
            '- "message": a brief summary of your analysis\n'
            '- "issues": a list of specific issues found\n'
            '- "suggestions": a list of suggestions for improvement\n\n'
            "JSON response:"
        )

        # Use default improvement prompt template if not provided
        self.improvement_prompt_template = improvement_prompt_template or (
            "Please improve the following text based on the critique:\n\n"
            "Text:\n```\n{text}\n```\n\n"
            "Critique:\n{critique}\n\n"
            "Improved text:"
        )

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the critique prompt template.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        prompt = self.critique_prompt_template.format(text=text)

        try:
            response = self._generate(prompt)

            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                # No JSON found, create a default response
                return {
                    "needs_improvement": True,
                    "message": "Unable to parse critique response, but proceeding with improvement",
                    "issues": ["Unable to identify specific issues"],
                    "suggestions": ["General improvement"],
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("needs_improvement", True)
            critique.setdefault("message", "Text needs improvement")
            critique.setdefault("issues", [])
            critique.setdefault("suggestions", [])

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
            }
        except Exception as e:
            raise ImproverError(f"Error critiquing text: {str(e)}")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text based on critique.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        # Format critique as a string
        critique_str = f"Issues:\n"
        for issue in critique.get("issues", []):
            critique_str += f"- {issue}\n"

        critique_str += f"\nSuggestions:\n"
        for suggestion in critique.get("suggestions", []):
            critique_str += f"- {suggestion}\n"

        # Create improvement prompt
        prompt = self.improvement_prompt_template.format(
            text=text,
            critique=critique_str,
        )

        try:
            response = self._generate(prompt)

            # Extract improved text from response
            improved_text = response.strip()

            # Remove any markdown code block markers
            if improved_text.startswith("```") and improved_text.endswith("```"):
                improved_text = improved_text[3:-3].strip()

            return improved_text
        except Exception as e:
            raise ImproverError(f"Error improving text: {str(e)}")


def create_prompt_critic(
    model: Model,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> PromptCritic:
    """Create a prompt critic.

    This is a convenience function for creating a PromptCritic.

    Args:
        model: The model to use for critiquing and improving text.
        critique_prompt_template: The template to use for critiquing text.
        improvement_prompt_template: The template to use for improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the PromptCritic.

    Returns:
        A PromptCritic instance.
    """
    return PromptCritic(
        model=model,
        critique_prompt_template=critique_prompt_template,
        improvement_prompt_template=improvement_prompt_template,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )
