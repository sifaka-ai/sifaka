"""
Constitutional critic for Sifaka.

This module provides a critic that evaluates text against a set of principles.

Based on the paper:
"Constitutional AI: Harmlessness from AI Feedback"
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson,
Christopher Olah, Circulation, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Hendrycks,
Ethan Perez, Jamie Kerr, Jared Kaplan, Jeremie M. Harris, Joseph Gonzalez, Josh Landau,
Liane Lovitt, Michael Sellitto, Miles Brundage, Pamela Mishkin, Paul Christiano, Rachel Hao,
Raphael Millière, Sam Bowman, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Scott Sievert,
Sheer El-Showk, Stanislav Fort, Timothy Telleen-Lawton, Thomas Langlois, Tyna Eloundou,
Varun Sundar, Yuntao Bai, Zac Hatfield-Dodds
arXiv:2212.08073 [cs.CL]
https://arxiv.org/abs/2212.08073
"""

import json
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver


class ConstitutionalCritic(Critic):
    """Critic that evaluates text against a set of principles.

    This critic evaluates text against a set of principles (a "constitution")
    and provides feedback when violations are detected.

    Attributes:
        model: The model to use for critiquing and improving text.
        principles: The principles to evaluate text against.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        principles: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the constitutional critic.

        Args:
            model: The model to use for critiquing and improving text.
            principles: The principles to evaluate text against.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert editor who specializes in evaluating text against principles. "
                "Your goal is to identify violations of principles and provide feedback for improvement."
            )

        super().__init__(model, system_prompt, temperature, **options)

        # Use default principles if not provided
        self.principles = principles or [
            "The text should be clear and concise.",
            "The text should be grammatically correct.",
            "The text should be well-structured.",
            "The text should be factually accurate.",
            "The text should be appropriate for the intended audience.",
        ]

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text against the principles.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        # Format principles as a bulleted list
        principles_str = "\n".join(f"- {p}" for p in self.principles)

        prompt = f"""
        Please evaluate the following text against these principles:

        Principles:
        {principles_str}

        Text:
        ```
        {text}
        ```

        Provide your evaluation in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text violates any principles
        - "message": a brief summary of your evaluation
        - "violations": a list of specific principle violations
        - "suggestions": a list of suggestions for improvement

        JSON response:
        """

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
                    "violations": ["Unable to identify specific violations"],
                    "suggestions": ["General improvement"],
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("needs_improvement", True)
            critique.setdefault("message", "Text needs improvement")
            critique.setdefault("violations", [])
            critique.setdefault("suggestions", [])

            # Add issues field for compatibility with base Critic
            critique["issues"] = critique.get("violations", [])

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "violations": ["Unable to identify specific violations"],
                "suggestions": ["General improvement"],
                "issues": ["Unable to identify specific violations"],
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
        # Format principles as a bulleted list
        principles_str = "\n".join(f"- {p}" for p in self.principles)

        # Format violations and suggestions
        violations = critique.get("violations", [])
        suggestions = critique.get("suggestions", [])

        violations_str = "\n".join(f"- {v}" for v in violations)
        suggestions_str = "\n".join(f"- {s}" for s in suggestions)

        prompt = f"""
        Please improve the following text to address the violations of principles:

        Principles:
        {principles_str}

        Text:
        ```
        {text}
        ```

        Violations:
        {violations_str}

        Suggestions:
        {suggestions_str}

        Improved text:
        """

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


@register_improver("constitutional")
def create_constitutional_critic(
    model: Model,
    principles: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> ConstitutionalCritic:
    """Create a constitutional critic.

    This factory function creates a ConstitutionalCritic based on the paper
    "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        principles: The principles to evaluate text against.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the ConstitutionalCritic.

    Returns:
        A ConstitutionalCritic instance.
    """
    return ConstitutionalCritic(
        model=model,
        principles=principles,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )
