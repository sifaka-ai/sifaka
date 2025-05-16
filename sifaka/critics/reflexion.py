"""
Reflexion critic for Sifaka.

This module provides a critic that uses self-reflection to improve text.

Based on the paper:
"Reflexion: Language Agents with Verbal Reinforcement Learning"
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
arXiv:2303.11366 [cs.AI]
https://arxiv.org/abs/2303.11366
"""

import json
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver


class ReflexionCritic(Critic):
    """Critic that uses self-reflection to improve text.

    This critic uses a multi-step process where the model reflects on its own
    output and iteratively improves it.

    Attributes:
        model: The model to use for critiquing and improving text.
        reflection_rounds: The number of reflection rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        reflection_rounds: int = 1,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the reflexion critic.

        Args:
            model: The model to use for critiquing and improving text.
            reflection_rounds: The number of reflection rounds to perform.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert editor who specializes in self-reflection and improvement. "
                "Your goal is to reflect on text and iteratively improve it."
            )

        super().__init__(model, system_prompt, temperature, **options)

        self.reflection_rounds = max(1, reflection_rounds)

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text through self-reflection.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        prompt = f"""
        Please reflect on the following text and identify areas for improvement:

        ```
        {text}
        ```

        Provide your reflection in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text needs improvement
        - "message": a brief summary of your reflection
        - "issues": a list of specific issues identified
        - "suggestions": a list of suggestions for improvement
        - "reflections": your detailed thoughts on the text

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
                    "issues": ["Unable to identify specific issues"],
                    "suggestions": ["General improvement"],
                    "reflections": ["Unable to generate reflections"],
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("needs_improvement", True)
            critique.setdefault("message", "Text needs improvement")
            critique.setdefault("issues", [])
            critique.setdefault("suggestions", [])
            critique.setdefault("reflections", [])

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
                "reflections": ["Unable to generate reflections"],
            }
        except Exception as e:
            raise ImproverError(f"Error critiquing text: {str(e)}")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text through multiple rounds of reflection.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        current_text = text
        reflections = []

        # Add initial reflection from critique
        if isinstance(critique.get("reflections"), list):
            reflections.extend(critique.get("reflections", []))
        elif isinstance(critique.get("reflections"), str):
            reflections.append(critique.get("reflections"))

        # Format issues and suggestions
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])

        issues_str = "\n".join(f"- {i}" for i in issues)
        suggestions_str = "\n".join(f"- {s}" for s in suggestions)

        # Perform initial improvement
        prompt = f"""
        Please improve the following text based on the issues and suggestions:

        Text:
        ```
        {current_text}
        ```

        Issues:
        {issues_str}

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

            current_text = improved_text

            # Perform additional reflection rounds
            for i in range(1, self.reflection_rounds):
                # Generate reflection on current text
                reflection_prompt = f"""
                Please reflect on the following text and identify areas for further improvement:

                ```
                {current_text}
                ```

                Previous reflections:
                {self._format_list(reflections)}

                Provide your reflection:
                """

                reflection = self._generate(reflection_prompt)
                reflections.append(reflection)

                # Improve text based on reflection
                improvement_prompt = f"""
                Please improve the following text based on the reflection:

                Text:
                ```
                {current_text}
                ```

                Reflection:
                {reflection}

                Improved text:
                """

                response = self._generate(improvement_prompt)

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()

                current_text = improved_text

            return current_text
        except Exception as e:
            raise ImproverError(f"Error improving text: {str(e)}")

    def _format_list(self, items: list) -> str:
        """Format a list of items as a numbered list.

        Args:
            items: The list of items to format.

        Returns:
            A string with the items formatted as a numbered list.
        """
        if not items:
            return "None"

        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


@register_improver("reflexion")
def create_reflexion_critic(
    model: Model,
    reflection_rounds: int = 1,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> ReflexionCritic:
    """Create a reflexion critic.

    This factory function creates a ReflexionCritic based on the paper
    "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        reflection_rounds: The number of reflection rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the ReflexionCritic.

    Returns:
        A ReflexionCritic instance.
    """
    return ReflexionCritic(
        model=model,
        reflection_rounds=reflection_rounds,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )
