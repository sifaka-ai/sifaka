"""
Self-Refine critic for Sifaka.

This module provides a critic that uses the Self-Refine technique.

Based on the paper:
"Self-Refine: Iterative Refinement with Self-Feedback"
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe,
Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder,
Shashank Gupta, Amir Yazdanbakhsh, Peter Clark
arXiv:2303.17651 [cs.CL]
https://arxiv.org/abs/2303.17651
"""

import json
from typing import Dict, Any, Optional, List, Tuple

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver


class SelfRefineCritic(Critic):
    """Critic that uses the Self-Refine technique.

    This critic implements the Self-Refine technique, which uses a multi-step
    process of feedback and refinement to iteratively improve text.

    Attributes:
        model: The model to use for critiquing and improving text.
        refinement_rounds: The number of refinement rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        refinement_rounds: int = 2,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model: The model to use for critiquing and improving text.
            refinement_rounds: The number of refinement rounds to perform.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert editor who specializes in iterative refinement. "
                "Your goal is to provide detailed feedback and iteratively improve text."
            )

        super().__init__(model, system_prompt, temperature, **options)

        self.refinement_rounds = max(1, refinement_rounds)

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the Self-Refine technique.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        prompt = f"""
        Please evaluate the following text and provide detailed feedback for improvement:

        ```
        {text}
        ```

        Provide your evaluation in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text needs improvement
        - "message": a brief summary of your evaluation
        - "issues": a list of specific issues identified
        - "suggestions": a list of suggestions for improvement
        - "evaluation_criteria": a list of criteria you used to evaluate the text

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
                    "evaluation_criteria": ["Clarity", "Coherence", "Correctness"],
                    "refinement_history": [],
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("needs_improvement", True)
            critique.setdefault("message", "Text needs improvement")
            critique.setdefault("issues", [])
            critique.setdefault("suggestions", [])
            critique.setdefault("evaluation_criteria", ["Clarity", "Coherence", "Correctness"])
            critique["refinement_history"] = []

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
                "evaluation_criteria": ["Clarity", "Coherence", "Correctness"],
                "refinement_history": [],
            }
        except Exception as e:
            raise ImproverError(f"Error critiquing text: {str(e)}")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the Self-Refine technique.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        current_text = text
        refinement_history = critique.get("refinement_history", [])

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

            # Add initial improvement to refinement history
            refinement_history.append(
                {
                    "round": 1,
                    "text": current_text,
                    "feedback": {
                        "issues": issues,
                        "suggestions": suggestions,
                    },
                }
            )

            # Perform additional refinement rounds
            for i in range(2, self.refinement_rounds + 1):
                # Generate feedback on current text
                feedback_prompt = f"""
                Please evaluate the following text and provide detailed feedback for further improvement:

                ```
                {current_text}
                ```

                Previous feedback:
                {self._format_feedback_history(refinement_history)}

                Provide your evaluation in JSON format with the following fields:
                - "issues": a list of specific issues that still need to be addressed
                - "suggestions": a list of suggestions for further improvement

                JSON response:
                """

                feedback_response = self._generate(feedback_prompt)

                # Extract JSON from response
                json_start = feedback_response.find("{")
                json_end = feedback_response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, create a default response
                    feedback = {
                        "issues": ["Further refinement needed"],
                        "suggestions": ["Continue improving the text"],
                    }
                else:
                    try:
                        json_str = feedback_response[json_start:json_end]
                        feedback = json.loads(json_str)

                        # Ensure all required fields are present
                        feedback.setdefault("issues", [])
                        feedback.setdefault("suggestions", [])
                    except:
                        feedback = {
                            "issues": ["Further refinement needed"],
                            "suggestions": ["Continue improving the text"],
                        }

                # Format issues and suggestions
                issues_str = "\n".join(f"- {i}" for i in feedback.get("issues", []))
                suggestions_str = "\n".join(f"- {s}" for s in feedback.get("suggestions", []))

                # Improve text based on feedback
                refinement_prompt = f"""
                Please further improve the following text based on the issues and suggestions:

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

                refinement_response = self._generate(refinement_prompt)

                # Extract improved text from response
                improved_text = refinement_response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()

                current_text = improved_text

                # Add refinement to history
                refinement_history.append(
                    {
                        "round": i,
                        "text": current_text,
                        "feedback": feedback,
                    }
                )

            return current_text
        except Exception as e:
            raise ImproverError(f"Error improving text: {str(e)}")

    def _format_feedback_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the feedback history as a string.

        Args:
            history: The feedback history.

        Returns:
            A string representation of the feedback history.
        """
        if not history:
            return "No previous feedback"

        result = []

        for entry in history:
            round_num = entry.get("round", 0)
            feedback = entry.get("feedback", {})

            issues = feedback.get("issues", [])
            suggestions = feedback.get("suggestions", [])

            result.append(f"Round {round_num} feedback:")

            if issues:
                result.append("Issues:")
                for issue in issues:
                    result.append(f"- {issue}")

            if suggestions:
                result.append("Suggestions:")
                for suggestion in suggestions:
                    result.append(f"- {suggestion}")

            result.append("")

        return "\n".join(result)


@register_improver("self_refine")
def create_self_refine_critic(
    model: Model,
    refinement_rounds: int = 2,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> SelfRefineCritic:
    """Create a Self-Refine critic.

    This factory function creates a SelfRefineCritic based on the paper
    "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        refinement_rounds: The number of refinement rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the SelfRefineCritic.

    Returns:
        A SelfRefineCritic instance.
    """
    return SelfRefineCritic(
        model=model,
        refinement_rounds=refinement_rounds,
        system_prompt=system_prompt,
        temperature=temperature,
        **options,
    )
