"""Implementation of a prompt critic using a language model."""

from typing import Dict, Any, List, Optional
from pydantic import Field

from .base import Critic


class PromptCritic(Critic):
    """A critic that uses a language model to evaluate and validate prompts.

    This critic analyzes prompts for clarity, ambiguity, completeness, and effectiveness
    using a language model to generate feedback and validation scores.

    Attributes:
        name: Name of the critic (defaults to "Prompt Critic").
        description: Description of what this critic evaluates.
        model: The language model to use for critiquing.
        min_confidence: Minimum confidence score for validation (default: 0.7).
        config: Additional configuration parameters.
    """

    name: str = Field(default="Prompt Critic", description="Name of the critic")
    description: str = Field(
        default="Evaluates prompts for clarity, completeness, and effectiveness",
        description="Description of what this critic evaluates",
    )
    model: Any = Field(..., description="Language model to use for critiquing")

    def critique(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and provide detailed feedback.

        Uses the language model to evaluate the prompt based on:
        - Clarity and ambiguity
        - Completeness of instructions
        - Adherence to constraints
        - Overall effectiveness

        Args:
            prompt: The prompt to critique.

        Returns:
            Dict containing:
                - score: float between 0 and 1
                - feedback: str with general feedback
                - issues: list of identified issues
                - suggestions: list of improvement suggestions

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
        """
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Construct critique prompt for the language model
        critique_prompt = f"""
        Please evaluate the following prompt for clarity, completeness, and effectiveness:

        {prompt}

        Consider:
        1. Is the prompt clear and unambiguous?
        2. Are the instructions complete?
        3. Are any constraints clearly specified?
        4. Will this prompt be effective for its purpose?
        """

        # Get response from the model
        response = self.model.generate(critique_prompt)

        # Validate response format
        if not isinstance(response, dict):
            raise TypeError("Model response must be a dictionary")

        required_keys = ["score", "feedback", "issues", "suggestions"]
        missing_keys = [key for key in required_keys if key not in response]
        if missing_keys:
            raise KeyError(f"Model response missing required keys: {missing_keys}")

        if not isinstance(response["score"], (int, float)):
            raise TypeError("Score must be a number")
        if not isinstance(response["feedback"], str):
            raise TypeError("Feedback must be a string")
        if not isinstance(response["issues"], list):
            raise TypeError("Issues must be a list")
        if not isinstance(response["suggestions"], list):
            raise TypeError("Suggestions must be a list")

        return response

    def validate(self, prompt: str) -> bool:
        """Check if a prompt meets quality standards.

        Uses the critique method to determine if the prompt's score meets
        the minimum confidence threshold.

        Args:
            prompt: The prompt to validate.

        Returns:
            bool: True if the prompt meets quality standards, False otherwise.

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            KeyError: If model response is missing required fields
            TypeError: If model response contains invalid types
        """
        result = self.critique(prompt)
        return float(result["score"]) >= self.min_confidence
