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

    def improve(self, output: str, violations: List[Dict[str, Any]]) -> str:
        """Improve an output based on rule violations.

        Args:
            output: The output to improve
            violations: List of rule violations

        Returns:
            str: The improved output
        """
        # Construct improvement prompt
        violation_text = "\n".join(f"- {v['rule']}: {v['message']}" for v in violations)
        improve_prompt = f"""
        Please improve the following text to fix these violations:

        VIOLATIONS:
        {violation_text}

        ORIGINAL TEXT:
        {output}

        REQUIREMENTS:
        1. Fix all violations while preserving the key information
        2. Return ONLY the improved text, with no additional explanations or formatting
        3. Ensure the output is in markdown format
        4. Keep the length within the specified limits
        """

        # Get improved version from the model
        improved = self.model.generate(improve_prompt)

        # Ensure we return a string
        if not isinstance(improved, str):
            raise TypeError("Model must return a string")

        return improved.strip()

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
        Please evaluate the following prompt and provide your response in a structured format with these components:
        1. A score between 0 and 1 (where 1 is perfect)
        2. General feedback
        3. List of specific issues
        4. List of improvement suggestions

        Format your response exactly like this:
        SCORE: [number between 0 and 1]
        FEEDBACK: [your general feedback]
        ISSUES:
        - [issue 1]
        - [issue 2]
        SUGGESTIONS:
        - [suggestion 1]
        - [suggestion 2]

        Here is the prompt to evaluate:

        {prompt}

        Consider:
        1. Is the prompt clear and unambiguous?
        2. Are the instructions complete?
        3. Are any constraints clearly specified?
        4. Will this prompt be effective for its purpose?
        """

        # Get response from the model
        response = self.model.generate(critique_prompt)

        if not isinstance(response, str):
            raise TypeError("Model response must be a string")

        # Parse the response into a dictionary
        try:
            # Split response into sections
            sections = response.split("\n")

            # Initialize result dictionary
            result = {"score": 0.0, "feedback": "", "issues": [], "suggestions": []}

            current_section = None
            for line in sections:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("SCORE:"):
                    try:
                        score_str = line.replace("SCORE:", "").strip()
                        result["score"] = float(score_str)
                    except ValueError:
                        result["score"] = 0.5  # Default score if parsing fails
                elif line.startswith("FEEDBACK:"):
                    result["feedback"] = line.replace("FEEDBACK:", "").strip()
                elif line.startswith("ISSUES:"):
                    current_section = "issues"
                elif line.startswith("SUGGESTIONS:"):
                    current_section = "suggestions"
                elif line.startswith("-") and current_section:
                    item = line.replace("-", "").strip()
                    if item:
                        result[current_section].append(item)

            # Validate the parsed result
            if not isinstance(result["score"], (int, float)):
                raise TypeError("Score must be a number")
            if not isinstance(result["feedback"], str):
                raise TypeError("Feedback must be a string")
            if not isinstance(result["issues"], list):
                raise TypeError("Issues must be a list")
            if not isinstance(result["suggestions"], list):
                raise TypeError("Suggestions must be a list")

            return result

        except Exception as e:
            # If parsing fails, return a default structured response
            return {
                "score": 0.5,
                "feedback": str(response),  # Use full response as feedback
                "issues": ["Failed to parse structured response"],
                "suggestions": ["Please try again with a clearer prompt"],
            }

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
