"""
Constitutional critic for Sifaka.

This module provides a critic that evaluates text against a set of principles
(a "constitution") and provides feedback when violations are detected.

Reference: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
https://arxiv.org/abs/2212.08073
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base_critic import BaseCritic

# Configure logger
logger = logging.getLogger(__name__)

# Default principles
DEFAULT_PRINCIPLES = [
    "The text should be helpful, harmless, and honest.",
    "The text should not contain harmful, offensive, or misleading content.",
    "The text should respect privacy and confidentiality.",
    "The text should be accurate and factual.",
    "The text should be clear, concise, and well-structured.",
]


class ConstitutionalCritic(BaseCritic):
    """
    Critic that evaluates text against a set of principles.

    This critic evaluates text against a set of principles (a "constitution")
    and provides feedback when violations are detected.

    Reference: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
    https://arxiv.org/abs/2212.08073

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
        name: str = "constitutional",
        **options: Any,
    ):
        """
        Initialize the constitutional critic.

        Args:
            model: The model to use for critiquing and improving text.
            principles: The principles to evaluate text against.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            name: Name of the critic.
            **options: Additional options for the critic.

        Raises:
            ValueError: If the model is not provided.
        """
        # Use default principles if none provided
        self.principles = principles or DEFAULT_PRINCIPLES

        # Initialize base critic
        super().__init__(
            model=model,
            system_prompt=system_prompt
            or (
                "You are a helpful assistant that evaluates text against a set of principles. "
                "You provide detailed feedback when violations are detected."
            ),
            temperature=temperature,
            name=name,
            **options,
        )

        # Log initialization
        logger.debug(
            f"Initialized {self.name} critic with {len(self.principles)} principles, "
            f"temperature={temperature}"
        )

    def _critique(self, thought: Thought) -> Dict[str, Any]:
        """
        Critique text against the principles.

        Args:
            thought: The thought containing the text to critique.

        Returns:
            A dictionary with critique results.
        """
        text = thought.text

        # Format principles as a bulleted list
        principles_str = "\n".join(f"- {p}" for p in self.principles)

        # Create critique prompt
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

        # Generate critique
        response = self._generate_with_model(prompt)

        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                critique_data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                logger.warning(
                    f"{self.name}: No JSON found in response, using default critique data"
                )
                critique_data = {
                    "needs_improvement": True,
                    "message": "Failed to parse critique response",
                    "violations": ["Unable to extract structured critique from model response"],
                    "suggestions": ["Try again with a different prompt or model"],
                }

            # Extract critique information
            needs_improvement = critique_data.get("needs_improvement", True)
            message = critique_data.get("message", "Evaluation completed")
            violations = critique_data.get("violations", [])
            suggestions = critique_data.get("suggestions", [])

            # Log critique results
            logger.debug(
                f"{self.name}: Critique completed, needs_improvement={needs_improvement}, "
                f"violations={len(violations)}, suggestions={len(suggestions)}"
            )

            # Create feedback text
            feedback = f"{message}\n\n"

            if violations:
                feedback += "Violations:\n"
                for violation in violations:
                    feedback += f"- {violation}\n"
                feedback += "\n"

            if suggestions:
                feedback += "Suggestions:\n"
                for suggestion in suggestions:
                    feedback += f"- {suggestion}\n"

            # Return critique results
            return {
                "feedback": feedback,
                "suggestions": suggestions,
                "details": {
                    "needs_improvement": needs_improvement,
                    "message": message,
                    "violations": violations,
                    "principles": self.principles,
                },
            }

        except Exception as e:
            # Handle JSON parsing errors
            logger.error(f"Error parsing critique response: {str(e)}")

            # Create fallback feedback
            feedback = (
                "Failed to parse critique response. The text may violate some principles, "
                "but specific violations could not be determined."
            )

            return {
                "feedback": feedback,
                "suggestions": ["Try again with a different prompt or model"],
                "details": {
                    "needs_improvement": True,
                    "message": "Failed to parse critique response",
                    "error": str(e),
                    "raw_response": response,
                },
            }


def create_constitutional_critic(
    model: Model,
    principles: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    name: str = "constitutional",
    **options: Any,
) -> ConstitutionalCritic:
    """
    Create a constitutional critic.

    This factory function creates a ConstitutionalCritic based on the paper
    "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).

    Args:
        model: The model to use for critiquing and improving text.
        principles: The principles to evaluate text against.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        name: Name of the critic.
        **options: Additional options for the critic.

    Returns:
        A ConstitutionalCritic instance.

    Raises:
        ValueError: If the model is not provided.
    """
    return ConstitutionalCritic(
        model=model,
        principles=principles,
        system_prompt=system_prompt,
        temperature=temperature,
        name=name,
        **options,
    )
