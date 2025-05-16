"""
Language Agent Correction (LAC) critic for Sifaka.

This module provides a critic that uses the Language Agent Correction technique.

Based on the paper:
"Language Feedback Improves Language Model-based Decision Making"
Xiang Fan, Nico Daheim, Alejandro Graese, Elias Stengel-Eskin, Xiaoyu Tong, Yixuan Meng,
Yao Zhao, Lucia Zheng, Yushi Hu, Ellie Pavlick, Karthik Narasimhan, Alexander M. Rush
arXiv:2403.03692 [cs.CL]
https://arxiv.org/abs/2403.03692
"""

import json
import re
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver


class LACCritic(Critic):
    """Critic that uses Language Agent Correction (LAC) technique.

    This critic implements the Language Agent Correction (LAC) technique from the paper
    "Language Feedback Improves Language Model-based Decision Making" (Fan et al., 2023).

    LAC combines two key components:
    1. A feedback critic that provides natural language feedback
    2. A value critic that provides numerical scores

    Together, these components guide the improvement of text through a structured
    process that leverages both qualitative feedback and quantitative evaluation.

    Attributes:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        feedback_weight: Weight given to feedback critic (vs. value critic).
        max_improvement_iterations: Maximum number of improvement iterations.
    """

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        feedback_weight: float = 0.7,
        max_improvement_iterations: int = 3,
        **options: Any,
    ):
        """Initialize the LAC critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            feedback_weight: Weight given to feedback critic (vs. value critic).
            max_improvement_iterations: Maximum number of improvement iterations.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert language model that provides both detailed feedback "
                "and numerical evaluation to improve text quality. You follow the "
                "Language Agent Correction (LAC) approach to provide structured guidance."
            )

        super().__init__(model, system_prompt, temperature, **options)

        self.feedback_weight = max(0.0, min(1.0, feedback_weight))  # Clamp between 0 and 1
        self.max_improvement_iterations = max(1, max_improvement_iterations)

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the LAC technique.

        This method implements the dual-critic approach from the LAC paper:
        1. Generate natural language feedback
        2. Generate a numerical value score

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        try:
            # Step 1: Generate feedback (natural language critique)
            feedback = self._generate_feedback(text)

            # Step 2: Generate value score (numerical evaluation)
            value_score = self._generate_value_score(text)

            # Combine into a comprehensive critique
            critique = {
                "needs_improvement": value_score < 0.7,  # Threshold for improvement
                "message": feedback,
                "score": value_score,
                "feedback": feedback,
                "value": value_score,
                # For compatibility with base Critic
                "issues": [feedback],
                "suggestions": ["Improve based on the feedback provided"],
            }

            return critique
        except Exception as e:
            raise ImproverError(f"Error critiquing text with LAC: {str(e)}")

    def _generate_feedback(self, text: str) -> str:
        """Generate natural language feedback on the text.

        This implements the feedback critic component from the LAC paper.

        Args:
            text: The text to critique.

        Returns:
            Natural language feedback.

        Raises:
            ImproverError: If feedback cannot be generated.
        """
        prompt = f"""
        You are a language feedback critic as described in the paper "Language Feedback Improves
        Language Model-based Decision Making" (Fan et al., 2023).

        Your task is to provide detailed, constructive feedback on the following text:

        ```
        {text}
        ```

        Please provide specific, actionable feedback that:
        1. Identifies strengths and weaknesses
        2. Suggests concrete improvements
        3. Explains the reasoning behind your suggestions
        4. Focuses on clarity, coherence, accuracy, and effectiveness

        Your feedback should be comprehensive but concise, and should help improve the text
        without completely rewriting it.

        Feedback:
        """

        try:
            response = self._generate(prompt)
            return response.strip()
        except Exception as e:
            raise ImproverError(f"Error generating feedback: {str(e)}")

    def _generate_value_score(self, text: str) -> float:
        """Generate a numerical value score for the text.

        This implements the value critic component from the LAC paper.

        Args:
            text: The text to evaluate.

        Returns:
            A numerical score between 0.0 and 1.0.

        Raises:
            ImproverError: If the value score cannot be generated.
        """
        prompt = f"""
        You are a language value critic as described in the paper "Language Feedback Improves
        Language Model-based Decision Making" (Fan et al., 2023).

        Your task is to evaluate the following text and assign a numerical score between 0 and 10,
        where 0 is extremely poor quality and 10 is perfect quality:

        ```
        {text}
        ```

        Consider the following criteria in your evaluation:
        1. Clarity and coherence
        2. Accuracy and factual correctness
        3. Relevance and completeness
        4. Style and tone appropriateness

        Provide only a single number between 0 and 10 as your response.

        Score (0-10):
        """

        try:
            response = self._generate(prompt)

            # Extract the numerical score
            score_text = response.strip()

            # Try to parse as a float
            try:
                # Extract just the first number from the response
                number_match = re.search(r"\d+(\.\d+)?", score_text)
                if number_match:
                    score_text = number_match.group(0)

                score = float(score_text)

                # Normalize to 0-1 range
                normalized_score = max(0.0, min(1.0, score / 10.0))
                return normalized_score
            except ValueError:
                # Default score if parsing fails
                return 0.5
        except Exception as e:
            raise ImproverError(f"Error generating value score: {str(e)}")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the LAC technique.

        This method implements the iterative improvement process from the LAC paper,
        using both feedback and value scores to guide the improvement.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        current_text = text
        current_score = critique.get("value", 0.0)
        feedback = critique.get("feedback", "")

        # Iterative improvement process
        for iteration in range(self.max_improvement_iterations):
            try:
                # Generate improved text based on feedback
                improved_text = self._generate_improved_text(current_text, feedback)

                # Evaluate the improved text
                improved_critique = self._critique(improved_text)
                improved_score = improved_critique.get("value", 0.0)

                # If the score improved, update the current text and feedback
                if improved_score > current_score:
                    current_text = improved_text
                    current_score = improved_score
                    feedback = improved_critique.get("feedback", "")
                else:
                    # No improvement, stop iterating
                    break

                # If the score is high enough, stop iterating
                if current_score >= 0.9:
                    break
            except Exception as e:
                # If an error occurs during improvement, return the current text
                break

        return current_text

    def _generate_improved_text(self, text: str, feedback: str) -> str:
        """Generate improved text based on feedback.

        Args:
            text: The text to improve.
            feedback: The feedback to guide improvement.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        prompt = f"""
        You are a language improvement agent as described in the paper "Language Feedback Improves
        Language Model-based Decision Making" (Fan et al., 2023).

        Your task is to improve the following text based on the provided feedback:

        Original text:
        ```
        {text}
        ```

        Feedback:
        {feedback}

        Please rewrite the text to address the issues identified in the feedback. Maintain the
        original meaning and intent, but improve the quality based on the feedback.

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
            raise ImproverError(f"Error generating improved text: {str(e)}")


@register_improver("lac")
def create_lac_critic(
    model: Model,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    feedback_weight: float = 0.7,
    max_improvement_iterations: int = 3,
    **options: Any,
) -> LACCritic:
    """Create a LAC critic.

    This factory function creates a LACCritic based on the paper
    "Language Feedback Improves Language Model-based Decision Making" (Fan et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        feedback_weight: Weight given to feedback critic (vs. value critic).
        max_improvement_iterations: Maximum number of improvement iterations.
        **options: Additional options to pass to the LACCritic.

    Returns:
        A LACCritic instance.
    """
    return LACCritic(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        feedback_weight=feedback_weight,
        max_improvement_iterations=max_improvement_iterations,
        **options,
    )
