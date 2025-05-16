"""
LAC (LLM-Based Actor-Critic) critic for text evaluation.

This module provides a critic that implements the LLM-Based Actor-Critic (LAC) approach,
combining language feedback and value scoring for comprehensive text evaluation.
"""

from typing import Any, Dict, List, Optional
import json
from ..models import ModelProvider
from ..types import ValidationResult
from ..di import inject


class LACCritic:
    """
    Critic that implements the LLM-Based Actor-Critic (LAC) approach.

    This critic combines language feedback and value scoring to improve
    language model-based decision making. It integrates both qualitative feedback
    and quantitative assessment for comprehensive text evaluation.

    Based on: Language Feedback Improves Language Model-based Decision Making
    https://arxiv.org/abs/2403.03692
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        system_prompt: str = "You are an expert evaluator providing both feedback and numerical ratings for text.",
        temperature: float = 0.7,
        feedback_template: Optional[str] = None,
        value_template: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the LAC critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            system_prompt: System prompt for the model
            temperature: Temperature for generation
            feedback_template: Custom template for feedback generation
            value_template: Custom template for value scoring
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.kwargs = kwargs

        # Default templates
        self.feedback_template = feedback_template or (
            "Provide detailed feedback on the following response to the task below.\n\n"
            "Task: {task}\n\n"
            "Response:\n{response}\n\n"
            "Your feedback should be constructive, specific, and highlight both strengths and areas for improvement."
        )

        self.value_template = value_template or (
            "Rate the quality of the following response to the task on a scale from 0.0 to 1.0, "
            "where 1.0 is perfect and 0.0 is completely inadequate.\n\n"
            "Task: {task}\n\n"
            "Response:\n{response}\n\n"
            "Please provide only a number between 0.0 and 1.0 without any explanation."
        )

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against quality criteria.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult indicating whether the text passes quality criteria
        """
        # LAC critics focus on providing feedback rather than validation
        # Default to passing validation
        critique = self.critique(text)

        # Use the value score
        score = critique.get("score", 0.7)
        passed = score >= 0.7

        # Extract feedback and suggestions
        feedback = critique.get("feedback", "")

        # Create validation result
        return ValidationResult(
            passed=passed,
            score=score,
            message=feedback,
            issues=[],
            suggestions=[],
        )

    def critique(self, text: str, task: str = "Evaluate the text") -> dict:
        """
        Evaluate text and provide both feedback and value.

        Args:
            text: The text to evaluate
            task: Optional task context

        Returns:
            A dictionary with feedback and value score
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Generate qualitative feedback
        feedback_prompt = self.feedback_template.format(
            task=task,
            response=text,
        )

        feedback = self.model_provider.generate(
            feedback_prompt,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            **self.kwargs,
        )

        # Generate quantitative value
        value_prompt = self.value_template.format(
            task=task,
            response=text,
        )

        value_response = self.model_provider.generate(
            value_prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,  # Lower temperature for more consistent scoring
            **self.kwargs,
        )

        # Extract the numeric value
        try:
            # Try to extract a float from the response
            # First look for a pattern like "0.8" or "0.75"
            import re

            value_match = re.search(r"(\d+\.\d+|\d+)", value_response)
            if value_match:
                value = float(value_match.group(1))
                # Clamp to valid range
                value = max(0.0, min(1.0, value))
            else:
                value = 0.7  # Default value
        except (ValueError, TypeError):
            value = 0.7  # Default value if parsing fails

        # Create result
        return {
            "score": value,
            "feedback": feedback,
            "issues": [],
            "suggestions": [],
        }

    def improve(self, text: str, issues: Optional[List[str]] = None) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            issues: Optional list of issues to address

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # First generate feedback if not provided through issues
        task = "Improve the text"

        if not issues or len(issues) == 0:
            critique_result = self.critique(text, task)
            feedback = critique_result.get("feedback", "")
        else:
            feedback = "\n".join([f"- {issue}" for issue in issues])

        # Create improvement prompt
        improvement_prompt = f"""
        {self.system_prompt}

        Improve the following text based on this feedback:

        Text to improve:
        ---
        {text}
        ---

        Feedback:
        {feedback}

        Provide only the improved text without any explanations or additional commentary.
        """

        improved_text = self.model_provider.generate(
            improvement_prompt, temperature=self.temperature, **self.kwargs
        )

        return improved_text
