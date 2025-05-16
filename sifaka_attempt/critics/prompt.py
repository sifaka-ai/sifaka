"""
Prompt critic for evaluating and improving text.

This module provides a basic critic that uses a model provider to evaluate and improve text.
"""

from typing import Dict, Any, Optional, List
from ..models import ModelProvider
from ..di import inject


class PromptCritic:
    """
    Critic that uses a model provider to evaluate and improve text.

    This critic sends prompts to a model provider to evaluate and improve text.
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        instructions: str = "Evaluate the text for clarity, conciseness, and accuracy.",
        **kwargs: Any,
    ):
        """
        Initialize the prompt critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            instructions: Instructions for evaluation
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.instructions = instructions
        self.kwargs = kwargs

    def critique(self, text: str) -> dict:
        """
        Evaluate text and provide feedback.

        Args:
            text: The text to evaluate

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        prompt = f"""
        {self.instructions}

        Here is the text to evaluate:

        ---
        {text}
        ---

        Please provide your evaluation in the following JSON format:
        {{
            "score": <a number between 0.0 and 1.0>,
            "feedback": "<overall feedback>",
            "issues": ["<issue 1>", "<issue 2>", ...],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
        }}

        Only respond with valid JSON.
        """

        response = self.model_provider.generate(prompt, **self.kwargs)

        # Simple JSON parsing - in a real implementation, we would handle parsing errors
        import json

        try:
            result = json.loads(response)
            # Ensure all required fields exist
            result.setdefault("score", 0.5)
            result.setdefault("feedback", "")
            result.setdefault("issues", [])
            result.setdefault("suggestions", [])
            return result
        except json.JSONDecodeError:
            # Fallback if the model doesn't generate valid JSON
            return {"score": 0.5, "feedback": response, "issues": [], "suggestions": []}

    def improve(self, text: str, issues: Optional[List[str]] = None) -> str:
        """
        Improve text based on issues found.

        Args:
            text: The text to improve
            issues: Optional list of issues to address

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        issues_text = ""
        if issues and len(issues) > 0:
            issues_text = "Address the following issues:\n" + "\n".join(
                [f"- {issue}" for issue in issues]
            )

        prompt = f"""
        Improve the following text. {issues_text}

        ---
        {text}
        ---

        Please provide only the improved text without any other explanations or comments.
        """

        return self.model_provider.generate(prompt, **self.kwargs)
