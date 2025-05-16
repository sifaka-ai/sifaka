"""
Constitutional critic for evaluating text against principles.

This module provides a critic that evaluates text against a set of principles
(a "constitution") and provides feedback when violations are detected.
"""

from typing import Any, Dict, List, Optional
import json
from ..models import ModelProvider
from ..types import ValidationResult
from ..di import inject


class ConstitutionalCritic:
    """
    Critic that evaluates text against a set of principles.

    This critic analyzes text for alignment with specified principles and
    generates critiques when violations are detected. It can validate, critique,
    and improve text based on constitutional principles.
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        principles: Optional[List[str]] = None,
        system_prompt: str = "You are an expert evaluator that ensures text adheres to ethical principles.",
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize the constitutional critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            principles: List of principles to evaluate against
            system_prompt: System prompt for the model
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.principles = principles or [
            "Do not provide harmful, offensive, or biased content.",
            "Explain reasoning in a clear and truthful manner.",
            "Respect user autonomy and avoid manipulative language.",
            "Be helpful, harmless, and honest in all interactions.",
            "Provide factually accurate and well-reasoned information.",
        ]
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.kwargs = kwargs

    def validate(self, text: str, task: str = "") -> ValidationResult:
        """
        Validate text against constitutional principles.

        Args:
            text: The text to validate
            task: Optional context about what task the text is responding to

        Returns:
            A ValidationResult indicating whether the text passes constitutional principles
        """
        critique = self.critique(text, task)

        # Determine if the text passes based on score
        score = critique.get("score", 0.0)
        passed = score >= 0.7

        # Extract issues and suggestions
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])

        # Create validation result
        return ValidationResult(
            passed=passed,
            score=score,
            message=critique.get("feedback", ""),
            issues=issues,
            suggestions=suggestions,
        )

    def critique(self, text: str, task: str = "") -> dict:
        """
        Evaluate text against constitutional principles.

        Args:
            text: The text to evaluate
            task: Optional context about what task the text is responding to

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Format principles for the prompt
        principles_text = self._format_principles()

        # Include task context if provided
        task_context = f"\nContext: The text is responding to this task: {task}" if task else ""

        # Create the critique prompt
        prompt = f"""
        {self.system_prompt}

        You must evaluate the following text against these constitutional principles:
        {principles_text}
        {task_context}

        Text to evaluate:
        ---
        {text}
        ---

        Analyze the text for adherence to each principle. Be fair but thorough in your evaluation.

        Please provide your evaluation in the following JSON format:
        {{
            "score": <a number between 0.0 and 1.0, where 1.0 means all principles are satisfied>,
            "feedback": "<overall feedback>",
            "issues": ["<issue 1>", "<issue 2>", ...],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", ...],
            "principle_violations": ["<principle that was violated>", ...],
            "principle_adherence": ["<principle that was followed>", ...]
        }}

        Only respond with valid JSON.
        """

        response = self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

        # Parse response
        try:
            result = json.loads(response)
            # Ensure all required fields exist
            result.setdefault("score", 0.5)
            result.setdefault("feedback", "")
            result.setdefault("issues", [])
            result.setdefault("suggestions", [])
            result.setdefault("principle_violations", [])
            result.setdefault("principle_adherence", [])
            return result
        except json.JSONDecodeError:
            # Fallback if the model doesn't generate valid JSON
            return {
                "score": 0.5,
                "feedback": response,
                "issues": [],
                "suggestions": [],
                "principle_violations": [],
                "principle_adherence": [],
            }

    def improve(self, text: str, issues: Optional[List[str]] = None, task: str = "") -> str:
        """
        Improve text to better align with constitutional principles.

        Args:
            text: The text to improve
            issues: Optional list of issues to address
            task: Optional context about what task the text is responding to

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Format principles for the prompt
        principles_text = self._format_principles()

        # Format issues if provided
        issues_text = ""
        if issues and len(issues) > 0:
            issues_text = "Address the following issues:\n" + "\n".join(
                [f"- {issue}" for issue in issues]
            )

        # Include task context if provided
        task_context = f"\nContext: The text is responding to this task: {task}" if task else ""

        # Create the improvement prompt
        prompt = f"""
        {self.system_prompt}

        You must improve the following text to better align with these constitutional principles:
        {principles_text}
        {task_context}

        Text to improve:
        ---
        {text}
        ---

        {issues_text}

        Please revise the text to better align with all constitutional principles.
        Focus on addressing any principle violations while preserving the core message.
        Provide only the improved text without any explanations or additional commentary.
        """

        return self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

    def _format_principles(self) -> str:
        """
        Format principles for inclusion in prompts.

        Returns:
            A formatted string of principles
        """
        return "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(self.principles)])
