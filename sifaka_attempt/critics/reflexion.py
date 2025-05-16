"""
Reflexion critic for improving text through reflection.

This module provides a critic that uses reflection on past feedback to improve text.
"""

from typing import Any, Dict, List, Optional
import json
import time
from ..models import ModelProvider
from ..types import ValidationResult
from ..di import inject


class ReflexionCritic:
    """
    Critic that uses reflection on past feedback to improve text.

    This critic maintains a memory of past reflections and incorporates them
    into future improvements, enabling a form of learning without model weight updates.
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        system_prompt: str = "You are an expert editor that improves text through reflection.",
        memory_buffer_size: int = 5,
        reflection_depth: int = 1,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize the reflexion critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            system_prompt: System prompt for the model
            memory_buffer_size: Maximum number of reflections to keep in memory
            reflection_depth: How many past reflections to include in prompts
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.memory_buffer_size = memory_buffer_size
        self.reflection_depth = reflection_depth
        self.temperature = temperature
        self.kwargs = kwargs
        self.reflection_memory = []

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against quality criteria.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult indicating whether the text passes quality criteria
        """
        critique = self.critique(text)

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

    def critique(self, text: str) -> dict:
        """
        Evaluate text and provide feedback with reflection.

        Args:
            text: The text to evaluate

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Prepare reflections context from memory
        reflections_context = self._get_reflections_context()

        # Create the critique prompt
        prompt = f"""
        {self.system_prompt}

        {reflections_context}

        Here is the text to evaluate:

        ---
        {text}
        ---

        Please provide your evaluation in the following JSON format:
        {{
            "score": <a number between 0.0 and 1.0>,
            "feedback": "<overall feedback>",
            "issues": ["<issue 1>", "<issue 2>", ...],
            "suggestions": ["<suggestion 1>", "<suggestion 2>", ...],
            "reflection": "<your reflection on what could be improved and why>"
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
            result.setdefault("reflection", "")

            # Store the reflection in memory if available
            if "reflection" in result and result["reflection"]:
                self._add_reflection(text, result["reflection"])

            return result
        except json.JSONDecodeError:
            # Fallback if the model doesn't generate valid JSON
            return {
                "score": 0.5,
                "feedback": response,
                "issues": [],
                "suggestions": [],
                "reflection": "",
            }

    def improve(self, text: str, issues: Optional[List[str]] = None) -> str:
        """
        Improve text based on issues found, using reflections.

        Args:
            text: The text to improve
            issues: Optional list of issues to address

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Prepare reflections context from memory
        reflections_context = self._get_reflections_context()

        # Format issues if provided
        issues_text = ""
        if issues and len(issues) > 0:
            issues_text = "Address the following issues:\n" + "\n".join(
                [f"- {issue}" for issue in issues]
            )

        # Create the improvement prompt
        prompt = f"""
        {self.system_prompt}

        {reflections_context}

        Improve the following text. {issues_text}

        ---
        {text}
        ---

        Please provide the improved text followed by a reflection on what you improved and why.
        Format your response as:

        IMPROVED TEXT:
        <your improved version>

        REFLECTION:
        <your reflection>
        """

        response = self.model_provider.generate(prompt, temperature=self.temperature, **self.kwargs)

        # Extract the improved text and reflection
        improved_text = response
        reflection = ""

        # Try to extract improved text and reflection based on the format
        if "IMPROVED TEXT:" in response and "REFLECTION:" in response:
            parts = response.split("REFLECTION:", 1)
            improved_text_part = parts[0]
            reflection = parts[1].strip() if len(parts) > 1 else ""

            # Extract just the improved text without the header
            if "IMPROVED TEXT:" in improved_text_part:
                improved_text = improved_text_part.split("IMPROVED TEXT:", 1)[1].strip()

        # Store the reflection in memory if available
        if reflection:
            self._add_reflection(text, reflection)

        return improved_text

    def _add_reflection(self, text: str, reflection: str) -> None:
        """
        Add a reflection to the memory buffer.

        Args:
            text: The text that was reflected on
            reflection: The reflection to add
        """
        # Create a reflection entry
        entry = {
            "timestamp": time.time(),
            "text_sample": text[:100] + "..." if len(text) > 100 else text,
            "reflection": reflection,
        }

        # Add to memory and maintain buffer size
        self.reflection_memory.append(entry)
        if len(self.reflection_memory) > self.memory_buffer_size:
            self.reflection_memory = self.reflection_memory[-self.memory_buffer_size :]

    def _get_reflections_context(self) -> str:
        """
        Get the context of past reflections for the prompt.

        Returns:
            A string with past reflections to include in the prompt
        """
        if not self.reflection_memory:
            return ""

        # Get relevant reflections based on depth
        relevant_reflections = self.reflection_memory[-self.reflection_depth :]

        # Format reflections as context
        context = "Previous reflections for your reference:\n\n"
        for i, entry in enumerate(relevant_reflections):
            context += f"Reflection {i+1}:\n"
            context += f"Text sample: {entry['text_sample']}\n"
            context += f"Reflection: {entry['reflection']}\n\n"

        return context
