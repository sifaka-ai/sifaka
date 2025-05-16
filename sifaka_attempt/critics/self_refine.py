"""
Self-Refine critic for iteratively improving text through refinement.

This module provides a critic that repeatedly critiques and refines text
to improve its quality through multiple iterations.
"""

from typing import Any, Dict, List, Optional
import json
import time
from ..models import ModelProvider
from ..types import ValidationResult
from ..di import inject


class SelfRefineCritic:
    """
    Critic that uses iterative self-refinement to improve text.

    This critic uses an iterative process of critique and refinement
    to progressively improve the quality of text, stopping when quality
    thresholds are met or maximum iterations are reached.
    """

    @inject(model_provider="model.openai")
    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        system_prompt: str = "You are an expert editor that improves text through iterative refinement.",
        max_iterations: int = 3,
        min_improvement_threshold: float = 0.1,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize the self-refine critic.

        Args:
            model_provider: Model provider to use (injected if not provided)
            system_prompt: System prompt for the model
            max_iterations: Maximum number of refinement iterations
            min_improvement_threshold: Minimum improvement required to continue refining
            temperature: Temperature for generation
            **kwargs: Additional arguments to pass to the model provider
        """
        if not model_provider:
            from ..di import resolve

            model_provider = resolve("model.openai")

        self.model_provider = model_provider
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.min_improvement_threshold = min_improvement_threshold
        self.temperature = temperature
        self.kwargs = kwargs

        # Default prompt templates
        self.critique_prompt_template = (
            "Please critique the following response to the task. "
            "Focus on accuracy, clarity, and completeness.\n\n"
            "Task:\n{task}\n\n"
            "Response:\n{response}\n\n"
            "Critique:"
        )

        self.revision_prompt_template = (
            "Please revise the following response based on the critique.\n\n"
            "Task:\n{task}\n\n"
            "Response:\n{response}\n\n"
            "Critique:\n{critique}\n\n"
            "Revised response:"
        )

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

    def critique(self, text: str, task: str = "Analyze the following text") -> dict:
        """
        Evaluate text and provide feedback.

        Args:
            text: The text to evaluate
            task: Optional task context

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Create the critique prompt
        prompt = self.critique_prompt_template.format(
            task=task,
            response=text,
        )

        response = self.model_provider.generate(
            prompt, temperature=self.temperature, system_prompt=self.system_prompt, **self.kwargs
        )

        # Parse response
        issues = []
        suggestions = []

        # Extract issues and suggestions from critique
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                if (
                    "should" in line.lower()
                    or "could" in line.lower()
                    or "recommend" in line.lower()
                ):
                    suggestions.append(line[2:])
                else:
                    issues.append(line[2:])

        # Calculate score based on issues
        score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) * 0.1))

        # Create result
        return {
            "score": score,
            "feedback": response,
            "issues": issues,
            "suggestions": suggestions,
        }

    def improve(
        self,
        text: str,
        issues: Optional[List[str]] = None,
        task: str = "Improve the following text",
    ) -> str:
        """
        Improve text based on issues found.

        Args:
            text: The text to improve
            issues: Optional list of issues to address
            task: Optional task context

        Returns:
            Improved text
        """
        if not self.model_provider:
            raise ValueError("No model provider available")

        # Format issues if provided
        task_with_issues = task
        if issues and len(issues) > 0:
            task_with_issues = f"{task}\nAddress the following issues:\n" + "\n".join(
                [f"- {issue}" for issue in issues]
            )

        # Perform iterative refinement
        current_output = text
        last_score = 0.0

        for iteration in range(self.max_iterations):
            # Step 1: Critique the current output
            critique_prompt = self.critique_prompt_template.format(
                task=task_with_issues,
                response=current_output,
            )

            critique = self.model_provider.generate(
                critique_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                **self.kwargs,
            ).strip()

            # Simple heuristic stopping condition
            no_issues_phrases = [
                "no issues",
                "looks good",
                "well written",
                "excellent",
                "great job",
                "perfect",
            ]
            if any(phrase in critique.lower() for phrase in no_issues_phrases):
                break

            # Step 2: Revise using the critique
            revision_prompt = self.revision_prompt_template.format(
                task=task_with_issues,
                response=current_output,
                critique=critique,
            )

            revised_output = self.model_provider.generate(
                revision_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                **self.kwargs,
            ).strip()

            # Step 3: Check if improvement is significant
            critique_result = self.critique(revised_output, task)
            current_score = critique_result.get("score", 0.0)

            # Update output if better
            if current_score > last_score:
                improvement = current_score - last_score
                current_output = revised_output
                last_score = current_score

                # Stop if improvement is below threshold
                if improvement < self.min_improvement_threshold:
                    break
            else:
                # No improvement, keep the previous version
                break

        return current_output
