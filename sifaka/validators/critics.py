"""
LLM-based critics for Sifaka.

This module provides critics that use LLMs to validate and improve text.
These critics follow the Validator and Improver protocols defined in the chain module.
"""

import re
from typing import Optional, Any, Tuple, Union

from sifaka.results import ValidationResult, ImprovementResult
from sifaka.models.base import Model, create_model


class Critic:
    """Base class for LLM-based critics.

    Critics can both validate and improve text using LLMs.
    """

    def __init__(self, model: Union[str, Model], name: Optional[str] = None, **model_options: Any):
        """Initialize a critic.

        Args:
            model: Model to use for validation and improvement.
                Can be a model instance or a string in the format "provider:model_name".
            name: Name of the critic. If not provided, will be derived from the class name.
            **model_options: Additional options to pass to the model.
        """
        self.name = name or self.__class__.__name__

        # Initialize model
        if isinstance(model, str):
            provider, model_name = model.split(":", 1)
            self.model = create_model(provider, model_name)
        else:
            self.model = model

        self.model_options = model_options

    def validate(self, text: str) -> ValidationResult:
        """Validate text using the critic.

        Args:
            text: Text to validate.

        Returns:
            ValidationResult indicating whether the text passed validation.
        """
        raise NotImplementedError("Subclasses must implement validate")

    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text using the critic.

        Args:
            text: Text to improve.

        Returns:
            Tuple of (improved_text, ImprovementResult).
        """
        raise NotImplementedError("Subclasses must implement improve")


class ClarityAndCoherenceCritic(Critic):
    """Critic that evaluates and improves text clarity and coherence."""

    def validate(self, text: str) -> ValidationResult:
        """Validate text clarity and coherence.

        Args:
            text: Text to validate.

        Returns:
            ValidationResult indicating whether the text is clear and coherent.
        """
        prompt = f"""
        Evaluate the clarity and coherence of the following text.
        Consider factors such as:
        - Logical flow and organization
        - Clear explanations of concepts
        - Appropriate transitions between ideas
        - Consistent terminology
        - Absence of ambiguity

        Text to evaluate:
        ---
        {text}
        ---

        First, provide a score from 1-10 where:
        1-3: Poor clarity and coherence
        4-6: Moderate clarity and coherence
        7-10: Excellent clarity and coherence

        Then, explain your reasoning in detail.

        Format your response as:
        SCORE: [your score]
        REASONING: [your detailed explanation]
        PASSED: [YES if score >= 7, NO if score < 7]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract score and passed status
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        passed_match = re.search(r"PASSED:\s*(YES|NO)", response)
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?=PASSED:|$)", response, re.DOTALL)

        if not score_match or not passed_match:
            return ValidationResult(
                passed=False,
                message="Could not determine validation result from critic response",
                details={"response": response},
            )

        score = int(score_match.group(1))
        passed = passed_match.group(1) == "YES"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return ValidationResult(
            passed=passed,
            message=f"Clarity and coherence score: {score}/10. {reasoning}",
            details={"score": score, "reasoning": reasoning, "response": response},
        )

    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve text clarity and coherence.

        Args:
            text: Text to improve.

        Returns:
            Tuple of (improved_text, ImprovementResult).
        """
        prompt = f"""
        Improve the clarity and coherence of the following text.
        Focus on:
        - Enhancing logical flow and organization
        - Clarifying explanations of concepts
        - Improving transitions between ideas
        - Ensuring consistent terminology
        - Reducing ambiguity

        Text to improve:
        ---
        {text}
        ---

        First, rewrite the text to improve clarity and coherence.
        Then, explain the changes you made and why they improve the text.

        Format your response as:
        IMPROVED TEXT:
        [your improved version of the text]

        EXPLANATION:
        [explanation of changes made]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract improved text and explanation
        text_match = re.search(r"IMPROVED TEXT:\s*(.*?)(?=EXPLANATION:|$)", response, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)$", response, re.DOTALL)

        if not text_match:
            return text, ImprovementResult(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Could not extract improved text from critic response",
                details={"response": response},
            )

        improved_text = text_match.group(1).strip()
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # Check if the text was actually changed
        changes_made = improved_text != text

        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=changes_made,
            message=f"Clarity and coherence improvements: {explanation}",
            details={"explanation": explanation, "response": response},
        )


class FactualAccuracyCritic(Critic):
    """Critic that evaluates and improves factual accuracy of text."""

    def validate(self, text: str) -> ValidationResult:
        """Validate factual accuracy of text.

        Args:
            text: Text to validate.

        Returns:
            ValidationResult indicating whether the text is factually accurate.
        """
        prompt = f"""
        Evaluate the factual accuracy of the following text.
        Identify any factual errors, unsupported claims, or misleading statements.

        Text to evaluate:
        ---
        {text}
        ---

        First, list any factual errors or unsupported claims you find.
        Then, provide an overall assessment of the factual accuracy.

        Format your response as:
        ERRORS: [list of factual errors or "None found" if none]
        ASSESSMENT: [your overall assessment]
        PASSED: [YES if no significant factual errors, NO if significant errors found]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract errors and passed status
        errors_match = re.search(r"ERRORS:\s*(.*?)(?=ASSESSMENT:|$)", response, re.DOTALL)
        passed_match = re.search(r"PASSED:\s*(YES|NO)", response)
        assessment_match = re.search(r"ASSESSMENT:\s*(.*?)(?=PASSED:|$)", response, re.DOTALL)

        if not passed_match:
            return ValidationResult(
                passed=False,
                message="Could not determine validation result from critic response",
                details={"response": response},
            )

        passed = passed_match.group(1) == "YES"
        errors = errors_match.group(1).strip() if errors_match else ""
        assessment = assessment_match.group(1).strip() if assessment_match else ""

        if passed:
            message = f"Text appears factually accurate. {assessment}"
        else:
            message = f"Factual errors detected: {errors}. {assessment}"

        return ValidationResult(
            passed=passed,
            message=message,
            details={"errors": errors, "assessment": assessment, "response": response},
        )

    def improve(self, text: str) -> Tuple[str, ImprovementResult]:
        """Improve factual accuracy of text.

        Args:
            text: Text to improve.

        Returns:
            Tuple of (improved_text, ImprovementResult).
        """
        prompt = f"""
        Improve the factual accuracy of the following text.
        Correct any factual errors, unsupported claims, or misleading statements.

        Text to improve:
        ---
        {text}
        ---

        First, rewrite the text to correct factual errors.
        Then, explain the corrections you made.

        Format your response as:
        IMPROVED TEXT:
        [your improved version of the text]

        CORRECTIONS:
        [explanation of factual corrections made]
        """

        response = self.model.generate(prompt, **self.model_options)

        # Extract improved text and corrections
        text_match = re.search(r"IMPROVED TEXT:\s*(.*?)(?=CORRECTIONS:|$)", response, re.DOTALL)
        corrections_match = re.search(r"CORRECTIONS:\s*(.*?)$", response, re.DOTALL)

        if not text_match:
            return text, ImprovementResult(
                original_text=text,
                improved_text=text,
                changes_made=False,
                message="Could not extract improved text from critic response",
                details={"response": response},
            )

        improved_text = text_match.group(1).strip()
        corrections = corrections_match.group(1).strip() if corrections_match else ""

        # Check if the text was actually changed
        changes_made = improved_text != text

        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=changes_made,
            message=f"Factual accuracy improvements: {corrections}",
            details={"corrections": corrections, "response": response},
        )
