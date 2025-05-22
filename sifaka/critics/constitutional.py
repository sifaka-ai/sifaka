"""Constitutional critic for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on Constitutional AI: https://arxiv.org/abs/2212.08073

The ConstitutionalCritic evaluates text against a set of predefined principles
and provides detailed feedback on principle violations, ensuring that generated
content aligns with ethical guidelines and quality standards.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class ConstitutionalCritic(ContextAwareMixin):
    """Critic that evaluates text against constitutional principles.

    This critic implements the Constitutional AI approach by evaluating text
    against a set of predefined principles (a "constitution"). It provides
    detailed feedback on principle violations and suggests improvements.

    Attributes:
        model: The language model to use for critique and improvement.
        principles: List of constitutional principles to evaluate against.
        strict_mode: Whether to require all principles to be satisfied.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        principles: Optional[List[str]] = None,
        strict_mode: bool = False,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Constitutional critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            principles: List of constitutional principles to evaluate against.
            strict_mode: Whether to require all principles to be satisfied.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        # Set up the model
        if model:
            self.model = model
        elif model_name:
            self.model = create_model(model_name, **model_kwargs)
        else:
            # Default to a mock model for testing
            self.model = create_model("mock:default", **model_kwargs)

        # Set up principles (constitution)
        self.principles = principles or [
            "Do not provide harmful, offensive, or biased content.",
            "Explain reasoning in a clear and truthful manner.",
            "Respect user autonomy and avoid manipulative language.",
            "Provide accurate and factual information.",
            "Be helpful and constructive in responses.",
        ]

        self.strict_mode = strict_mode

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Evaluate the following text against the constitutional principles provided.\n\n"
            "Constitutional Principles:\n{principles}\n\n"
            "Original task: {prompt}\n\n"
            "Text to evaluate:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please analyze the text against each principle and provide:\n"
            "1. Which principles (if any) are violated\n"
            "2. Specific examples of violations\n"
            "3. The severity of each violation\n"
            "4. Suggestions for addressing violations\n"
            "5. How well the text uses factual information from the context (if available)\n\n"
            "If the text adheres to all principles, please state that clearly.\n"
            "Be specific and constructive in your feedback."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text to better align with the constitutional principles.\n\n"
            "Constitutional Principles:\n{principles}\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Constitutional critique:\n{critique}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses all principle violations identified in the critique\n"
            "2. Maintains the core message and usefulness of the response\n"
            "3. Fully adheres to all constitutional principles\n"
            "4. Remains helpful and relevant to the original task\n"
            "5. Better incorporates factual information from the context (if available)"
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text against constitutional principles.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results including principle violations.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ConstitutionalCritic",
            operation="critique",
            message_prefix="Failed to critique text with Constitutional principles",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "message": "No text available for critique",
                    "issues": ["Text is empty or None"],
                    "suggestions": ["Provide text to critique"],
                    "principle_violations": [],
                }

            # Format principles for the prompt
            principles_text = "\n".join(
                f"{i+1}. {principle}" for i, principle in enumerate(self.principles)
            )

            # Prepare context from retrieved documents (using mixin)
            context = self._prepare_context(thought)

            # Create critique prompt with context
            critique_prompt = self.critique_prompt_template.format(
                principles=principles_text,
                prompt=thought.prompt,
                text=thought.text,
                context=context,
            )

            # Generate critique
            critique_response = self.model.generate(
                prompt=critique_prompt,
                system_prompt="You are an expert constitutional AI evaluator analyzing text for principle violations.",
            )

            # Analyze critique for violations
            violations = self._extract_violations(critique_response)
            needs_improvement = len(violations) > 0

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"ConstitutionalCritic: Critique completed in {processing_time:.2f}ms, "
                f"found {len(violations)} violations"
            )

            return {
                "needs_improvement": needs_improvement,
                "message": critique_response,
                "critique": critique_response,
                "principle_violations": violations,
                "principles_evaluated": len(self.principles),
                "strict_mode": self.strict_mode,
                "processing_time_ms": processing_time,
            }

    def improve(self, thought: Thought) -> str:
        """Improve text to align with constitutional principles.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that better aligns with constitutional principles.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ConstitutionalCritic",
            operation="improve",
            message_prefix="Failed to improve text with Constitutional principles",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="ConstitutionalCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            critique = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "ConstitutionalCritic":
                        critique = feedback.feedback.get("critique", "")
                        break

            # Format principles for the prompt
            principles_text = "\n".join(
                f"{i+1}. {principle}" for i, principle in enumerate(self.principles)
            )

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with context
            improve_prompt = self.improve_prompt_template.format(
                principles=principles_text,
                prompt=thought.prompt,
                text=thought.text,
                critique=critique,
                context=context,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor improving text to align with constitutional principles.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"ConstitutionalCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _extract_violations(self, critique: str) -> List[Dict[str, Any]]:
        """Extract principle violations from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of violation dictionaries.
        """
        violations = []
        critique_lower = critique.lower()

        # Simple heuristic to detect violations
        violation_indicators = [
            "violates",
            "violation",
            "does not adhere",
            "fails to",
            "problematic",
            "concerning",
            "inappropriate",
            "harmful",
            "biased",
            "misleading",
            "inaccurate",
        ]

        adherence_indicators = [
            "adheres to all principles",
            "no violations",
            "follows all principles",
            "complies with",
            "aligns with all",
            "satisfies all principles",
        ]

        # Check for explicit adherence statements
        for indicator in adherence_indicators:
            if indicator in critique_lower:
                return []  # No violations found

        # Check for violation indicators
        for indicator in violation_indicators:
            if indicator in critique_lower:
                # Extract context around the violation
                lines = critique.split("\n")
                for line in lines:
                    if indicator in line.lower():
                        violations.append(
                            {
                                "type": "principle_violation",
                                "description": line.strip(),
                                "severity": "medium",  # Default severity
                            }
                        )
                break

        return violations


def create_constitutional_critic(
    model: Optional[Model] = None,
    model_name: Optional[str] = None,
    principles: Optional[List[str]] = None,
    strict_mode: bool = False,
    **model_kwargs: Any,
) -> ConstitutionalCritic:
    """Create a Constitutional critic.

    Args:
        model: The language model to use for critique and improvement.
        model_name: The name of the model to use if model is not provided.
        principles: List of constitutional principles to evaluate against.
        strict_mode: Whether to require all principles to be satisfied.
        **model_kwargs: Additional keyword arguments for model creation.

    Returns:
        A ConstitutionalCritic instance.
    """
    return ConstitutionalCritic(
        model=model,
        model_name=model_name,
        principles=principles,
        strict_mode=strict_mode,
        **model_kwargs,
    )
