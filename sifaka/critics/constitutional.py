"""Constitutional critic for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on Constitutional AI:

@misc{bai2022constitutionalaiharmlessnessai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.08073},
}

The ConstitutionalCritic evaluates text against a set of predefined principles
and provides detailed feedback on principle violations, ensuring that generated
content aligns with ethical guidelines and quality standards.
"""

import re
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

            # Calculate confidence based on violations found
            confidence = 1.0 - (len(violations) / len(self.principles)) if self.principles else 1.0
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            logger.debug(
                f"ConstitutionalCritic: Critique completed in {processing_time:.2f}ms, "
                f"found {len(violations)} violations, confidence: {confidence:.2f}"
            )

            return {
                "needs_improvement": needs_improvement,
                "violations": [v.get("description", str(v)) for v in violations],
                "suggestions": self._extract_suggestions(critique_response),
                "confidence": confidence,
                "feedback": {
                    "critique": critique_response,
                    "principle_violations": violations,
                    "principles_evaluated": len(self.principles),
                    "strict_mode": self.strict_mode,
                },
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

    async def _critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Critique text against constitutional principles asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync critique method but with non-blocking I/O.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results including principle violations.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ConstitutionalCritic",
            operation="critique_async",
            message_prefix="Failed to critique text with Constitutional AI (async)",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "violations": ["No text available for critique"],
                    "suggestions": ["Provide text to critique"],
                    "confidence": 0.0,
                    "feedback": {
                        "critique": "No text provided for critique",
                        "principle_violations": [],
                        "principles_evaluated": len(self.principles),
                        "strict_mode": self.strict_mode,
                    },
                }

            # Prepare context from retrieved documents (using mixin)
            context = self._prepare_context(thought)

            # Create principles text
            principles_text = "\n".join([f"- {principle}" for principle in self.principles])

            # Create critique prompt with context
            critique_prompt = self.critique_prompt_template.format(
                principles=principles_text,
                prompt=thought.prompt,
                text=thought.text,
                context=context,
            )

            # Generate critique (async)
            critique_response = await self.model._generate_async(
                prompt=critique_prompt,
                system_message="You are an expert evaluator of constitutional principles and ethical guidelines.",
            )

            # Parse violations and suggestions from critique
            violations = self._extract_violations(critique_response)
            suggestions = self._extract_suggestions(critique_response)

            # Determine if improvement is needed
            needs_improvement = len(violations) > 0
            if self.strict_mode:
                needs_improvement = needs_improvement or "concern" in critique_response.lower()

            # Calculate confidence based on clarity of violations
            confidence = 0.9 if violations else 0.7

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"ConstitutionalCritic: Async critique completed in {processing_time:.2f}ms"
            )

            return {
                "needs_improvement": needs_improvement,
                "violations": [v.get("description", str(v)) for v in violations],
                "suggestions": suggestions,
                "confidence": confidence,
                "feedback": {
                    "critique": critique_response,
                    "principle_violations": violations,
                    "principles_evaluated": len(self.principles),
                    "strict_mode": self.strict_mode,
                },
                "processing_time_ms": processing_time,
            }

    async def _improve_async(self, thought: Thought) -> str:
        """Improve text to align with constitutional principles asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that better aligns with constitutional principles.
        """
        start_time = time.time()

        with critic_context(
            critic_name="ConstitutionalCritic",
            operation="improve_async",
            message_prefix="Failed to improve text with Constitutional AI (async)",
        ):
            # Check if text and critique are available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="ConstitutionalCritic",
                    operation="improve_async",
                    suggestions=["Provide text to improve"],
                )

            if not thought.critic_feedback:
                raise ImproverError(
                    message="No critique available for improvement",
                    component="ConstitutionalCritic",
                    operation="improve_async",
                    suggestions=["Run critique before improvement"],
                )

            # Extract critique from feedback
            critique = ""
            for feedback in thought.critic_feedback:
                if feedback.critic_name == "ConstitutionalCritic":
                    critique = feedback.feedback.get("critique", "")
                    break

            if not critique:
                critique = "Please improve the text to better align with constitutional principles."

            # Create principles text
            principles_text = "\n".join([f"- {principle}" for principle in self.principles])

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

            # Generate improved text (async)
            improved_text = await self.model._generate_async(
                prompt=improve_prompt,
                system_message="You are an expert editor improving text to align with constitutional principles.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"ConstitutionalCritic: Async improvement completed in {processing_time:.2f}ms"
            )

            return improved_text.strip()

    def _extract_violations(self, critique: str) -> List[Dict[str, Any]]:
        """Extract principle violations from critique text.

        This is a fallback method for when the model doesn't return structured JSON.
        The preferred approach is to get structured data directly from the model.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of violation dictionaries.
        """
        # Try to find JSON in the critique first
        import json

        try:
            # Look for JSON blocks in the critique
            if "{" in critique and "}" in critique:
                start = critique.find("{")
                end = critique.rfind("}") + 1
                json_str = critique[start:end]
                data = json.loads(json_str)

                # Extract violations from structured data
                violations = []
                if "violations" in data:
                    for violation in data["violations"]:
                        if isinstance(violation, dict):
                            violations.append(violation)
                        else:
                            # Convert string to dict format
                            violations.append(
                                {
                                    "type": "principle_violation",
                                    "description": str(violation),
                                    "severity": "unknown",
                                }
                            )
                return violations
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple text parsing (much simpler than before)
        violations = []
        lines = critique.split("\n")

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Look for lines that clearly indicate violations
            line_lower = line_clean.lower()
            if any(
                phrase in line_lower
                for phrase in [
                    "violation:",
                    "violates principle",
                    "does not adhere to",
                    "fails to meet",
                ]
            ):
                # Skip obvious non-violations
                if any(
                    phrase in line_lower
                    for phrase in [
                        "no violation",
                        "violation: none",
                        "no violations found",
                        "suggestions for addressing",
                        "addressing the violation",
                        "to address the violation",
                    ]
                ):
                    continue

                violations.append(
                    {
                        "type": "principle_violation",
                        "description": line_clean,
                        "severity": "unknown",  # Let the model specify severity
                    }
                )

        return violations

    def _extract_suggestions(self, critique: str) -> List[str]:
        """Extract suggestions from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of suggestions for improvement.
        """
        suggestions = []
        critique_lines = critique.split("\n")

        # Look for numbered principles and their violations
        for i, line in enumerate(critique_lines):
            line_stripped = line.strip()

            # Check for numbered principles (e.g., "1. The content should...")
            if re.match(r"^\d+\.\s+", line_stripped):
                suggestions.append(line_stripped)

                # Check the next few lines for violation details
                for j in range(i + 1, min(i + 3, len(critique_lines))):
                    next_line = critique_lines[j].strip()
                    if next_line.startswith("-") and (
                        "not met" in next_line.lower() or "violation" in next_line.lower()
                    ):
                        # This principle has a violation, make sure it's included
                        if line_stripped not in suggestions:
                            suggestions.append(line_stripped)
                        break

        # Look for suggestion indicators in remaining lines
        suggestion_indicators = [
            "suggest",
            "recommend",
            "should",
            "could",
            "might",
            "consider",
            "try",
            "improve",
            "address",
            "rewrite",
        ]

        for line in critique_lines:
            line_lower = line.lower().strip()
            if any(indicator in line_lower for indicator in suggestion_indicators):
                # Clean up the suggestion
                suggestion = line.strip()
                if (
                    suggestion
                    and suggestion not in suggestions
                    and not re.match(r"^\d+\.\s+", suggestion)
                ):
                    suggestions.append(suggestion)

        # If no specific suggestions found, provide generic ones
        if not suggestions:
            suggestions = [
                "Review the text against constitutional principles",
                "Ensure factual accuracy and balanced perspective",
                "Consider ethical implications of the content",
            ]

        return suggestions[:5]  # Limit to 5 suggestions


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
