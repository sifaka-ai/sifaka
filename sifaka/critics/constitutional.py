"""Constitutional critic for Sifaka.

This module implements a Constitutional AI approach for critics, which evaluates
responses against a set of human-written principles (a "constitution") and provides
natural language feedback when violations are detected.

Based on "Constitutional AI: Harmlessness from AI Feedback":
https://arxiv.org/abs/2212.08073

@misc{bai2022constitutionalaiharmlessnessai,
      title={Constitutional AI: Harmlessness from AI Feedback},
      author={Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan},
      year={2022},
      eprint={2212.08073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.08073},
}

The ConstitutionalCritic implements core Constitutional AI concepts:
1. Principle-based evaluation against a written constitution
2. Natural language feedback on principle violations
3. Iterative improvement through constitutional critique
4. Harmlessness assessment through AI feedback

IMPORTANT IMPLEMENTATION CAVEAT:
This implementation adapts the core Constitutional AI principle-based evaluation
for text critique. The original Constitutional AI paper focuses on training methodology
for creating more aligned models through two phases:
1. Supervised Learning: Generate → Critique → Revise → Train on revisions
2. RL from AI Feedback: Generate multiple responses → AI evaluation → RL training

This implementation extracts the "Critique → Revise" component for single-text
improvement scenarios, providing practical value for content moderation and
ethical AI applications without requiring model training.
"""

import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.critics.mixins.validation_aware import ValidationAwareMixin
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin
from sifaka.validators.validation_context import create_validation_context

logger = get_logger(__name__)


class ConstitutionalCritic(BaseCritic, ValidationAwareMixin, ContextAwareMixin):
    """Critic that evaluates text against constitutional principles with validation awareness.

    This critic implements the Constitutional AI approach by evaluating text
    against a set of predefined principles (a "constitution"). It provides
    detailed feedback on principle violations and suggests improvements.

    Enhanced with validation context awareness to prioritize validation constraints
    over conflicting constitutional suggestions.
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
        super().__init__(model=model, model_name=model_name, **model_kwargs)

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
            "Please analyze the text against each principle and provide:\n\n"
            "Issues:\n- [List any principle violations here]\n\n"
            "Suggestions:\n- [List specific suggestions for addressing violations]\n\n"
            "Overall Assessment: [Brief assessment of constitutional compliance]\n\n"
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
            "5. Better incorporates factual information from the context (if available)\n\n"
            "Improved text:"
        )

        # Store the last improvement prompt used for debugging/logging
        self.last_improvement_prompt = None

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using Constitutional AI approach asynchronously.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Format principles for the prompt
        principles_text = "\n".join(
            f"{i + 1}. {principle}" for i, principle in enumerate(self.principles)
        )

        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt
        critique_prompt = self.critique_prompt_template.format(
            principles=principles_text,
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique asynchronously (async only)
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_prompt="You are an expert constitutional AI evaluator. Assess text against constitutional principles.",
        )

        # Parse the critique
        issues, suggestions = self._parse_critique(critique_response)
        violations = self._extract_violations(critique_response)

        # Determine if improvement is needed
        needs_improvement = len(violations) > 0 or len(issues) > 0
        if self.strict_mode:
            needs_improvement = needs_improvement or "concern" in critique_response.lower()

        # Calculate confidence based on violations found
        confidence = 1.0 - (len(violations) / len(self.principles)) if self.principles else 1.0
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        logger.debug(
            f"ConstitutionalCritic: Async critique found {len(violations)} violations, confidence: {confidence:.2f}"
        )

        return {
            "needs_improvement": needs_improvement,
            "message": critique_response,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": confidence,
            "metadata": {
                "principle_violations": violations,
                "principles_evaluated": len(self.principles),
                "strict_mode": self.strict_mode,
            },
        }

    async def improve_async(self, thought: Thought) -> str:
        """Improve text to align with constitutional principles asynchronously.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text that better aligns with constitutional principles.

        Raises:
            ImproverError: If the improvement fails.
        """
        # Use the enhanced method with validation context from thought
        validation_context = create_validation_context(getattr(thought, "validation_results", None))
        return await self.improve_with_validation_context_async(thought, validation_context)

    async def improve_with_validation_context_async(
        self, thought: Thought, validation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Improve text with validation context awareness.

        Args:
            thought: The Thought container with the text to improve and critique.
            validation_context: Optional validation context for constraint awareness.

        Returns:
            The improved text that prioritizes validation constraints.

        Raises:
            ImproverError: If the improvement fails.
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
                        critique = feedback.feedback
                        break

            # If no critique available, generate one using async method
            if not critique:
                logger.debug("No critique found in thought, generating new critique")
                critique_result = await self._perform_critique_async(thought)
                critique = critique_result["message"]

            # Format principles for the prompt
            principles_text = "\n".join(
                f"{i + 1}. {principle}" for i, principle in enumerate(self.principles)
            )

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Parse critique to extract suggestions for filtering
            _, suggestions = self._parse_critique(critique)

            # Create improvement prompt with validation awareness
            if validation_context:
                # Use enhanced prompt with validation awareness
                improve_prompt = self._create_enhanced_improvement_prompt(
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                    validation_context=validation_context,
                    critic_suggestions=suggestions,
                )
            else:
                # Use original prompt template
                improve_prompt = self.improve_prompt_template.format(
                    principles=principles_text,
                    prompt=thought.prompt,
                    text=thought.text,
                    critique=critique,
                    context=context,
                )

            # Store the actual prompt for logging/debugging
            self.last_improvement_prompt = improve_prompt

            # Generate improved text (async only)
            improved_text = await self.model._generate_async(
                prompt=improve_prompt,
                system_prompt="You are an expert editor improving text to align with constitutional principles.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"ConstitutionalCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _parse_critique(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse critique text to extract issues and suggestions.

        Args:
            critique: The critique text to parse.

        Returns:
            A tuple of (issues, suggestions) lists.
        """
        issues = []
        suggestions = []

        # Simple parsing logic
        in_issues = False
        in_suggestions = False

        for line in critique.split("\n"):
            line = line.strip()
            if line.lower().startswith("issues:"):
                in_issues = True
                in_suggestions = False
                continue
            elif line.lower().startswith("suggestions:"):
                in_issues = False
                in_suggestions = True
                continue
            elif line.lower().startswith("overall assessment:"):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        # If no structured format found, extract from general content
        if not issues and not suggestions:
            critique_lower = critique.lower()
            if any(
                word in critique_lower for word in ["violation", "violates", "fails to", "does not"]
            ):
                issues.append("Constitutional principle violations identified")
            if any(word in critique_lower for word in ["improve", "suggest", "should", "consider"]):
                suggestions.append("See critique for constitutional improvement suggestions")

        return issues, suggestions

    def _extract_violations(self, critique: str) -> List[Dict[str, Any]]:
        """Extract principle violations from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A list of violation dictionaries.
        """
        violations = []
        lines = critique.split("\n")

        for line in lines:
            line_clean = line.strip().lower()
            if not line_clean:
                continue

            # Look for lines that clearly indicate violations
            if any(
                phrase in line_clean
                for phrase in [
                    "violation:",
                    "violates principle",
                    "does not adhere to",
                    "fails to meet",
                    "principle violated",
                ]
            ):
                # Skip obvious non-violations
                if not any(
                    phrase in line_clean
                    for phrase in ["no violation", "violation: none", "no violations found"]
                ):
                    violations.append(
                        {
                            "type": "principle_violation",
                            "description": line.strip(),
                            "severity": "medium",
                        }
                    )

        return violations
