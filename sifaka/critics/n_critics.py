"""N-Critics critic for Sifaka.

This module implements the N-Critics approach for text improvement, which uses
an ensemble of specialized critics to provide comprehensive feedback and guide
the refinement process.

Based on "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics"

The NCriticsCritic leverages multiple specialized critics, each focusing on different
aspects of the text, to provide comprehensive feedback and guide the refinement process.
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


class NCriticsCritic(ContextAwareMixin):
    """Critic that uses an ensemble of specialized critics.

    This critic implements the N-Critics technique, which leverages an ensemble
    of specialized critics, each focusing on different aspects of the text,
    to provide comprehensive feedback and guide the refinement process.

    Attributes:
        model: The language model to use for critique and improvement.
        num_critics: Number of specialized critics to use.
        critic_roles: List of specialized critic roles/perspectives.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        num_critics: int = 3,
        critic_roles: Optional[List[str]] = None,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the N-Critics critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            num_critics: Number of specialized critics to use.
            critic_roles: List of specialized critic roles/perspectives.
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

        self.num_critics = num_critics

        # Set up critic roles
        self.critic_roles = (
            critic_roles
            or [
                "Content Expert: Focus on factual accuracy, completeness, and relevance of information",
                "Style Editor: Focus on writing style, tone, clarity, and readability",
                "Structure Analyst: Focus on organization, flow, coherence, and logical structure",
                "Audience Specialist: Focus on appropriateness for target audience and effectiveness",
                "Quality Assurance: Focus on overall quality, consistency, and adherence to requirements",
            ][:num_critics]
        )

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "You are a {role}.\n\n"
            "Please critique the following text from your specialized perspective.\n\n"
            "Original task: {prompt}\n\n"
            "Text to critique:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Provide a focused critique addressing:\n"
            "1. How well does the text perform in your area of expertise?\n"
            "2. What specific issues do you identify?\n"
            "3. What improvements would you recommend?\n"
            "4. Rate the text from 1-10 in your area of focus\n"
            "5. How well does the text use information from the retrieved context (if available)?\n\n"
            "Be specific and constructive in your feedback."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text based on feedback from multiple specialized critics.\n\n"
            "Original task: {prompt}\n\n"
            "Current text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critic Feedback:\n{aggregated_feedback}\n\n"
            "Please provide an improved version that:\n"
            "1. Addresses all the issues identified by the critics\n"
            "2. Incorporates the suggestions from each specialist\n"
            "3. Maintains the core message and purpose\n"
            "4. Balances the different perspectives and requirements\n"
            "5. Better incorporates relevant information from the context (if available)"
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text using ensemble of specialized critics.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with aggregated critique results from all critics.
        """
        start_time = time.time()

        with critic_context(
            critic_name="NCriticsCritic",
            operation="critique",
            message_prefix="Failed to critique text with N-Critics ensemble",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "message": "No text available for critique",
                    "issues": ["Text is empty or None"],
                    "suggestions": ["Provide text to critique"],
                    "critic_feedback": [],
                }

            # Generate critiques from each specialized critic
            critic_critiques = []
            for i, role in enumerate(self.critic_roles):
                logger.debug(
                    f"NCriticsCritic: Generating critique from critic {i+1}/{len(self.critic_roles)}"
                )

                critique = self._generate_critic_critique(thought, role)
                critic_critiques.append(critique)

            # Aggregate critiques
            aggregated_critique = self._aggregate_critiques(critic_critiques)

            # Determine if improvement is needed
            needs_improvement = any(
                critique.get("needs_improvement", True) for critique in critic_critiques
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(
                f"NCriticsCritic: Critique completed in {processing_time:.2f}ms "
                f"with {len(critic_critiques)} critics"
            )

            return {
                "needs_improvement": needs_improvement,
                "message": aggregated_critique["summary"],
                "critique": aggregated_critique["summary"],
                "critic_feedback": critic_critiques,
                "aggregated_score": aggregated_critique["average_score"],
                "num_critics": len(critic_critiques),
                "processing_time_ms": processing_time,
            }

    def improve(self, thought: Thought) -> str:
        """Improve text based on ensemble critic feedback.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text based on aggregated critic feedback.
        """
        start_time = time.time()

        with critic_context(
            critic_name="NCriticsCritic",
            operation="improve",
            message_prefix="Failed to improve text with N-Critics ensemble",
        ):
            # Check if text is available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="NCriticsCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            aggregated_feedback = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "NCriticsCritic":
                        critic_feedback = feedback.feedback.get("critic_feedback", [])
                        aggregated_feedback = self._format_feedback_for_improvement(critic_feedback)
                        break

            # Prepare context for improvement (using mixin)
            context = self._prepare_context(thought)

            # Create improvement prompt with context
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                aggregated_feedback=aggregated_feedback,
                context=context,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor incorporating feedback from multiple specialized critics.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"NCriticsCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _generate_critic_critique(self, thought: Thought, role: str) -> Dict[str, Any]:
        """Generate critique from a single specialized critic.

        Args:
            thought: The Thought container with the text to critique.
            role: The role/perspective of the specialized critic.

        Returns:
            A dictionary with the critic's feedback.
        """
        # Prepare context for this critic (using mixin)
        context = self._prepare_context(thought)

        # Create critique prompt for this specific critic with context
        critique_prompt = self.critique_prompt_template.format(
            role=role,
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique
        critique_response = self.model.generate(
            prompt=critique_prompt,
            system_prompt=f"You are a specialized critic with the role: {role}",
        )

        # Extract score and determine improvement need
        score = self._extract_score_from_critique(critique_response)
        needs_improvement = score < 7.0

        return {
            "role": role,
            "critique": critique_response,
            "score": score,
            "needs_improvement": needs_improvement,
        }

    def _aggregate_critiques(self, critic_critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate critiques from multiple critics.

        Args:
            critic_critiques: List of critique dictionaries from each critic.

        Returns:
            An aggregated critique summary.
        """
        if not critic_critiques:
            return {
                "summary": "No critiques available",
                "average_score": 0.0,
                "issues": [],
                "suggestions": [],
            }

        # Calculate average score
        scores = [critique.get("score", 5.0) for critique in critic_critiques]
        average_score = sum(scores) / len(scores)

        # Create summary
        summary_parts = []
        for i, critique in enumerate(critic_critiques):
            role = critique.get("role", f"Critic {i+1}")
            score = critique.get("score", 5.0)
            summary_parts.append(f"{role} (Score: {score}/10):\n{critique.get('critique', '')}")

        summary = "\n\n".join(summary_parts)

        return {
            "summary": summary,
            "average_score": average_score,
            "individual_scores": scores,
            "num_critics": len(critic_critiques),
        }

    def _extract_score_from_critique(self, critique: str) -> float:
        """Extract numerical score from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            A score between 0.0 and 10.0.
        """
        import re

        # Look for score patterns
        score_patterns = [
            r"(\d+(?:\.\d+)?)[/\s]*(?:out of\s*)?10",
            r"score[:\s]+(\d+(?:\.\d+)?)",
            r"rate[:\s]+(\d+(?:\.\d+)?)",
            r"rating[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, critique.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(10.0, max(0.0, score))
                except ValueError:
                    continue

        # Default score based on sentiment
        if any(word in critique.lower() for word in ["excellent", "outstanding", "perfect"]):
            return 8.5
        elif any(word in critique.lower() for word in ["good", "solid", "well"]):
            return 7.0
        elif any(word in critique.lower() for word in ["poor", "bad", "terrible"]):
            return 3.0
        else:
            return 5.5

    def _format_feedback_for_improvement(self, critic_feedback: List[Dict[str, Any]]) -> str:
        """Format critic feedback for the improvement prompt.

        Args:
            critic_feedback: List of feedback from individual critics.

        Returns:
            A formatted string with all critic feedback.
        """
        if not critic_feedback:
            return "No specific feedback available."

        feedback_parts = []
        for feedback in critic_feedback:
            role = feedback.get("role", "Critic")
            critique = feedback.get("critique", "")
            score = feedback.get("score", 5.0)

            feedback_parts.append(f"{role} (Score: {score}/10):\n{critique}")

        return "\n\n".join(feedback_parts)


def create_n_critics_critic(
    model: Optional[Model] = None,
    model_name: Optional[str] = None,
    num_critics: int = 3,
    critic_roles: Optional[List[str]] = None,
    **model_kwargs: Any,
) -> NCriticsCritic:
    """Create an N-Critics critic.

    Args:
        model: The language model to use for critique and improvement.
        model_name: The name of the model to use if model is not provided.
        num_critics: Number of specialized critics to use.
        critic_roles: List of specialized critic roles/perspectives.
        **model_kwargs: Additional keyword arguments for model creation.

    Returns:
        An NCriticsCritic instance.
    """
    return NCriticsCritic(
        model=model,
        model_name=model_name,
        num_critics=num_critics,
        critic_roles=critic_roles,
        **model_kwargs,
    )
