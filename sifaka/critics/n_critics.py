"""N-Critics critic for Sifaka.

This module implements the N-Critics approach for text improvement, which uses
an ensemble of specialized critics to provide comprehensive feedback and guide
the refinement process.

Based on "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics":
https://arxiv.org/abs/2310.18679

The NCriticsCritic leverages multiple specialized critics, each focusing on different
aspects of the text, to provide comprehensive feedback.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.critics.base import BaseCritic
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class NCriticsCritic(BaseCritic):
    """Critic that uses an ensemble of specialized critics.

    This critic implements the N-Critics technique, which leverages an ensemble
    of specialized critics, each focusing on different aspects of the text,
    to provide comprehensive feedback and guide the refinement process.
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
        super().__init__(model=model, model_name=model_name, **model_kwargs)

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
            "Provide a structured critique with the following format:\n\n"
            "PERFORMANCE: [How well does the text perform in your area of expertise?]\n\n"
            "Issues:\n- [List specific problems you identify]\n\n"
            "Suggestions:\n- [List specific improvements you recommend]\n\n"
            "SCORE: [Rate the text from 1-10 in your area of focus]\n\n"
            "Be specific, constructive, and focus on your area of expertise."
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
            "5. Better incorporates relevant information from the context (if available)\n\n"
            "Improved text:"
        )

    async def _perform_critique_async(self, thought: Thought) -> Dict[str, Any]:
        """Perform the actual critique logic using N-Critics ensemble approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results (without processing_time_ms).
        """
        # Generate critiques from each specialized critic (async)
        critic_tasks = []
        for role in self.critic_roles:
            critic_tasks.append(self._generate_critic_critique_async(thought, role))

        # Wait for all critiques to complete
        critic_critiques = await asyncio.gather(*critic_tasks, return_exceptions=True)

        # Process results and handle exceptions
        valid_critiques = []
        for i, result in enumerate(critic_critiques):
            if isinstance(result, Exception):
                logger.error(f"NCriticsCritic: Error in critic {i+1}: {result}")
                # Create error critique
                error_critique = {
                    "role": self.critic_roles[i],
                    "score": 0.0,
                    "needs_improvement": True,
                    "issues": [f"Critic error: {str(result)}"],
                    "suggestions": ["Please try again or check the critic configuration."],
                    "critique": f"Error in critic: {str(result)}",
                }
                valid_critiques.append(error_critique)
            else:
                valid_critiques.append(result)

        # Aggregate feedback from all critics
        aggregated_feedback = self._aggregate_critiques(valid_critiques)

        # Extract issues and suggestions from all critics
        all_issues: List[str] = []
        all_suggestions: List[str] = []
        for critique in valid_critiques:
            if "issues" in critique and isinstance(critique["issues"], list):
                all_issues.extend(critique["issues"])
            if "suggestions" in critique and isinstance(critique["suggestions"], list):
                all_suggestions.extend(critique["suggestions"])

        # Determine if improvement is needed
        needs_improvement = aggregated_feedback.get("average_score", 5.0) < 7.0

        logger.debug(f"NCriticsCritic: Completed with {len(valid_critiques)} critics")

        return {
            "needs_improvement": needs_improvement,
            "message": aggregated_feedback["summary"],
            "issues": all_issues,
            "suggestions": all_suggestions,
            "confidence": 0.8,  # Default confidence for N-Critics
            "metadata": {
                "critic_feedback": valid_critiques,
                "aggregated_score": aggregated_feedback["average_score"],
                "num_critics": len(valid_critiques),
            },
        }

    def improve(self, thought: Thought) -> str:
        """Improve text based on ensemble critic feedback.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text based on aggregated critic feedback.

        Raises:
            ImproverError: If the improvement fails.
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
                        critic_feedback = feedback.metadata.get("critic_feedback", [])
                        aggregated_feedback = self._format_feedback_for_improvement(critic_feedback)
                        break

            # If no critique available, generate one
            if not aggregated_feedback:
                logger.debug("No critique found in thought, generating new critique")
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._perform_critique_async(thought))
                        critique_result = future.result()
                except RuntimeError:
                    critique_result = asyncio.run(self._perform_critique_async(thought))

                critic_feedback = critique_result["metadata"]["critic_feedback"]
                aggregated_feedback = self._format_feedback_for_improvement(critic_feedback)

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

    async def _generate_critic_critique_async(self, thought: Thought, role: str) -> Dict[str, Any]:
        """Generate critique from a single specialized critic asynchronously.

        Args:
            thought: The Thought container with the text to critique.
            role: The role/specialty of the critic.

        Returns:
            A dictionary with the critic's feedback.
        """
        # Prepare context from retrieved documents (using mixin)
        context = self._prepare_context(thought)

        # Create role-specific critique prompt
        critique_prompt = self.critique_prompt_template.format(
            role=role,
            prompt=thought.prompt,
            text=thought.text,
            context=context,
        )

        # Generate critique (async)
        critique_response = await self.model._generate_async(
            prompt=critique_prompt,
            system_message=f"You are a specialized critic with the role: {role}",
        )

        # Extract score and determine improvement need
        score = self._extract_score_from_critique(critique_response)
        needs_improvement = score < 7.0

        # Parse structured feedback from the critique response
        issues, suggestions = self._parse_structured_feedback(critique_response)

        return {
            "role": role,
            "score": score,
            "needs_improvement": needs_improvement,
            "issues": issues,
            "suggestions": suggestions,
            "critique": critique_response,
        }

    def _aggregate_critiques(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate feedback from multiple critics.

        Args:
            critiques: List of critique dictionaries from individual critics.

        Returns:
            A dictionary with aggregated feedback.
        """
        if not critiques:
            return {
                "summary": "No critiques available",
                "average_score": 0.0,
                "num_critics": 0,
            }

        # Calculate average score
        scores = [c.get("score", 0.0) for c in critiques]
        average_score = sum(scores) / len(scores) if scores else 0.0

        # Create summary
        summary_parts = []
        for i, critique in enumerate(critiques, 1):
            role = critique.get("role", f"Critic {i}")
            score = critique.get("score", 0.0)
            summary_parts.append(f"{role}: Score {score}/10")

        summary = (
            f"Ensemble feedback from {len(critiques)} critics (Average: {average_score:.1f}/10):\n"
        )
        summary += "\n".join(summary_parts)

        return {
            "summary": summary,
            "average_score": average_score,
            "num_critics": len(critiques),
        }

    def _extract_score_from_critique(self, critique: str) -> float:
        """Extract numerical score from critique text.

        Args:
            critique: The critique text to analyze.

        Returns:
            The extracted score (0.0-10.0).
        """
        import re

        # Look for "SCORE: X" pattern
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", critique, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(10.0, score))  # Clamp to [0, 10]
            except ValueError:
                pass

        # Look for "X/10" pattern
        score_match = re.search(r"(\d+(?:\.\d+)?)/10", critique)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(10.0, score))
            except ValueError:
                pass

        # Default score based on content analysis
        critique_lower = critique.lower()
        if any(word in critique_lower for word in ["excellent", "great", "perfect"]):
            return 8.0
        elif any(word in critique_lower for word in ["good", "well", "solid"]):
            return 7.0
        elif any(word in critique_lower for word in ["poor", "bad", "terrible"]):
            return 3.0
        else:
            return 5.0  # Default neutral score

    def _parse_structured_feedback(self, critique: str) -> tuple[List[str], List[str]]:
        """Parse structured feedback from critique text.

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
            elif line.lower().startswith(("score:", "performance:", "context")):
                in_issues = False
                in_suggestions = False
                continue
            elif not line or line.startswith("#"):
                continue

            if in_issues and line.startswith("-"):
                issues.append(line[1:].strip())
            elif in_suggestions and line.startswith("-"):
                suggestions.append(line[1:].strip())

        return issues, suggestions

    def _format_feedback_for_improvement(self, critic_feedback: List[Dict[str, Any]]) -> str:
        """Format critic feedback for improvement prompt.

        Args:
            critic_feedback: List of feedback from individual critics.

        Returns:
            A formatted string with aggregated feedback.
        """
        if not critic_feedback:
            return "No specific feedback available."

        feedback_parts = []
        for i, feedback in enumerate(critic_feedback, 1):
            role = feedback.get("role", f"Critic {i}")
            score = feedback.get("score", 0.0)
            issues = feedback.get("issues", [])
            suggestions = feedback.get("suggestions", [])

            part = f"{role} (Score: {score}/10):\n"
            if issues:
                part += "Issues:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n"
            if suggestions:
                part += (
                    "Suggestions:\n"
                    + "\n".join(f"- {suggestion}" for suggestion in suggestions)
                    + "\n"
                )

            feedback_parts.append(part)

        return "\n".join(feedback_parts)
