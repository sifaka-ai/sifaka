"""Criticism execution for PydanticAI chains.

This module handles the criticism phase, running critics concurrently
and collecting feedback.
"""

import asyncio
from typing import Any, Dict, List

from sifaka.core.interfaces import Critic
from sifaka.core.thought import CriticFeedback, Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class CriticismExecutor:
    """Handles criticism execution for thoughts."""

    def __init__(self, critics: List[Critic]):
        """Initialize the criticism executor.

        Args:
            critics: List of critics to apply.
        """
        self.critics = critics

    async def execute(self, thought: Thought) -> Thought:
        """Execute criticism using configured critics.

        Args:
            thought: The thought to critique.

        Returns:
            Updated thought with critic feedback.
        """
        if not self.critics:
            logger.debug("No critics configured, skipping criticism")
            return thought

        logger.debug(f"Running async criticism with {len(self.critics)} critics")

        # Run all critics concurrently
        criticism_tasks = [self._critique_with_critic(critic, thought) for critic in self.critics]

        # Wait for all criticisms to complete
        criticism_results = await asyncio.gather(*criticism_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(criticism_results):
            critic = self.critics[i]
            critic_name = critic.__class__.__name__

            if isinstance(result, Exception):
                logger.error(f"Criticism error for {critic_name}: {result}")
                # Create error feedback
                error_feedback = CriticFeedback(
                    critic_name=critic_name,
                    feedback="Please try again or check the critic configuration",
                    confidence=0.0,
                    issues=[str(result)],
                    suggestions=["Please try again or check the critic configuration"],
                    needs_improvement=False,
                )
                thought = thought.add_critic_feedback(error_feedback)
            elif isinstance(result, dict):
                # Convert to CriticFeedback object and add to thought
                feedback = CriticFeedback(
                    critic_name=critic_name,
                    feedback=result.get("feedback", "") or result.get("message", ""),
                    confidence=result.get("confidence", 0.0),
                    violations=result.get("issues", []),
                    suggestions=result.get("suggestions", []),
                    needs_improvement=result.get("needs_improvement", False),
                    metadata=result.get("metadata", {}),  # Include metadata from critic
                )
                thought = thought.add_critic_feedback(feedback)
                logger.debug(f"Added async feedback from {critic_name}")

        return thought

    async def _critique_with_critic(self, critic: Critic, thought: Thought) -> Dict[str, Any]:
        """Run a single critic asynchronously with error handling.

        Args:
            critic: The critic to run.
            thought: The thought to critique.

        Returns:
            The criticism result as a dictionary.
        """
        try:
            # All critics must now be async-only
            return await critic.critique_async(thought)
        except Exception as e:
            logger.error(f"Criticism failed for {critic.__class__.__name__}: {e}")
            # Return error feedback dict
            return {"error": str(e), "confidence": 0.0, "issues": [str(e)], "suggestions": []}
