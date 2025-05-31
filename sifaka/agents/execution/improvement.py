"""Improvement execution for PydanticAI chains.

This module handles the improvement loop, coordinating criticism, prompt building,
and iterative generation.
"""

from pydantic_ai import Agent

from sifaka.agents.execution.criticism import CriticismExecutor
from sifaka.agents.execution.generation import GenerationExecutor
from sifaka.agents.execution.retrieval import RetrievalExecutor
from sifaka.agents.execution.validation import ValidationExecutor
from sifaka.agents.extractors.prompt_builder import ImprovementPromptBuilder
from sifaka.agents.storage.thought_storage import ThoughtStorage
from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ImprovementExecutor:
    """Handles improvement loop execution for thoughts."""

    def __init__(
        self,
        agent: Agent,
        criticism_executor: CriticismExecutor,
        validation_executor: ValidationExecutor,
    ):
        """Initialize the improvement executor.

        Args:
            agent: The PydanticAI agent for generation.
            criticism_executor: The criticism executor.
            validation_executor: The validation executor.
        """
        self.generation_executor = GenerationExecutor(agent)
        self.criticism_executor = criticism_executor
        self.validation_executor = validation_executor
        self.prompt_builder = ImprovementPromptBuilder()

    async def execute(
        self,
        thought: Thought,
        max_iterations: int,
        retrieval_executor: RetrievalExecutor = None,
        storage_manager: ThoughtStorage = None,
        **kwargs,
    ) -> Thought:
        """Execute improvement iterations using critics and agent feedback.

        Args:
            thought: The thought with validation results.
            max_iterations: Maximum number of improvement iterations.
            retrieval_executor: Optional retrieval executor for critic context.
            storage_manager: Optional storage manager for saving iterations.
            **kwargs: Additional arguments for the agent.

        Returns:
            Improved thought after iterations.
        """
        logger.debug(f"Starting improvement loop (max {max_iterations} iterations)")

        current_thought = thought

        for iteration in range(max_iterations):
            logger.debug(f"Improvement iteration {iteration + 1}")

            # Apply critic retrieval if available
            if retrieval_executor:
                current_thought = await retrieval_executor.execute_critic_retrieval(current_thought)

            # Apply critics to get feedback
            current_thought = await self.criticism_executor.execute(current_thought)

            # Try to use critic's improvement method if available, otherwise use prompt builder
            improved_text = None

            # Check if we have critics with improvement capabilities
            logger.debug(
                f"Checking critic improvement: feedback={bool(current_thought.critic_feedback)}, critics={len(self.criticism_executor.critics) if self.criticism_executor.critics else 0}"
            )

            if current_thought.critic_feedback and self.criticism_executor.critics:
                # Try to use the first critic's improvement method
                for critic in self.criticism_executor.critics:
                    logger.debug(
                        f"Checking critic {critic.__class__.__name__} for improve_async method"
                    )
                    if hasattr(critic, "improve_async"):
                        try:
                            logger.debug(f"Using {critic.__class__.__name__} improve_async method")
                            improved_text = await critic.improve_async(current_thought)
                            logger.debug(
                                f"Critic improvement completed: {len(improved_text)} characters"
                            )
                            break
                        except Exception as e:
                            logger.warning(
                                f"Critic {critic.__class__.__name__} improvement failed: {e}"
                            )
                            continue
                    else:
                        logger.debug(
                            f"Critic {critic.__class__.__name__} does not have improve_async method"
                        )
            else:
                logger.debug("No critic feedback or critics available for improvement")

            # Fallback to prompt builder if critic improvement failed or unavailable
            if improved_text is None:
                # Create improvement prompt based on feedback
                improvement_prompt = self.prompt_builder.create_improvement_prompt(current_thought)
                logger.debug(f"Using prompt builder fallback: {improvement_prompt[:200]}...")

                # Generate improved text using PydanticAI
                try:
                    logger.debug(
                        f"Running improvement iteration {iteration + 1} with prompt length: {len(improvement_prompt)}"
                    )

                    current_thought = await self.generation_executor.execute_improvement(
                        current_thought, improvement_prompt, **kwargs
                    )
                except Exception as e:
                    logger.error(f"Prompt builder improvement failed: {e}")
                    break
            else:
                # Use the critic-improved text
                current_thought.text = improved_text
                current_thought.iteration += 1

            # Re-validate
            current_thought = await self.validation_executor.execute(current_thought)

            # Save intermediate iteration if storage manager available
            if storage_manager:
                await storage_manager.save_intermediate_thought(current_thought)

            # Check if validation now passes
            if self.validation_executor.validation_passed(current_thought):
                logger.debug(f"Validation passed after iteration {iteration + 1}")
                break

        return current_thought
