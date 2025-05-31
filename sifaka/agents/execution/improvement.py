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

            # Create improvement prompt based on feedback
            improvement_prompt = self.prompt_builder.create_improvement_prompt(current_thought)
            logger.debug(f"Improvement prompt: {improvement_prompt[:200]}...")

            # Generate improved text using PydanticAI
            try:
                logger.debug(
                    f"Running improvement iteration {iteration + 1} with prompt length: {len(improvement_prompt)}"
                )

                current_thought = await self.generation_executor.execute_improvement(
                    current_thought, improvement_prompt, **kwargs
                )

                logger.debug(
                    f"Improvement iteration {iteration + 1} completed, generated {len(current_thought.text or '')} characters"
                )

                # Re-validate
                current_thought = await self.validation_executor.execute(current_thought)

                # Save intermediate iteration if storage manager available
                if storage_manager:
                    await storage_manager.save_intermediate_thought(current_thought)

                # Check if validation now passes
                if self.validation_executor.validation_passed(current_thought):
                    logger.debug(f"Validation passed after iteration {iteration + 1}")
                    break

            except Exception as e:
                logger.error(f"Improvement iteration {iteration + 1} failed: {e}")
                break

        return current_thought
