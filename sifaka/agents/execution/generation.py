"""Generation execution for PydanticAI chains.

This module handles the agent generation phase, including output extraction
and metadata collection.
"""

from pydantic_ai import Agent

from sifaka.agents.extractors.agent_data import AgentDataExtractor
from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class GenerationExecutor:
    """Handles PydanticAI agent generation execution."""

    def __init__(self, agent: Agent):
        """Initialize the generation executor.

        Args:
            agent: The PydanticAI agent to use for generation.
        """
        self.agent = agent
        self.data_extractor = AgentDataExtractor(agent)

    async def execute(self, thought: Thought, **kwargs) -> Thought:
        """Execute agent generation.

        Args:
            thought: The current thought state.
            **kwargs: Additional arguments for the agent.

        Returns:
            Updated thought with generated text and metadata.
        """
        logger.debug("Executing PydanticAI agent generation")

        with time_operation("agent_generation"):
            try:
                # Run the PydanticAI agent asynchronously
                result = await self.agent.run(thought.prompt, **kwargs)

                # Extract output and metadata
                output, updated_thought = self.data_extractor.extract_output(result, thought)
                thought = (updated_thought or thought).set_text(output)

                # Extract comprehensive metadata
                metadata = self.data_extractor.extract_metadata(result, thought.prompt)

                thought = thought.model_copy(
                    update={
                        # Don't increment iteration for initial generation - stay on iteration 0
                        "model_name": metadata["model_name"],
                        "model_prompt": metadata["model_prompt"],
                        "system_prompt": metadata["system_prompt"],
                    }
                )

                logger.debug(f"Agent generated {len(output)} characters")
                return thought

            except Exception as e:
                logger.error(f"Agent generation failed: {e}")
                raise ChainError(f"PydanticAI agent generation failed: {e}")

    async def execute_improvement(
        self, thought: Thought, improvement_prompt: str, **kwargs
    ) -> Thought:
        """Execute agent generation for improvement iteration.

        Args:
            thought: The current thought state.
            improvement_prompt: The improvement prompt to use.
            **kwargs: Additional arguments for the agent.

        Returns:
            Updated thought with improved text and metadata.
        """
        logger.debug("Executing improvement generation")

        with time_operation("improvement_generation"):
            try:
                result = await self.agent.run(improvement_prompt, **kwargs)
                improved_text, updated_thought = self.data_extractor.extract_output(result, thought)

                # Don't create new iteration here - that's handled by ImprovementExecutor
                current_thought = (updated_thought or thought).set_text(improved_text)

                # Extract and update metadata for improvement iteration
                metadata = self.data_extractor.extract_metadata(result, improvement_prompt)

                current_thought = current_thought.model_copy(
                    update={
                        "model_name": metadata["model_name"],
                        "model_prompt": metadata["model_prompt"],
                        "system_prompt": metadata["system_prompt"],
                    }
                )

                logger.debug(f"Improvement generated {len(improved_text)} characters")
                return current_thought

            except Exception as e:
                logger.error(f"Improvement generation failed: {e}")
                raise ChainError(f"Improvement generation failed: {e}")
