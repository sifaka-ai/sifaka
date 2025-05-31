"""Chain execution orchestrator for PydanticAI chains.

This module handles the main execution workflow, delegating to specialized
components for each phase of the chain execution.
"""

import uuid
from typing import List, Optional

from pydantic_ai import Agent

from sifaka.agents.execution.criticism import CriticismExecutor
from sifaka.agents.execution.generation import GenerationExecutor
from sifaka.agents.execution.improvement import ImprovementExecutor
from sifaka.agents.execution.retrieval import RetrievalExecutor
from sifaka.agents.execution.validation import ValidationExecutor
from sifaka.agents.storage.thought_storage import ThoughtStorage
from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import Thought
from sifaka.storage.protocol import Storage
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class ChainExecutor:
    """Orchestrates the execution of a PydanticAI chain workflow.

    This class coordinates the different phases of chain execution:
    1. Pre-generation retrieval
    2. Agent generation
    3. Validation
    4. Improvement loop (if needed)

    Each phase is handled by a specialized executor component.
    """

    def __init__(
        self,
        agent: "Agent",
        storage: Storage,
        validators: List[Validator],
        critics: List[Critic],
        model_retrievers: List,
        critic_retrievers: List,
        max_improvement_iterations: int = 2,
        always_apply_critics: bool = False,
        chain_id: Optional[str] = None,
    ):
        """Initialize the chain executor.

        Args:
            agent: The PydanticAI agent to use for generation.
            storage: Storage backend for thoughts.
            validators: List of validators to apply.
            critics: List of critics to apply.
            model_retrievers: List of retrievers for pre-generation context.
            critic_retrievers: List of retrievers for critic-specific context.
            max_improvement_iterations: Maximum number of improvement iterations.
            always_apply_critics: Whether to always apply critics.
            chain_id: Optional chain identifier.
        """
        self.agent = agent
        self.chain_id = chain_id or str(uuid.uuid4())
        self.max_improvement_iterations = max_improvement_iterations
        self.always_apply_critics = always_apply_critics

        # Initialize specialized executors
        self.storage_manager = ThoughtStorage(storage)
        self.retrieval_executor = RetrievalExecutor(model_retrievers, critic_retrievers)
        self.generation_executor = GenerationExecutor(agent)
        self.validation_executor = ValidationExecutor(validators)
        self.criticism_executor = CriticismExecutor(critics)
        self.improvement_executor = ImprovementExecutor(
            agent, self.criticism_executor, self.validation_executor
        )

    async def execute(self, prompt: str, **kwargs) -> Thought:
        """Execute the complete chain workflow.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        logger.info(f"Starting chain execution for prompt: {prompt[:50]}...")

        with time_operation("chain_execution"):
            # Create initial thought
            thought = Thought(prompt=prompt, chain_id=self.chain_id, iteration=0)

            # Phase 1: Pre-generation retrieval
            thought = await self.retrieval_executor.execute_model_retrieval(thought)

            # Phase 2: Initial generation
            thought = await self.generation_executor.execute(thought, **kwargs)

            # Save initial iteration
            await self.storage_manager.save_intermediate_thought(thought)

            # Phase 3: Validation
            thought = await self.validation_executor.execute(thought)

            # Phase 4: Improvement loop if needed
            should_improve = self._should_improve(thought)
            if should_improve:
                thought = await self.improvement_executor.execute(
                    thought,
                    self.max_improvement_iterations,
                    self.retrieval_executor,
                    self.storage_manager,
                    **kwargs,
                )

            # Save final result
            await self.storage_manager.save_final_thought(thought)

            logger.info(f"Chain execution completed. Final iteration: {thought.iteration}")
            return thought

    def _should_improve(self, thought: Thought) -> bool:
        """Determine if improvement iterations should be run.

        Args:
            thought: The thought to check.

        Returns:
            True if improvement should be attempted.
        """
        validation_passed = self.validation_executor.validation_passed(thought)

        return (not validation_passed and self.max_improvement_iterations > 0) or (
            self.always_apply_critics and self.max_improvement_iterations > 0
        )
