"""PydanticAI Chain implementation for Sifaka.

This module provides the main PydanticAIChain class that orchestrates
PydanticAI agents with Sifaka's validation and criticism framework.

This is a simplified orchestrator that delegates to specialized components
for better maintainability and proper async patterns.
"""

from typing import List, Optional

from sifaka.agents.core.async_utils import ensure_async_compatibility
from sifaka.agents.core.executor import ChainExecutor
from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import Thought
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.protocol import Storage
from sifaka.utils.error_handling import ChainError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Import PydanticAI with availability check
try:
    from pydantic_ai import Agent

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None


class PydanticAIChain:
    """Hybrid chain that orchestrates PydanticAI agents with Sifaka components.

    This class provides a composition-based approach to integrating PydanticAI
    agents with Sifaka's validation and criticism framework. It delegates to
    specialized components for better maintainability and proper async patterns.
    """

    def __init__(
        self,
        agent: "Agent",
        storage: Optional[Storage] = None,
        validators: Optional[List[Validator]] = None,
        critics: Optional[List[Critic]] = None,
        model_retrievers: Optional[List] = None,
        critic_retrievers: Optional[List] = None,
        max_improvement_iterations: int = 2,
        always_apply_critics: bool = False,
        enable_critic_tools: bool = True,
        enable_validator_tools: bool = True,
        chain_id: Optional[str] = None,
    ):
        """Initialize the PydanticAI chain.

        Args:
            agent: The PydanticAI agent to use for generation.
            storage: Storage backend for thoughts. Defaults to MemoryStorage.
            validators: List of Sifaka validators to apply.
            critics: List of Sifaka critics to apply.
            model_retrievers: List of retrievers for pre-generation context.
            critic_retrievers: List of retrievers for critic-specific context.
            max_improvement_iterations: Maximum number of improvement iterations.
            always_apply_critics: Whether to always apply critics, even if validation passes.
            enable_critic_tools: Whether to register critics as PydanticAI tools.
            enable_validator_tools: Whether to register validators as PydanticAI tools.
            chain_id: Optional chain identifier. Generated if not provided.
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise ChainError("PydanticAI is not available")

        # Initialize the chain executor with all components
        self.executor = ChainExecutor(
            agent=agent,
            storage=storage if storage is not None else MemoryStorage(),
            validators=validators or [],
            critics=critics or [],
            model_retrievers=model_retrievers or [],
            critic_retrievers=critic_retrievers or [],
            max_improvement_iterations=max_improvement_iterations,
            always_apply_critics=always_apply_critics,
            chain_id=chain_id,
        )

        # Setup agent tools if enabled (placeholder for Phase 2)
        if enable_critic_tools:
            self._setup_critic_tools()
        if enable_validator_tools:
            self._setup_validator_tools()

    @ensure_async_compatibility
    async def run(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain.

        This method provides both sync and async interfaces through proper
        async compatibility handling, replacing the problematic sync/async pattern.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        return await self.run_async(prompt, **kwargs)

    async def run_async(self, prompt: str, **kwargs) -> Thought:
        """Execute the hybrid chain asynchronously.

        This method delegates to the ChainExecutor for the actual workflow execution.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            A Thought object containing the final result.
        """
        return await self.executor.execute(prompt, **kwargs)

    def _setup_critic_tools(self):
        """Setup critics as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Critic tools setup (placeholder)")

    def _setup_validator_tools(self):
        """Setup validators as PydanticAI tools (placeholder for Phase 2)."""
        # This will be implemented in Phase 2
        logger.debug("Validator tools setup (placeholder)")

    # Backward compatibility properties
    @property
    def chain_id(self) -> str:
        """Get the chain ID."""
        return self.executor.chain_id

    @property
    def storage(self):
        """Get the storage backend."""
        return self.executor.storage_manager.storage

    @property
    def validators(self):
        """Get the validators."""
        return self.executor.validation_executor.validators

    @property
    def critics(self):
        """Get the critics."""
        return self.executor.criticism_executor.critics

    @property
    def model_retrievers(self):
        """Get the model retrievers."""
        return self.executor.retrieval_executor.model_retrievers

    @property
    def critic_retrievers(self):
        """Get the critic retrievers."""
        return self.executor.retrieval_executor.critic_retrievers

    @property
    def max_improvement_iterations(self) -> int:
        """Get the max improvement iterations."""
        return self.executor.max_improvement_iterations

    @property
    def always_apply_critics(self) -> bool:
        """Get the always apply critics setting."""
        return self.executor.always_apply_critics
