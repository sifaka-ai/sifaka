"""SifakaEngine: Main orchestration engine for Sifaka.

This module implements the core engine that orchestrates the entire Sifaka
workflow using PydanticAI graphs. It provides the main API for thought
processing and conversation management.

Key features:
- Graph-based workflow orchestration
- Thought processing with complete audit trails
- Conversation continuity and management
- State persistence integration
- Error handling and recovery
"""

import uuid
from typing import List, Optional

from pydantic_graph import Graph
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.in_mem import FullStatePersistence

from sifaka.core.thought import SifakaThought
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.graph.nodes_unified import CritiqueNode, GenerateNode, ValidateNode
from sifaka.utils.errors import GraphExecutionError
from sifaka.utils.logging import get_logger
from sifaka.utils.validation import validate_prompt, validate_max_iterations

logger = get_logger(__name__)


class SifakaEngine:
    """Main orchestration engine for Sifaka.

    This engine provides the primary API for processing thoughts using
    the PydanticAI graph-based workflow. It handles:

    - Single thought processing
    - Conversation continuity
    - Batch processing
    - State persistence
    - Error recovery

    Example:
        ```python
        # Create engine with default configuration
        engine = SifakaEngine()

        # Process a single thought
        thought = await engine.think("Explain renewable energy")
        print(thought.final_text)

        # Continue conversation
        follow_up = await engine.continue_thought(
            thought,
            "Focus on solar panels"
        )
        ```
    """

    def __init__(
        self,
        dependencies: Optional[SifakaDependencies] = None,
        persistence: Optional[BaseStatePersistence] = None,
    ):
        """Initialize the Sifaka engine.

        Args:
            dependencies: Dependency container with agents, validators, etc.
                         If None, creates default configuration.
            persistence: State persistence implementation.
                        If None, uses Sifaka's memory persistence.
        """
        self.deps = dependencies or SifakaDependencies.create_default()

        # Use PydanticAI's FullStatePersistence by default
        if persistence is None:
            self.persistence = FullStatePersistence()
        else:
            self.persistence = persistence

        # Create the PydanticAI graph
        self.graph = Graph(
            nodes=[GenerateNode, ValidateNode, CritiqueNode],
            state_type=SifakaThought,
            run_end_type=SifakaThought,
            name="SifakaWorkflow",
        )

        logger.info(
            "SifakaEngine initialized",
            extra={
                "dependencies_type": type(self.deps).__name__,
                "persistence_type": type(self.persistence).__name__,
                "graph_nodes": [
                    node.__name__ for node in [GenerateNode, ValidateNode, CritiqueNode]
                ],
            },
        )

    async def think(self, prompt: str, max_iterations: int = 3) -> SifakaThought:
        """Process a single thought through the Sifaka workflow.

        This is the main method for processing thoughts. It creates a new
        thought and runs it through the complete workflow of generation,
        validation, and critique until completion.

        Args:
            prompt: The input prompt for generation
            max_iterations: Maximum number of improvement iterations

        Returns:
            Completed SifakaThought with full audit trail

        Raises:
            ValidationError: If input parameters are invalid
            GraphExecutionError: If the workflow encounters an unrecoverable error
        """
        # Validate inputs
        prompt = validate_prompt(prompt)
        max_iterations = validate_max_iterations(max_iterations)

        logger.info(
            "Starting thought processing",
            extra={
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "max_iterations": max_iterations,
            },
        )

        try:
            # Create initial thought
            thought = SifakaThought(prompt=prompt, max_iterations=max_iterations)

            logger.log_thought_event(
                "thought_created",
                thought.id,
                extra={
                    "prompt_length": len(prompt),
                    "max_iterations": max_iterations,
                },
            )

            # Run the graph starting with generation
            with logger.performance_timer("graph_execution", thought_id=thought.id):
                result = await self.graph.run(
                    GenerateNode(), state=thought, deps=self.deps, persistence=self.persistence
                )

            final_thought = result.output

            logger.log_thought_event(
                "thought_completed",
                final_thought.id,
                iteration=final_thought.iteration,
                extra={
                    "final_iteration": final_thought.iteration,
                    "techniques_applied": final_thought.techniques_applied,
                    "text_length": (
                        len(final_thought.current_text) if final_thought.current_text else 0
                    ),
                    "is_finalized": final_thought.final_text is not None,
                },
            )

            return final_thought
        except Exception as e:
            logger.error(
                "Failed to process thought",
                extra={
                    "prompt_preview": prompt[:100],
                    "max_iterations": max_iterations,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise GraphExecutionError(
                f"Failed to process thought: {str(e)}",
                execution_stage="graph_execution",
                context={"prompt": prompt[:100], "max_iterations": max_iterations},
            ) from e

    async def continue_thought(
        self, parent_thought: SifakaThought, new_prompt: str, max_iterations: int = 3
    ) -> SifakaThought:
        """Continue a conversation with a new thought connected to a parent.

        This method creates a new thought that is connected to the parent
        thought for conversation continuity. The new thought will have
        access to the conversation history.

        Args:
            parent_thought: The parent thought to continue from
            new_prompt: The new prompt for the follow-up thought
            max_iterations: Maximum number of improvement iterations

        Returns:
            New SifakaThought connected to the parent

        Raises:
            ValidationError: If input parameters are invalid
            GraphExecutionError: If the workflow encounters an unrecoverable error
        """
        # Validate inputs
        if not isinstance(parent_thought, SifakaThought):
            raise GraphExecutionError(
                f"parent_thought must be a SifakaThought, got {type(parent_thought).__name__}",
                execution_stage="input_validation",
                context={"parent_thought_type": type(parent_thought).__name__},
                suggestions=[
                    "Ensure parent_thought is a SifakaThought instance",
                    "Use engine.think() to create the initial thought first",
                ],
            )

        new_prompt = validate_prompt(new_prompt)
        max_iterations = validate_max_iterations(max_iterations)

        try:
            # Create new thought connected to parent
            new_thought = SifakaThought(prompt=new_prompt, max_iterations=max_iterations)
            new_thought.connect_to(parent_thought)

            # Run the graph for the new thought
            result = await self.graph.run(
                GenerateNode(), state=new_thought, deps=self.deps, persistence=self.persistence
            )

            return result.output
        except Exception as e:
            raise GraphExecutionError(
                f"Failed to continue thought: {str(e)}",
                execution_stage="graph_execution",
                context={
                    "new_prompt": new_prompt[:100],
                    "max_iterations": max_iterations,
                    "parent_id": parent_thought.id,
                },
            ) from e

    async def batch_think(self, prompts: List[str], max_iterations: int = 3) -> List[SifakaThought]:
        """Process multiple prompts in parallel.

        This method allows for efficient parallel processing of multiple
        independent thoughts. Each thought is processed through the complete
        workflow independently.

        Args:
            prompts: List of prompts to process
            max_iterations: Maximum iterations for each thought

        Returns:
            List of completed thoughts in the same order as input prompts

        Raises:
            ValidationError: If input parameters are invalid
            GraphExecutionError: If any workflow encounters an unrecoverable error
        """
        import asyncio

        # Validate inputs
        if not isinstance(prompts, list):
            raise GraphExecutionError(
                f"prompts must be a list, got {type(prompts).__name__}",
                execution_stage="input_validation",
                context={"prompts_type": type(prompts).__name__},
                suggestions=[
                    "Ensure prompts is a list of strings",
                    "Use [prompt] for single prompt processing",
                ],
            )

        if not prompts:
            raise GraphExecutionError(
                "prompts list cannot be empty",
                execution_stage="input_validation",
                suggestions=[
                    "Provide at least one prompt to process",
                    "Use engine.think() for single prompt processing",
                ],
            )

        if len(prompts) > 50:
            raise GraphExecutionError(
                f"Too many prompts: {len(prompts)} (maximum: 50)",
                execution_stage="input_validation",
                context={"prompt_count": len(prompts)},
                suggestions=[
                    "Process prompts in smaller batches",
                    "Consider if all prompts are really needed",
                ],
            )

        max_iterations = validate_max_iterations(max_iterations)

        # Create tasks for parallel processing
        tasks = [self.think(prompt, max_iterations) for prompt in prompts]

        # Wait for all thoughts to complete
        return await asyncio.gather(*tasks)

    def create_conversation(self, initial_prompt: str = "") -> str:
        """Create a new conversation ID for tracking related thoughts.

        This method generates a unique conversation ID that can be used
        to group related thoughts together. The first thought in a
        conversation will have this ID as its conversation_id.

        Args:
            initial_prompt: The initial prompt (for logging/debugging, optional)

        Returns:
            Unique conversation ID string
        """
        conversation_id = str(uuid.uuid4())
        if initial_prompt:
            logger.debug(
                f"Created conversation {conversation_id} for prompt: {initial_prompt[:50]}..."
            )
        return conversation_id

    async def get_conversation_thoughts(self, conversation_id: str) -> List[SifakaThought]:
        """Retrieve all thoughts in a conversation.

        This method uses the Sifaka persistence backend to retrieve all
        thoughts that belong to a specific conversation.

        Args:
            conversation_id: The conversation ID to retrieve

        Returns:
            List of thoughts in the conversation, ordered by creation time
        """
        try:
            # Check if persistence supports Sifaka-specific operations
            if hasattr(self.persistence, "list_thoughts"):
                return await self.persistence.list_thoughts(conversation_id=conversation_id)
            else:
                # Fallback for non-Sifaka persistence backends
                logger.warning(
                    f"Persistence backend {type(self.persistence).__name__} "
                    "does not support conversation queries"
                )
                return []
        except Exception as e:
            logger.error(f"Failed to retrieve conversation thoughts: {e}")
            return []

    async def get_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a specific thought by ID.

        Args:
            thought_id: The thought ID to retrieve

        Returns:
            The thought if found, None otherwise
        """
        try:
            # Check if persistence supports Sifaka-specific operations
            if hasattr(self.persistence, "retrieve_thought"):
                return await self.persistence.retrieve_thought(thought_id)
            else:
                # Fallback for non-Sifaka persistence backends
                logger.warning(
                    f"Persistence backend {type(self.persistence).__name__} "
                    "does not support thought retrieval"
                )
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve thought {thought_id}: {e}")
            return None

    def get_graph_diagram(self, format: str = "mermaid") -> str:
        """Get a visual representation of the workflow graph.

        Args:
            format: Diagram format ('mermaid' is currently supported)

        Returns:
            String representation of the graph diagram
        """
        if format == "mermaid":
            return self.graph.mermaid_code(start_node=GenerateNode)
        else:
            raise ValueError(f"Unsupported diagram format: {format}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.deps.__enter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.deps.__exit__(exc_type, exc_val, exc_tb)
