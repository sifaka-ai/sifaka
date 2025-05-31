"""Retrieval execution for PydanticAI chains.

This module handles the retrieval phases, both pre-generation and critic-specific
retrieval operations.
"""

import asyncio
from typing import List

from sifaka.core.thought import Thought
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class RetrievalExecutor:
    """Handles retrieval execution for thoughts."""

    def __init__(self, model_retrievers: List, critic_retrievers: List):
        """Initialize the retrieval executor.

        Args:
            model_retrievers: List of retrievers for pre-generation context.
            critic_retrievers: List of retrievers for critic-specific context.
        """
        self.model_retrievers = model_retrievers
        self.critic_retrievers = critic_retrievers

    async def execute_model_retrieval(self, thought: Thought) -> Thought:
        """Execute pre-generation retrieval using model retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with retrieved context.
        """
        if not self.model_retrievers:
            logger.debug("No model retrievers configured, skipping model retrieval")
            return thought

        logger.debug(f"Running async model retrieval with {len(self.model_retrievers)} retrievers")

        with time_operation("model_retrieval"):
            # Run all retrievers concurrently
            retrieval_tasks = [
                self._retrieve_with_retriever(retriever, thought, is_pre_generation=True)
                for retriever in self.model_retrievers
            ]

            # Wait for all retrievals to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results sequentially to maintain thought state consistency
            for i, result in enumerate(retrieval_results):
                retriever = self.model_retrievers[i]
                retriever_name = retriever.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Model retrieval error for {retriever_name}: {result}")
                    # Continue with other retrievers
                else:
                    # Update thought with retrieved context
                    thought = result
                    logger.debug(f"Applied async model retriever: {retriever_name}")

            return thought

    async def execute_critic_retrieval(self, thought: Thought) -> Thought:
        """Execute retrieval for critics using critic retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic-specific context.
        """
        if not self.critic_retrievers:
            logger.debug("No critic retrievers configured, skipping critic retrieval")
            return thought

        logger.debug(
            f"Running async critic retrieval with {len(self.critic_retrievers)} retrievers"
        )

        with time_operation("critic_retrieval"):
            # Run all retrievers concurrently
            retrieval_tasks = [
                self._retrieve_with_retriever(retriever, thought, is_pre_generation=False)
                for retriever in self.critic_retrievers
            ]

            # Wait for all retrievals to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Process results sequentially to maintain thought state consistency
            for i, result in enumerate(retrieval_results):
                retriever = self.critic_retrievers[i]
                retriever_name = retriever.__class__.__name__

                if isinstance(result, Exception):
                    logger.error(f"Critic retrieval error for {retriever_name}: {result}")
                    # Continue with other retrievers
                else:
                    # Update thought with retrieved context
                    thought = result
                    logger.debug(f"Applied async critic retriever: {retriever_name}")

            return thought

    async def _retrieve_with_retriever(
        self, retriever, thought: Thought, is_pre_generation: bool
    ) -> Thought:
        """Run a single retriever asynchronously with error handling.

        Args:
            retriever: The retriever to run.
            thought: The thought to retrieve context for.
            is_pre_generation: Whether this is pre-generation retrieval.

        Returns:
            Updated thought with retrieved context.
        """
        try:
            from sifaka.core.thought import Document

            # Check which retriever interface this retriever implements
            if hasattr(retriever, "retrieve_for_thought_async"):
                # Core interface - async thought-based retrieval
                return await retriever.retrieve_for_thought_async(thought, is_pre_generation)
            elif hasattr(retriever, "retrieve_for_thought"):
                # Core interface - sync thought-based retrieval (deprecated)
                return retriever.retrieve_for_thought(thought, is_pre_generation)
            elif hasattr(retriever, "retrieve_async"):
                # Core interface - async query-based retrieval
                query = thought.text if thought.text else thought.prompt
                docs = await retriever.retrieve_async(query)
                # Convert to Document objects and add to thought
                documents = []
                for doc in docs:
                    if isinstance(doc, Document):
                        documents.append(doc)
                    elif isinstance(doc, dict):
                        documents.append(Document(**doc))
                    else:
                        documents.append(Document(text=str(doc)))

                if is_pre_generation:
                    return thought.add_pre_generation_context(documents)
                else:
                    return thought.add_post_generation_context(documents)
            elif hasattr(retriever, "retrieve"):
                # Simple interface - sync query-based retrieval
                query = thought.text if thought.text else thought.prompt
                docs = retriever.retrieve(query)
                # Convert to Document objects and add to thought
                documents = []
                for doc in docs:
                    if isinstance(doc, Document):
                        documents.append(doc)
                    elif isinstance(doc, dict):
                        documents.append(Document(**doc))
                    else:
                        documents.append(Document(text=str(doc)))

                if is_pre_generation:
                    return thought.add_pre_generation_context(documents)
                else:
                    return thought.add_post_generation_context(documents)
            else:
                logger.error(
                    f"Retriever {retriever.__class__.__name__} does not implement any known interface"
                )
                return thought

        except Exception as e:
            logger.error(f"Async retrieval failed for {retriever.__class__.__name__}: {e}")
            raise
