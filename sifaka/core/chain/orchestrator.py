"""Chain orchestration for Sifaka.

This module contains the ChainOrchestrator class which handles high-level
workflow coordination for chain execution.
"""

from typing import List

from sifaka.core.chain.config import ChainConfig
from sifaka.core.thought import Document, Thought
from sifaka.utils.logging import get_logger
from sifaka.utils.performance import time_operation

logger = get_logger(__name__)


class ChainOrchestrator:
    """High-level workflow coordinator for chain execution.

    This class manages the overall execution workflow of a chain,
    coordinating between different phases like retrieval, generation,
    validation, and criticism.
    """

    def __init__(self, config: ChainConfig):
        """Initialize the orchestrator with a chain configuration.

        Args:
            config: The chain configuration to use.
        """
        self.config = config

    def orchestrate_retrieval(self, thought: Thought, phase: str) -> Thought:
        """Orchestrate retrieval operations for a specific phase.

        Args:
            thought: The current thought state.
            phase: The retrieval phase ('pre_generation', 'post_generation', 'critic').

        Returns:
            Updated thought with retrieved context.
        """
        if phase == "pre_generation":
            return self._handle_pre_generation_retrieval(thought)
        elif phase == "post_generation":
            return self._handle_post_generation_retrieval(thought)
        elif phase == "critic":
            return self._handle_critic_retrieval(thought)
        else:
            logger.warning(f"Unknown retrieval phase: {phase}")
            return thought

    def _retrieve_documents(self, retrievers, query: str, retrieval_type: str) -> List[Document]:
        """Generic document retrieval method.

        Args:
            retrievers: List of retrievers to use.
            query: Query string for retrieval.
            retrieval_type: Type of retrieval for logging.

        Returns:
            List of retrieved documents.
        """
        if not retrievers:
            logger.debug(f"No retrievers configured, skipping {retrieval_type} retrieval")
            return []

        with time_operation(f"{retrieval_type}_retrieval"):
            logger.debug(f"Running {retrieval_type} retrieval with {len(retrievers)} retrievers")

            all_documents = []
            for retriever in retrievers:
                try:
                    document_texts = retriever.retrieve(query)
                    # Convert strings to Document objects
                    documents = [
                        Document(
                            text=text,
                            metadata={
                                "source": retriever.__class__.__name__,
                                "query": query,
                            },
                            score=1.0 - (i * 0.1),  # Simple scoring based on rank
                        )
                        for i, text in enumerate(document_texts)
                    ]
                    all_documents.extend(documents)
                    logger.debug(
                        f"Retrieved {len(documents)} documents from {retriever.__class__.__name__}"
                    )
                except Exception as e:
                    logger.warning(
                        f"{retrieval_type.title()} retrieval failed for {retriever.__class__.__name__}: {e}"
                    )

            logger.debug(f"Retrieved {len(all_documents)} total documents for {retrieval_type}")
            return all_documents

    def _handle_pre_generation_retrieval(self, thought: Thought) -> Thought:
        """Handle pre-generation retrieval using model retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with pre-generation context.
        """
        documents = self._retrieve_documents(
            self.config.model_retrievers, thought.prompt, "pre_generation"
        )

        if documents:
            thought = thought.add_pre_generation_context(documents)
            logger.debug(f"Added {len(documents)} documents to pre-generation context")

        return thought

    def _handle_post_generation_retrieval(self, thought: Thought) -> Thought:
        """Handle post-generation retrieval using model retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with post-generation context.
        """
        if not thought.text:
            logger.warning("No generated text available for post-generation retrieval")
            return thought

        documents = self._retrieve_documents(
            self.config.model_retrievers, thought.text, "post_generation"
        )

        if documents:
            thought = thought.add_post_generation_context(documents)
            logger.debug(f"Added {len(documents)} documents to post-generation context")

        return thought

    def _handle_critic_retrieval(self, thought: Thought) -> Thought:
        """Handle critic-specific retrieval using critic retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic context.
        """
        if not thought.text:
            logger.warning("No generated text available for critic retrieval")
            return thought

        documents = self._retrieve_documents(self.config.critic_retrievers, thought.text, "critic")

        if documents:
            # Add to post-generation context since critics use this context
            thought = thought.add_post_generation_context(documents)
            logger.debug(f"Added {len(documents)} documents to critic context")

        return thought

    def should_apply_critics(self, validation_passed: bool) -> bool:
        """Determine whether critics should be applied based on configuration and validation results.

        Args:
            validation_passed: Whether all validations passed.

        Returns:
            True if critics should be applied, False otherwise.
        """
        always_apply = self.config.get_option("always_apply_critics", False)
        apply_on_failure = self.config.get_option("apply_improvers_on_validation_failure", False)

        if always_apply:
            logger.debug("Applying critics because always_apply_critics is True")
            return True

        if not validation_passed and apply_on_failure:
            logger.debug(
                "Applying critics because validation failed and apply_improvers_on_validation_failure is True"
            )
            return True

        if not validation_passed and not apply_on_failure:
            logger.debug(
                "Skipping critics because validation failed but apply_improvers_on_validation_failure is False"
            )
            return False

        if validation_passed and not always_apply:
            logger.debug(
                "Skipping critics because validation passed and always_apply_critics is False"
            )
            return False

        return False

    def get_max_iterations(self) -> int:
        """Get the maximum number of improvement iterations.

        Returns:
            The maximum number of iterations.
        """
        return int(self.config.get_option("max_improvement_iterations", 3))
