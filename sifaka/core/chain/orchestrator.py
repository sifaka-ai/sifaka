"""Chain orchestration for Sifaka.

This module contains the ChainOrchestrator class which handles high-level
workflow coordination for chain execution.
"""

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

    def _handle_pre_generation_retrieval(self, thought: Thought) -> Thought:
        """Handle pre-generation retrieval using model retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with pre-generation context.
        """
        if not self.config.model_retrievers:
            logger.debug("No model retrievers configured, skipping pre-generation retrieval")
            return thought

        with time_operation("pre_generation_retrieval"):
            logger.debug(
                f"Running pre-generation retrieval with {len(self.config.model_retrievers)} retrievers"
            )

            all_documents = []
            for retriever in self.config.model_retrievers:
                try:
                    document_texts = retriever.retrieve(thought.prompt)
                    # Convert strings to Document objects
                    documents = [
                        Document(
                            text=text,
                            metadata={
                                "source": retriever.__class__.__name__,
                                "query": thought.prompt,
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
                        f"Pre-generation retrieval failed for {retriever.__class__.__name__}: {e}"
                    )

            if all_documents:
                thought = thought.add_pre_generation_context(all_documents)
                logger.debug(f"Added {len(all_documents)} documents to pre-generation context")

            return thought

    def _handle_post_generation_retrieval(self, thought: Thought) -> Thought:
        """Handle post-generation retrieval using model retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with post-generation context.
        """
        if not self.config.model_retrievers:
            logger.debug("No model retrievers configured, skipping post-generation retrieval")
            return thought

        if not thought.text:
            logger.warning("No generated text available for post-generation retrieval")
            return thought

        with time_operation("post_generation_retrieval"):
            logger.debug(
                f"Running post-generation retrieval with {len(self.config.model_retrievers)} retrievers"
            )

            all_documents = []
            for retriever in self.config.model_retrievers:
                try:
                    document_texts = retriever.retrieve(thought.text)
                    # Convert strings to Document objects
                    documents = [
                        Document(
                            text=text,
                            metadata={
                                "source": retriever.__class__.__name__,
                                "query": thought.text,
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
                        f"Post-generation retrieval failed for {retriever.__class__.__name__}: {e}"
                    )

            if all_documents:
                thought = thought.add_post_generation_context(all_documents)
                logger.debug(f"Added {len(all_documents)} documents to post-generation context")

            return thought

    def _handle_critic_retrieval(self, thought: Thought) -> Thought:
        """Handle critic-specific retrieval using critic retrievers.

        Args:
            thought: The current thought state.

        Returns:
            Updated thought with critic context.
        """
        if not self.config.critic_retrievers:
            logger.debug("No critic retrievers configured, skipping critic retrieval")
            return thought

        if not thought.text:
            logger.warning("No generated text available for critic retrieval")
            return thought

        with time_operation("critic_retrieval"):
            logger.debug(
                f"Running critic retrieval with {len(self.config.critic_retrievers)} retrievers"
            )

            all_documents = []
            for retriever in self.config.critic_retrievers:
                try:
                    # Use the generated text as the query for critic retrieval
                    document_texts = retriever.retrieve(thought.text)
                    # Convert strings to Document objects
                    documents = [
                        Document(
                            text=text,
                            metadata={
                                "source": retriever.__class__.__name__,
                                "query": thought.text,
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
                        f"Critic retrieval failed for {retriever.__class__.__name__}: {e}"
                    )

            if all_documents:
                # Add to post-generation context since critics use this context
                thought = thought.add_post_generation_context(all_documents)
                logger.debug(f"Added {len(all_documents)} documents to critic context")

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
