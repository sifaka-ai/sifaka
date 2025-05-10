"""
Factory functions for creating retrieval components.

This module provides factory functions for creating retrieval components
in the Sifaka framework. These functions simplify the creation of retrievers
and related components with common configurations.
"""

from typing import Any, Dict, Optional

from sifaka.utils.errors import RetrievalError, handle_error
from sifaka.utils.logging import get_logger

from .config import RetrieverConfig
from .implementations.simple import SimpleRetriever
from .strategies.ranking import SimpleRankingStrategy, ScoreThresholdRankingStrategy

logger = get_logger(__name__)


def create_simple_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    max_results: int = 3,
    name: str = "SimpleRetriever",
    description: str = "Simple retriever for in-memory document collections",
    **kwargs: Any,
) -> SimpleRetriever:
    """
    Create a simple retriever.

    This function creates a SimpleRetriever with the specified documents
    and configuration.

    Args:
        documents: Dictionary mapping document keys to content
        corpus: Path to a text file containing documents (one per line)
        max_results: Maximum number of results to return
        name: Name of the retriever
        description: Description of the retriever
        **kwargs: Additional configuration parameters

    Returns:
        A SimpleRetriever instance

    Raises:
        RetrievalError: If creation fails
        FileNotFoundError: If corpus file doesn't exist
    """
    try:
        # Create configuration
        config = RetrieverConfig(
            retriever_type="simple",
            ranking={"top_k": max_results},
            **kwargs,
        )

        # Create retriever
        retriever = SimpleRetriever(
            documents=documents,
            corpus=corpus,
            config=config,
            name=name,
            description=description,
        )

        # Initialize the retriever
        retriever.initialize()

        logger.debug(f"Created simple retriever with {len(retriever.documents)} documents")
        return retriever

    except FileNotFoundError:
        # Re-raise file not found errors
        raise
    except Exception as e:
        # Handle other errors
        error_info = handle_error(e, "create_simple_retriever", "error")
        raise RetrievalError(f"Failed to create simple retriever: {str(e)}", metadata=error_info)


def create_threshold_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    max_results: int = 3,
    threshold: float = 0.5,
    name: str = "ThresholdRetriever",
    description: str = "Retriever with score thresholding",
    **kwargs: Any,
) -> SimpleRetriever:
    """
    Create a retriever with a score threshold.

    This function creates a SimpleRetriever with a ScoreThresholdRankingStrategy
    that filters out documents with scores below the threshold.

    Args:
        documents: Dictionary mapping document keys to content
        corpus: Path to a text file containing documents (one per line)
        max_results: Maximum number of results to return
        threshold: Minimum score threshold for results
        name: Name of the retriever
        description: Description of the retriever
        **kwargs: Additional configuration parameters

    Returns:
        A SimpleRetriever instance with score thresholding

    Raises:
        RetrievalError: If creation fails
        FileNotFoundError: If corpus file doesn't exist
    """
    try:
        # Create configuration
        config = RetrieverConfig(
            retriever_type="threshold",
            ranking={"top_k": max_results, "score_threshold": threshold},
            **kwargs,
        )

        # Create retriever
        retriever = SimpleRetriever(
            documents=documents,
            corpus=corpus,
            config=config,
            name=name,
            description=description,
        )

        # Replace the default ranking strategy with a threshold strategy
        base_strategy = SimpleRankingStrategy(retriever.config.ranking)
        threshold_strategy = ScoreThresholdRankingStrategy(
            base_strategy=base_strategy,
            threshold=threshold,
            config=retriever.config.ranking,
        )

        # Update the ranking strategy in state
        retriever._state_manager.update("ranking_strategy", threshold_strategy)

        # Initialize the retriever
        retriever.initialize()

        logger.debug(
            f"Created threshold retriever with {len(retriever.documents)} documents "
            f"and threshold {threshold}"
        )
        return retriever

    except FileNotFoundError:
        # Re-raise file not found errors
        raise
    except Exception as e:
        # Handle other errors
        error_info = handle_error(e, "create_threshold_retriever", "error")
        raise RetrievalError(f"Failed to create threshold retriever: {str(e)}", metadata=error_info)
