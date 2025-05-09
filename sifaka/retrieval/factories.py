"""
Factory functions for creating retrieval components.

This module provides factory functions for creating retrieval components
in the Sifaka framework. These functions simplify the creation of retrievers
and related components with common configurations.
"""

from typing import Any, Dict, Optional

from .config import RetrieverConfig
from .implementations.simple import SimpleRetriever
from .strategies.ranking import SimpleRankingStrategy, ScoreThresholdRankingStrategy


def create_simple_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    max_results: int = 3,
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
        **kwargs: Additional configuration parameters

    Returns:
        A SimpleRetriever instance

    Raises:
        ValueError: If both documents and corpus are None
        FileNotFoundError: If corpus file doesn't exist
    """
    config = RetrieverConfig(
        retriever_type="simple",
        ranking={"top_k": max_results},
        **kwargs,
    )
    
    return SimpleRetriever(
        documents=documents,
        corpus=corpus,
        config=config,
    )


def create_threshold_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    max_results: int = 3,
    threshold: float = 0.5,
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
        **kwargs: Additional configuration parameters

    Returns:
        A SimpleRetriever instance with score thresholding

    Raises:
        ValueError: If both documents and corpus are None
        FileNotFoundError: If corpus file doesn't exist
    """
    config = RetrieverConfig(
        retriever_type="simple",
        ranking={"top_k": max_results, "score_threshold": threshold},
        **kwargs,
    )
    
    retriever = SimpleRetriever(
        documents=documents,
        corpus=corpus,
        config=config,
    )
    
    # Replace the default ranking strategy with a threshold strategy
    base_strategy = SimpleRankingStrategy(retriever.config.ranking)
    retriever.ranking_strategy = ScoreThresholdRankingStrategy(
        base_strategy=base_strategy,
        threshold=threshold,
        config=retriever.config.ranking,
    )
    
    return retriever
