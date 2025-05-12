"""
Factory functions for creating retrieval components.

This module provides factory functions for creating retrieval components
in the Sifaka framework. These functions simplify the creation of retrievers
and related components with common configurations.

## Standardized Factory Pattern

These factory functions follow the standardized pattern used across all Sifaka components:
1. They create a configuration object with the provided parameters
2. They create a component instance with the configuration
3. They initialize the component
4. They return the initialized component

## Error Handling

All factory functions use standardized error handling:
1. They catch and handle exceptions
2. They use the handle_error utility function
3. They wrap exceptions in component-specific error types
"""

from typing import Any, Dict, Optional

from sifaka.utils.errors.component import RetrievalError
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger
from sifaka.interfaces.factories import create_component
from sifaka.utils.config.retrieval import RetrieverConfig

from .implementations.simple import SimpleRetriever
from .strategies.ranking import SimpleRankingStrategy, ScoreThresholdRankingStrategy

logger = get_logger(__name__)


def create_simple_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[Optional[str]] = None,
    max_results: int = 3,
    name: str = "SimpleRetriever",
    description: str = "Simple retriever for in-memory document collections",
    **kwargs: Any,
) -> SimpleRetriever:
    """
    Create a simple retriever.

    This function creates a SimpleRetriever with the specified documents
    and configuration, following the standardized component creation pattern.

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
            top_k=max_results,
            max_results=max_results,
            **kwargs,
        )

        # Create retriever using the standardized create_component function
        retriever = create_component(
            component_class=SimpleRetriever,
            name=name,
            description=description,
            config=config,
            documents=documents,
            corpus=corpus,
        )

        (logger and logger.debug(f"Created simple retriever with {len(retriever.documents)} documents")
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
    corpus: Optional[Optional[str]] = None,
    max_results: int = 3,
    threshold: float = 0.5,
    name: str = "ThresholdRetriever",
    description: str = "Retriever with score thresholding",
    **kwargs: Any,
) -> SimpleRetriever:
    """
    Create a retriever with a score threshold.

    This function creates a SimpleRetriever with a ScoreThresholdRankingStrategy
    that filters out documents with scores below the threshold, following the
    standardized component creation pattern.

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
            top_k=max_results,
            max_results=max_results,
            score_threshold=threshold,
            **kwargs,
        )

        # Create retriever using the standardized create_component function
        retriever = create_component(
            component_class=SimpleRetriever,
            name=name,
            description=description,
            config=config,
            documents=documents,
            corpus=corpus,
        )

        # Replace the default ranking strategy with a threshold strategy
        base_strategy = SimpleRankingStrategy(retriever.config.ranking)
        threshold_strategy = ScoreThresholdRankingStrategy(
            base_strategy=base_strategy,
            threshold=threshold,
            config=retriever.config.ranking,
        )

        # Update the ranking strategy in state
        retriever.(_state_manager and _state_manager.update("ranking_strategy", threshold_strategy)

        (logger and logger.debug(
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
