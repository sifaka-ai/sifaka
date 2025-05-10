"""
Ranking strategies for retrieval components.

This module provides ranking strategies for retrieval components in the Sifaka framework.
These strategies determine how retrieved documents are ranked based on their relevance
to the query.

## Ranking Strategies

1. **SimpleRankingStrategy**: Ranks documents based on keyword matching
2. **ScoreThresholdRankingStrategy**: Filters documents based on a score threshold

## Usage Examples

```python
from sifaka.retrieval.strategies.ranking import SimpleRankingStrategy, ScoreThresholdRankingStrategy
from sifaka.retrieval.config import RankingConfig

# Create a simple ranking strategy
config = RankingConfig(top_k=5)
strategy = SimpleRankingStrategy(config)

# Rank documents
documents = [
    {"content": "This is a document about machine learning."},
    {"content": "This document discusses natural language processing."},
]
ranked_docs = strategy.rank("machine learning", documents)

# Create a threshold strategy
threshold_strategy = ScoreThresholdRankingStrategy(
    base_strategy=strategy,
    threshold=0.5,
    config=config,
)

# Rank documents with threshold
filtered_docs = threshold_strategy.rank("machine learning", documents)
```
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sifaka.core.base import BaseComponent
from sifaka.utils.errors import RetrievalError, InputError, handle_error
from sifaka.utils.logging import get_logger

from ..config import RankingConfig

logger = get_logger(__name__)


class RankingStrategy(BaseComponent, ABC):
    """
    Abstract base class for ranking strategies.

    This class defines the interface for ranking strategies used by retrievers
    to rank retrieved documents based on their relevance to the query.

    ## Lifecycle Management

    1. **Initialization**
       - Configure ranking parameters
       - Initialize resources

    2. **Operation**
       - Rank documents based on relevance
       - Apply scoring functions
       - Filter results

    3. **Cleanup**
       - Release resources

    ## State Management

    The RankingStrategy maintains state for:
    - Configuration
    - Ranking statistics
    - Performance metrics
    """

    # State management is handled by BaseComponent

    def __init__(
        self,
        config: Optional[RankingConfig] = None,
        name: str = "RankingStrategy",
        description: str = "Base ranking strategy",
    ):
        """
        Initialize the ranking strategy.

        Args:
            config: The ranking configuration
            name: Name of the ranking strategy
            description: Description of the ranking strategy

        Raises:
            RetrievalError: If initialization fails
        """
        super().__init__(name=name, description=description)

        # Initialize state
        self._initialize_state(config)

        # Set metadata
        self._state_manager.set_metadata("component_type", "ranking_strategy")
        self._state_manager.set_metadata("creation_time", time.time())

    def _initialize_state(self, config: Optional[RankingConfig] = None) -> None:
        """
        Initialize the ranking strategy state.

        Args:
            config: The ranking configuration
        """
        # Call parent's _initialize_state first
        super()._initialize_state()

        config = config or RankingConfig()

        self._state_manager.update("config", config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("ranking_count", 0)
        self._state_manager.update("last_query", None)
        self._state_manager.update("last_result_count", 0)

    @property
    def config(self) -> RankingConfig:
        """
        Get the ranking configuration.

        Returns:
            The ranking configuration
        """
        return self._state_manager.get("config")

    @config.setter
    def config(self, config: RankingConfig) -> None:
        """
        Set the ranking configuration.

        Args:
            config: The new configuration

        Raises:
            RetrievalError: If the configuration is invalid
        """
        if not isinstance(config, RankingConfig):
            raise RetrievalError(
                "Config must be an instance of RankingConfig",
                metadata={"config_type": type(config).__name__},
            )
        self._state_manager.update("config", config)

    def process(self, input_data: Any, **kwargs: Any) -> Any:
        """
        Process input data.

        This method is required by the BaseComponent abstract class.

        Args:
            input_data: The input data to process
            **kwargs: Additional processing parameters

        Returns:
            The processed data
        """
        if isinstance(input_data, tuple) and len(input_data) == 2:
            query, documents = input_data
            return self.rank(query, documents, **kwargs)
        else:
            raise InputError(
                "Input data must be a tuple of (query, documents)",
                metadata={"input_type": type(input_data).__name__},
            )

    @abstractmethod
    def rank(
        self, query: str, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Rank documents based on their relevance to the query.

        Args:
            query: The query to rank documents for
            documents: The documents to rank
            **kwargs: Additional ranking parameters

        Returns:
            A list of ranked documents with scores

        Raises:
            InputError: If the query or documents are invalid
            RetrievalError: If ranking fails
        """
        # Track ranking count
        ranking_count = self._state_manager.get("ranking_count", 0)
        self._state_manager.update("ranking_count", ranking_count + 1)

        # Store the query in state
        self._state_manager.update("last_query", query)

        # Validate input
        if not query or not isinstance(query, str):
            raise InputError(
                "Query must be a non-empty string",
                metadata={
                    "query_type": type(query).__name__,
                    "query_length": len(str(query)) if query else 0,
                },
            )

        if not isinstance(documents, list):
            raise InputError(
                "Documents must be a list", metadata={"documents_type": type(documents).__name__}
            )

        # Record start time
        self._state_manager.set_metadata("last_ranking_start_time", time.time())

        # Subclasses should implement the actual ranking logic
        pass

    def _update_execution_stats(self, execution_time_ms: float, result_count: int) -> None:
        """
        Update execution statistics.

        Args:
            execution_time_ms: The execution time in milliseconds
            result_count: The number of results
        """
        # Update average execution time
        avg_time = self._state_manager.get_metadata("avg_execution_time_ms", 0)
        count = self._state_manager.get("ranking_count", 1)
        new_avg = ((avg_time * (count - 1)) + execution_time_ms) / count
        self._state_manager.set_metadata("avg_execution_time_ms", new_avg)

        # Update max execution time
        max_time = self._state_manager.get_metadata("max_execution_time_ms", 0)
        if execution_time_ms > max_time:
            self._state_manager.set_metadata("max_execution_time_ms", execution_time_ms)

        # Update result count
        self._state_manager.update("last_result_count", result_count)

        # Update average result count
        avg_count = self._state_manager.get_metadata("avg_result_count", 0)
        new_avg_count = ((avg_count * (count - 1)) + result_count) / count
        self._state_manager.set_metadata("avg_result_count", new_avg_count)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ranking strategy usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "name": self.name,
            "ranking_count": self._state_manager.get("ranking_count", 0),
            "avg_execution_time_ms": self._state_manager.get_metadata("avg_execution_time_ms", 0),
            "max_execution_time_ms": self._state_manager.get_metadata("max_execution_time_ms", 0),
            "avg_result_count": self._state_manager.get_metadata("avg_result_count", 0),
            "last_result_count": self._state_manager.get("last_result_count", 0),
            "top_k": self.config.top_k,
        }


class SimpleRankingStrategy(RankingStrategy):
    """
    Simple ranking strategy based on keyword matching.

    This strategy ranks documents based on the number of query keywords
    they contain. It's a basic strategy suitable for simple use cases.

    ## Ranking Algorithm

    1. Extract keywords from the query
    2. Count the number of keywords in each document
    3. Rank documents by keyword count
    4. Return the top-k documents

    ## Usage

    ```python
    strategy = SimpleRankingStrategy()
    documents = [
        {"content": "This is a document about machine learning."},
        {"content": "This document discusses natural language processing."},
    ]
    ranked_docs = strategy.rank("machine learning", documents)
    ```
    """

    def __init__(
        self,
        config: Optional[RankingConfig] = None,
        name: str = "SimpleRankingStrategy",
        description: str = "Simple ranking strategy based on keyword matching",
    ):
        """
        Initialize the simple ranking strategy.

        Args:
            config: The ranking configuration
            name: Name of the ranking strategy
            description: Description of the ranking strategy
        """
        super().__init__(config=config, name=name, description=description)

    def rank(
        self, query: str, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Rank documents based on keyword matching.

        Args:
            query: The query to rank documents for
            documents: The documents to rank
            **kwargs: Additional ranking parameters
                - top_k: Override the configured top_k value

        Returns:
            A list of ranked documents with scores

        Raises:
            InputError: If the query or documents are invalid
            RetrievalError: If ranking fails
        """
        # Call parent method to handle state tracking
        super().rank(query, documents, **kwargs)

        # Record start time
        start_time = time.time()

        try:
            if not documents:
                logger.debug(f"No documents to rank for query: {query}")
                return []

            # Get top_k from kwargs or config
            top_k = kwargs.get("top_k", self.config.top_k)

            # Extract keywords from query
            keywords = self._extract_keywords(query)
            if not keywords:
                # If no keywords, return documents in original order with zero scores
                logger.debug(f"No keywords extracted from query: {query}")
                result = [{**doc, "score": 0.0} for doc in documents[:top_k]]

                # Update execution statistics
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                self._update_execution_stats(execution_time_ms, len(result))

                return result

            # Calculate scores
            scored_docs = []
            for doc in documents:
                content = doc.get("content", "")
                if not isinstance(content, str):
                    content = str(content)

                score = self._calculate_score(content, keywords)
                scored_docs.append({**doc, "score": score})

            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            # Return top-k results
            result = scored_docs[:top_k]

            # Update execution statistics
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            self._update_execution_stats(execution_time_ms, len(result))

            # Store keyword statistics
            self._state_manager.set_metadata("last_keyword_count", len(keywords))

            return result

        except Exception as e:
            # If it's already a RetrievalError or InputError, re-raise
            if isinstance(e, (RetrievalError, InputError)):
                raise

            # Otherwise, wrap in RetrievalError
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Ranking failed: {str(e)}",
                metadata={"query": query, "document_count": len(documents), **error_info},
            )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query.

        Args:
            query: The query to extract keywords from

        Returns:
            A list of keywords
        """
        # Simple implementation: split by whitespace and lowercase
        return [word.lower() for word in query.split()]

    def _calculate_score(self, content: str, keywords: List[str]) -> float:
        """
        Calculate the relevance score for a document.

        Args:
            content: The document content
            keywords: The query keywords

        Returns:
            The relevance score
        """
        content_lower = content.lower()
        score = 0.0
        keyword_hits = 0

        for keyword in keywords:
            if keyword in content_lower:
                score += 1.0
                keyword_hits += 1

        # Normalize by number of keywords
        if keywords:
            score /= len(keywords)

        # Track keyword hit rate
        if keywords:
            hit_rate = keyword_hits / len(keywords)
            self._state_manager.set_metadata("last_keyword_hit_rate", hit_rate)

        return score

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ranking strategy usage.

        Returns:
            Dictionary with usage statistics
        """
        stats = super().get_statistics()
        stats.update(
            {
                "last_keyword_count": self._state_manager.get_metadata("last_keyword_count", 0),
                "last_keyword_hit_rate": self._state_manager.get_metadata(
                    "last_keyword_hit_rate", 0
                ),
            }
        )
        return stats


class ScoreThresholdRankingStrategy(RankingStrategy):
    """
    Ranking strategy that applies a score threshold.

    This strategy wraps another ranking strategy and filters out
    documents with scores below a threshold.

    ## Ranking Algorithm

    1. Use the wrapped strategy to rank documents
    2. Filter out documents with scores below the threshold
    3. Return the remaining documents

    ## Usage

    ```python
    base_strategy = SimpleRankingStrategy()
    threshold_strategy = ScoreThresholdRankingStrategy(
        base_strategy=base_strategy,
        threshold=0.5,
    )
    documents = [
        {"content": "This is a document about machine learning."},
        {"content": "This document discusses natural language processing."},
    ]
    ranked_docs = threshold_strategy.rank("machine learning", documents)
    ```
    """

    def __init__(
        self,
        base_strategy: RankingStrategy,
        threshold: float,
        config: Optional[RankingConfig] = None,
        name: str = "ScoreThresholdRankingStrategy",
        description: str = "Ranking strategy that applies a score threshold",
    ):
        """
        Initialize the score threshold ranking strategy.

        Args:
            base_strategy: The base ranking strategy to wrap
            threshold: The score threshold
            config: The ranking configuration
            name: Name of the ranking strategy
            description: Description of the ranking strategy

        Raises:
            RetrievalError: If initialization fails
        """
        super().__init__(config=config, name=name, description=description)

        if not isinstance(base_strategy, RankingStrategy):
            raise RetrievalError(
                "Base strategy must be an instance of RankingStrategy",
                metadata={"base_strategy_type": type(base_strategy).__name__},
            )

        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise RetrievalError(
                "Threshold must be a number between 0 and 1",
                metadata={"threshold": threshold, "threshold_type": type(threshold).__name__},
            )

        # Store base strategy and threshold in state
        self._state_manager.update("base_strategy", base_strategy)
        self._state_manager.update("threshold", threshold)

        # Set metadata
        self._state_manager.set_metadata("base_strategy_name", base_strategy.name)

    @property
    def base_strategy(self) -> RankingStrategy:
        """
        Get the base ranking strategy.

        Returns:
            The base ranking strategy
        """
        return self._state_manager.get("base_strategy")

    @base_strategy.setter
    def base_strategy(self, strategy: RankingStrategy) -> None:
        """
        Set the base ranking strategy.

        Args:
            strategy: The new base strategy

        Raises:
            RetrievalError: If the strategy is invalid
        """
        if not isinstance(strategy, RankingStrategy):
            raise RetrievalError(
                "Base strategy must be an instance of RankingStrategy",
                metadata={"base_strategy_type": type(strategy).__name__},
            )
        self._state_manager.update("base_strategy", strategy)
        self._state_manager.set_metadata("base_strategy_name", strategy.name)

    @property
    def threshold(self) -> float:
        """
        Get the score threshold.

        Returns:
            The score threshold
        """
        return self._state_manager.get("threshold")

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """
        Set the score threshold.

        Args:
            threshold: The new threshold

        Raises:
            RetrievalError: If the threshold is invalid
        """
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise RetrievalError(
                "Threshold must be a number between 0 and 1",
                metadata={"threshold": threshold, "threshold_type": type(threshold).__name__},
            )
        self._state_manager.update("threshold", threshold)

    def rank(
        self, query: str, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Rank documents and apply a score threshold.

        Args:
            query: The query to rank documents for
            documents: The documents to rank
            **kwargs: Additional ranking parameters
                - top_k: Override the configured top_k value
                - threshold: Override the configured threshold value

        Returns:
            A list of ranked documents with scores above the threshold

        Raises:
            InputError: If the query or documents are invalid
            RetrievalError: If ranking fails
        """
        # Call parent method to handle state tracking
        super().rank(query, documents, **kwargs)

        # Record start time
        start_time = time.time()

        try:
            # Get threshold from kwargs or state
            threshold = kwargs.get("threshold", self.threshold)

            # Get top_k from kwargs or config
            top_k = kwargs.get("top_k", self.config.top_k)

            # Use the base strategy to rank documents
            ranked_docs = self.base_strategy.rank(query, documents, **kwargs)

            # Filter out documents with scores below the threshold
            filtered_docs = [doc for doc in ranked_docs if doc.get("score", 0.0) >= threshold]

            # Return the filtered documents (up to top_k)
            result = filtered_docs[:top_k]

            # Update execution statistics
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            self._update_execution_stats(execution_time_ms, len(result))

            # Store filtering statistics
            self._state_manager.set_metadata("pre_filter_count", len(ranked_docs))
            self._state_manager.set_metadata("post_filter_count", len(filtered_docs))
            self._state_manager.set_metadata(
                "filter_ratio", len(filtered_docs) / len(ranked_docs) if ranked_docs else 0
            )

            return result

        except Exception as e:
            # If it's already a RetrievalError or InputError, re-raise
            if isinstance(e, (RetrievalError, InputError)):
                raise

            # Otherwise, wrap in RetrievalError
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Threshold ranking failed: {str(e)}",
                metadata={"query": query, "document_count": len(documents), **error_info},
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ranking strategy usage.

        Returns:
            Dictionary with usage statistics
        """
        stats = super().get_statistics()
        stats.update(
            {
                "threshold": self.threshold,
                "base_strategy": self.base_strategy.name,
                "pre_filter_count": self._state_manager.get_metadata("pre_filter_count", 0),
                "post_filter_count": self._state_manager.get_metadata("post_filter_count", 0),
                "filter_ratio": self._state_manager.get_metadata("filter_ratio", 0),
            }
        )
        return stats
