"""
Ranking strategies for retrieval components.

This module provides ranking strategies for retrieval components in the Sifaka framework.
These strategies determine how retrieved documents are ranked based on their relevance
to the query.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config import RankingConfig


class RankingStrategy(ABC):
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
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize the ranking strategy.

        Args:
            config: The ranking configuration
        """
        self.config = config or RankingConfig()

    @abstractmethod
    def rank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs: Any
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
            ValueError: If the query or documents are invalid
            RuntimeError: If ranking fails
        """
        pass


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
    """

    def rank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Rank documents based on keyword matching.

        Args:
            query: The query to rank documents for
            documents: The documents to rank
            **kwargs: Additional ranking parameters

        Returns:
            A list of ranked documents with scores

        Raises:
            ValueError: If the query or documents are invalid
            RuntimeError: If ranking fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if not documents:
            return []

        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            # If no keywords, return documents in original order with zero scores
            return [
                {**doc, "score": 0.0}
                for doc in documents[:self.config.top_k]
            ]

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
        return scored_docs[:self.config.top_k]

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

        for keyword in keywords:
            if keyword in content_lower:
                score += 1.0

        # Normalize by number of keywords
        if keywords:
            score /= len(keywords)

        return score


class ScoreThresholdRankingStrategy(RankingStrategy):
    """
    Ranking strategy that applies a score threshold.

    This strategy wraps another ranking strategy and filters out
    documents with scores below a threshold.

    ## Ranking Algorithm

    1. Use the wrapped strategy to rank documents
    2. Filter out documents with scores below the threshold
    3. Return the remaining documents
    """

    def __init__(
        self,
        base_strategy: RankingStrategy,
        threshold: float,
        config: Optional[RankingConfig] = None,
    ):
        """
        Initialize the score threshold ranking strategy.

        Args:
            base_strategy: The base ranking strategy to wrap
            threshold: The score threshold
            config: The ranking configuration
        """
        super().__init__(config)
        self.base_strategy = base_strategy
        self.threshold = threshold

    def rank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Rank documents and apply a score threshold.

        Args:
            query: The query to rank documents for
            documents: The documents to rank
            **kwargs: Additional ranking parameters

        Returns:
            A list of ranked documents with scores above the threshold

        Raises:
            ValueError: If the query or documents are invalid
            RuntimeError: If ranking fails
        """
        # Use the base strategy to rank documents
        ranked_docs = self.base_strategy.rank(query, documents, **kwargs)

        # Filter out documents with scores below the threshold
        filtered_docs = [
            doc for doc in ranked_docs
            if doc.get("score", 0.0) >= self.threshold
        ]

        # Return the filtered documents (up to top_k)
        return filtered_docs[:self.config.top_k]
