"""
Configuration for retrieval components.

This module provides configuration classes for retrieval components in the Sifaka framework.
These classes define the configuration options for retrievers and related components.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RankingConfig(BaseModel):
    """
    Configuration for ranking strategies.

    This class defines the configuration options for ranking strategies
    used by retrievers to rank retrieved documents.
    """

    strategy: str = Field(
        default="simple",
        description="The ranking strategy to use (e.g., 'simple', 'bm25', 'semantic')",
    )
    top_k: int = Field(
        default=3,
        description="The number of top results to return",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="The minimum score threshold for results (if None, no threshold is applied)",
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the ranking strategy",
    )


class IndexConfig(BaseModel):
    """
    Configuration for index management.

    This class defines the configuration options for index management
    used by retrievers to manage document indexes.
    """

    index_type: str = Field(
        default="in_memory",
        description="The type of index to use (e.g., 'in_memory', 'disk', 'database')",
    )
    index_path: Optional[str] = Field(
        default=None,
        description="The path to the index (if applicable)",
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the index",
    )


class QueryProcessingConfig(BaseModel):
    """
    Configuration for query processing.

    This class defines the configuration options for query processing
    used by retrievers to process queries.
    """

    preprocessing_steps: List[str] = Field(
        default_factory=lambda: ["lowercase", "remove_stopwords"],
        description="The preprocessing steps to apply to queries",
    )
    expansion_method: Optional[str] = Field(
        default=None,
        description="The query expansion method to use (if any)",
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for query processing",
    )


class RetrieverConfig(BaseModel):
    """
    Configuration for retrievers.

    This class defines the configuration options for retrievers
    in the Sifaka framework.
    """

    retriever_type: str = Field(
        default="simple",
        description="The type of retriever to use (e.g., 'simple', 'vector', 'hybrid')",
    )
    ranking: RankingConfig = Field(
        default_factory=RankingConfig,
        description="Configuration for ranking",
    )
    index: IndexConfig = Field(
        default_factory=IndexConfig,
        description="Configuration for index management",
    )
    query_processing: QueryProcessingConfig = Field(
        default_factory=QueryProcessingConfig,
        description="Configuration for query processing",
    )
    additional_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the retriever",
    )

    def with_ranking_strategy(self, strategy: str) -> "RetrieverConfig":
        """
        Set the ranking strategy.

        Args:
            strategy: The ranking strategy to use

        Returns:
            The updated configuration
        """
        self.ranking.strategy = strategy
        return self

    def with_top_k(self, top_k: int) -> "RetrieverConfig":
        """
        Set the number of top results to return.

        Args:
            top_k: The number of top results to return

        Returns:
            The updated configuration
        """
        self.ranking.top_k = top_k
        return self

    def with_score_threshold(self, threshold: Optional[float]) -> "RetrieverConfig":
        """
        Set the score threshold for results.

        Args:
            threshold: The score threshold (if None, no threshold is applied)

        Returns:
            The updated configuration
        """
        self.ranking.score_threshold = threshold
        return self

    def with_index_type(self, index_type: str) -> "RetrieverConfig":
        """
        Set the index type.

        Args:
            index_type: The type of index to use

        Returns:
            The updated configuration
        """
        self.index.index_type = index_type
        return self

    def with_index_path(self, index_path: Optional[str]) -> "RetrieverConfig":
        """
        Set the index path.

        Args:
            index_path: The path to the index (if applicable)

        Returns:
            The updated configuration
        """
        self.index.index_path = index_path
        return self

    def with_preprocessing_steps(self, steps: List[str]) -> "RetrieverConfig":
        """
        Set the preprocessing steps for query processing.

        Args:
            steps: The preprocessing steps to apply to queries

        Returns:
            The updated configuration
        """
        self.query_processing.preprocessing_steps = steps
        return self

    def with_expansion_method(self, method: Optional[str]) -> "RetrieverConfig":
        """
        Set the query expansion method.

        Args:
            method: The query expansion method to use (if any)

        Returns:
            The updated configuration
        """
        self.query_processing.expansion_method = method
        return self
