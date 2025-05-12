"""
Query management for retrieval components.

This module provides query management functionality for retrieval components
in the Sifaka framework. It includes the QueryManager class, which handles
query preprocessing, expansion, and other query-related operations.

## Query Processing

1. **Preprocessing**: Apply preprocessing steps to the query
   - Lowercase: Convert the query to lowercase
   - Remove Stopwords: Remove common stopwords from the query
   - Remove Punctuation: Remove punctuation from the query

2. **Expansion**: Expand the query with additional terms
   - Synonym Expansion: Add synonyms of query terms
   - Word Embedding Expansion: Add related terms based on word embeddings
   - Knowledge Graph Expansion: Add related terms from a knowledge graph

3. **Normalization**: Normalize the query for better matching
   - Stemming: Apply stemming to the query terms
   - Lemmatization: Apply lemmatization to the query terms
"""

import time
from typing import Any, Dict, Optional, Set

from sifaka.core.base import BaseComponent
from sifaka.utils.errors.component import RetrievalError
from sifaka.utils.errors.base import InputError
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger
from sifaka.utils.patterns import PUNCTUATION_PATTERN, replace_pattern
from sifaka.utils.config.retrieval import QueryProcessingConfig
from sifaka.interfaces.retrieval import QueryProcessor

logger = get_logger(__name__)


class QueryManager(BaseComponent, QueryProcessor):
    """
    Manager for query processing.

    This class handles query preprocessing, expansion, and other query-related
    operations for retrieval components in the Sifaka framework.

    ## Preprocessing Steps

    1. **Lowercase**: Convert the query to lowercase
    2. **Remove Stopwords**: Remove common stopwords from the query
    3. **Remove Punctuation**: Remove punctuation from the query
    4. **Stemming**: Apply stemming to the query terms
    5. **Lemmatization**: Apply lemmatization to the query terms

    ## Query Expansion Methods

    1. **Synonym Expansion**: Add synonyms of query terms
    2. **Word Embedding Expansion**: Add related terms based on word embeddings
    3. **Knowledge Graph Expansion**: Add related terms from a knowledge graph

    ## State Management

    The QueryManager maintains state for:
    - Configuration
    - Stopwords
    - Processing statistics
    - Cache of processed queries
    """

    # State management is handled by BaseComponent

    def __init__(
        self,
        config: Optional[QueryProcessingConfig] = None,
        name: str = "QueryManager",
        description: str = "Manager for query processing",
    ):
        """
        Initialize the query manager.

        Args:
            config: The query processing configuration
            name: Name of the query manager
            description: Description of the query manager

        Raises:
            RetrievalError: If initialization fails
        """
        super().__init__(name=name, description=description)

        # Initialize state
        self._initialize_state(config)

        # Set metadata
        self._state_manager.set_metadata("component_type", "query_manager")
        self._state_manager.set_metadata("creation_time", time.time())

    def _initialize_state(self, config: Optional[QueryProcessingConfig] = None) -> None:
        """
        Initialize the query manager state.

        Args:
            config: The query processing configuration
        """
        # Call parent's _initialize_state first
        super()._initialize_state()

        config = config or QueryProcessingConfig()

        self._state_manager.update("config", config)
        self._state_manager.update("stopwords", self._get_default_stopwords())
        self._state_manager.update("initialized", True)
        self._state_manager.update("query_count", 0)
        self._state_manager.update("query_cache", {})

    @property
    def config(self) -> QueryProcessingConfig:
        """
        Get the query processing configuration.

        Returns:
            The query processing configuration
        """
        return self._state_manager.get("config")

    @config.setter
    def config(self, config: QueryProcessingConfig) -> None:
        """
        Set the query processing configuration.

        Args:
            config: The new configuration

        Raises:
            RetrievalError: If the configuration is invalid
        """
        if not isinstance(config, QueryProcessingConfig):
            raise RetrievalError(
                "Config must be an instance of QueryProcessingConfig",
                metadata={"config_type": type(config).__name__},
            )
        self._state_manager.update("config", config)

    @property
    def stopwords(self) -> Set[str]:
        """
        Get the stopwords set.

        Returns:
            The set of stopwords
        """
        return self._state_manager.get("stopwords")

    def _get_default_stopwords(self) -> Set[str]:
        """
        Get the default set of stopwords.

        Returns:
            A set of common English stopwords
        """
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "of",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

    def _lowercase(self, query: str) -> str:
        """
        Convert a query to lowercase.

        Args:
            query: The query to convert

        Returns:
            The lowercase query
        """
        return query.lower()

    def _remove_stopwords(self, query: str) -> str:
        """
        Remove stopwords from a query.

        Args:
            query: The query to process

        Returns:
            The query with stopwords removed
        """
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)

    def _remove_punctuation(self, query: str) -> str:
        """
        Remove punctuation from a query.

        Args:
            query: The query to process

        Returns:
            The query with punctuation removed
        """
        return replace_pattern(query, PUNCTUATION_PATTERN, "")

    def _expand_query(self, query: str) -> str:
        """
        Expand a query with additional terms.

        Args:
            query: The query to expand

        Returns:
            The expanded query

        Raises:
            RetrievalError: If expansion fails
        """
        try:
            # Get expansion method from config
            expansion_method = self.config.expansion_method
            if expansion_method is None:
                return query

            # Simple implementation for demonstration purposes
            if expansion_method == "synonym":
                # Add some common synonyms (this is just a placeholder)
                synonyms = {
                    "good": ["great", "excellent"],
                    "bad": ["poor", "terrible"],
                    "big": ["large", "huge"],
                    "small": ["tiny", "little"],
                }

                words = query.split()
                expanded_words = []

                for word in words:
                    expanded_words.append(word)
                    if word in synonyms:
                        expanded_words.extend(synonyms[word])

                return " ".join(expanded_words)

            # Add more expansion methods here

            return query

        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Query expansion failed: {str(e)}",
                metadata={"query": query, "expansion_method": expansion_method, **error_info},
            )

    def process_query(self, query: str, **kwargs: Any) -> str:
        """
        Process a query.

        This method applies the configured preprocessing steps and
        expansion methods to the query.

        Args:
            query: The query to process
            **kwargs: Additional processing parameters
                - skip_cache: Whether to skip the cache (default: False)
                - preprocessing_steps: Override the configured preprocessing steps
                - expansion_method: Override the configured expansion method

        Returns:
            The processed query

        Raises:
            InputError: If the query is invalid
            RetrievalError: If processing fails
        """
        # Track query count
        query_count = self._state_manager.get("query_count", 0)
        self._state_manager.update("query_count", query_count + 1)

        # Validate input
        if not query or not isinstance(query, str):
            raise InputError(
                "Query must be a non-empty string",
                metadata={
                    "query_type": type(query).__name__,
                    "query_length": len(str(query)) if query else 0,
                },
            )

        # Check cache
        skip_cache = kwargs.get("skip_cache", False)
        if not skip_cache:
            cache = self._state_manager.get("query_cache", {})
            cache_key = f"{query}_{kwargs}"
            if cache_key in cache:
                self._state_manager.set_metadata("cache_hit", True)
                return cache[cache_key]

        # Mark as cache miss
        self._state_manager.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            processed_query = query

            # Get preprocessing steps (from kwargs or config)
            preprocessing_steps = kwargs.get("preprocessing_steps", self.config.preprocessing_steps)

            # Apply preprocessing steps in the configured order
            for step in preprocessing_steps:
                if step == "lowercase":
                    processed_query = self._lowercase(processed_query)
                elif step == "remove_stopwords":
                    processed_query = self._remove_stopwords(processed_query)
                elif step == "remove_punctuation":
                    processed_query = self._remove_punctuation(processed_query)

            # Get expansion method (from kwargs or config)
            expansion_method = kwargs.get("expansion_method", self.config.expansion_method)

            # Apply query expansion if configured
            if expansion_method:
                # Update config temporarily for expansion
                original_expansion = self.config.expansion_method
                self.config.expansion_method = expansion_method

                # Expand query
                processed_query = self._expand_query(processed_query)

                # Restore original config
                self.config.expansion_method = original_expansion

            # Record execution time
            end_time = time.time()
            exec_time_ms = (end_time - start_time) * 1000

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time_ms", 0)
            count = self._state_manager.get("query_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time_ms) / count
            self._state_manager.set_metadata("avg_execution_time_ms", new_avg)

            # Cache result
            if not skip_cache:
                cache = self._state_manager.get("query_cache", {})
                cache[cache_key] = processed_query
                self._state_manager.update("query_cache", cache)

            return processed_query

        except Exception as e:
            # If it's already a RetrievalError or InputError, re-raise
            if isinstance(e, (RetrievalError, InputError)):
                raise

            # Otherwise, wrap in RetrievalError
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Query processing failed: {str(e)}", metadata={"query": query, **error_info}
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about query processing.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "name": self.name,
            "query_count": self._state_manager.get("query_count", 0),
            "avg_execution_time_ms": self._state_manager.get_metadata("avg_execution_time_ms", 0),
            "cache_size": len(self._state_manager.get("query_cache", {})),
            "stopwords_count": len(self.stopwords),
            "preprocessing_steps": self.config.preprocessing_steps,
            "expansion_method": self.config.expansion_method,
        }

    def clear_cache(self) -> None:
        """
        Clear the query cache.
        """
        self._state_manager.update("query_cache", {})
        logger.debug(f"Cleared cache for query manager {self.name}")

    def process(self, input: str) -> str:
        """
        Process the input query.

        This method is required by the BaseComponent interface.

        Args:
            input: The query to process

        Returns:
            The processed query
        """
        return self.process_query(input)
