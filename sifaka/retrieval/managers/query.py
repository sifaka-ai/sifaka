"""
Query management for retrieval components.

This module provides query management functionality for retrieval components
in the Sifaka framework. It includes the QueryManager class, which handles
query preprocessing, expansion, and other query-related operations.
"""

import re
from typing import Any, Dict, List, Optional, Set

from ..config import QueryProcessingConfig
from ..interfaces.retriever import QueryProcessor


class QueryManager(QueryProcessor):
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
    """

    def __init__(self, config: Optional[QueryProcessingConfig] = None):
        """
        Initialize the query manager.

        Args:
            config: The query processing configuration
        """
        self.config = config or QueryProcessingConfig()
        self._stopwords = self._get_default_stopwords()

    def _get_default_stopwords(self) -> Set[str]:
        """
        Get the default set of stopwords.

        Returns:
            A set of common English stopwords
        """
        return {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "in", "on", "at", "to", "for", "with",
            "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down", "of",
            "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now"
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
        filtered_words = [word for word in words if word.lower() not in self._stopwords]
        return " ".join(filtered_words)

    def _remove_punctuation(self, query: str) -> str:
        """
        Remove punctuation from a query.

        Args:
            query: The query to process

        Returns:
            The query with punctuation removed
        """
        return re.sub(r'[^\w\s]', '', query)

    def _expand_query(self, query: str) -> str:
        """
        Expand a query with additional terms.

        Args:
            query: The query to expand

        Returns:
            The expanded query
        """
        # This is a placeholder implementation
        # Subclasses should override this method to implement actual query expansion
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
            
        return query

    def process_query(self, query: str, **kwargs: Any) -> str:
        """
        Process a query.

        This method applies the configured preprocessing steps and
        expansion methods to the query.

        Args:
            query: The query to process
            **kwargs: Additional processing parameters

        Returns:
            The processed query

        Raises:
            ValueError: If the query is invalid
            RuntimeError: If processing fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        processed_query = query
        
        # Apply preprocessing steps in the configured order
        for step in self.config.preprocessing_steps:
            if step == "lowercase":
                processed_query = self._lowercase(processed_query)
            elif step == "remove_stopwords":
                processed_query = self._remove_stopwords(processed_query)
            elif step == "remove_punctuation":
                processed_query = self._remove_punctuation(processed_query)
                
        # Apply query expansion if configured
        if self.config.expansion_method:
            processed_query = self._expand_query(processed_query)
            
        return processed_query
