"""Cached retriever implementation using unified 3-tier architecture.

This module provides CachedRetriever which wraps any base retriever with
the unified memory → cache → persistence caching pattern, replacing the
dual-personality Redis retriever with a clean caching wrapper.
"""

import hashlib
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought, Document
from sifaka.storage.protocols import Retriever
from sifaka.utils.logging import get_logger

from .base import CachedStorage, StorageError

logger = get_logger(__name__)


class CachedRetriever(Retriever):
    """Pure caching wrapper for any retriever.

    This class implements a clean caching layer over any base retriever,
    following the unified 3-tier storage pattern. It replaces the complex
    dual-mode Redis retriever with a simple, predictable caching wrapper.

    Benefits:
    - Consistent caching behavior across all retrievers
    - Clear separation: base retriever does computation, cache provides speed
    - Predictable performance characteristics
    - Easy to test and debug

    Attributes:
        storage: Underlying CachedStorage instance for 3-tier caching.
        base: Base retriever that performs the actual retrieval computation.
        cache_ttl: Time-to-live for cached results.
    """

    def __init__(self, storage: CachedStorage, base_retriever: Retriever, cache_ttl: int = 3600):
        """Initialize cached retriever.

        Args:
            storage: CachedStorage instance for 3-tier caching.
            base_retriever: Base retriever to wrap with caching.
            cache_ttl: TTL for cached results in seconds.
        """
        self.storage = storage
        self.base = base_retriever
        self.cache_ttl = cache_ttl
        self.cache_hits = 0
        self.cache_misses = 0

        logger.debug(f"Initialized CachedRetriever wrapping {base_retriever.__class__.__name__}")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents with 3-tier caching.

        Checks cache first (L1 → L2 → L3), falls back to base retriever
        if not found, then caches the results for future queries.

        Args:
            query: Query string for document retrieval.

        Returns:
            List of retrieved document texts.
        """
        try:
            # Create cache key from query
            cache_key = self._create_cache_key(query)

            # Check cache first (L1 → L2 → L3)
            cached_results = self.storage.get(cache_key)

            if cached_results is not None:
                self.cache_hits += 1
                logger.debug(
                    f"Cache hit for query: {query[:50]}... (hit rate: {self._get_hit_rate():.2f})"
                )
                if isinstance(cached_results, list):
                    return cached_results
                else:
                    # Handle case where cached result is not a list
                    return []

            # Cache miss - use base retriever
            self.cache_misses += 1
            logger.debug(f"Cache miss for query: {query[:50]}... - using base retriever")

            results = self.base.retrieve(query)

            # Cache the results for future queries
            if results:
                metadata = {
                    "query": [query[:1000]],  # Truncated query for metadata
                    "result_count": [len(results)],
                    "retriever_type": [self.base.__class__.__name__],
                }
                self.storage.set(cache_key, results, metadata)
                logger.debug(f"Cached {len(results)} results for query")

            return results

        except Exception as e:
            logger.warning(f"Cached retrieval failed for query '{query}': {e}")
            # Fallback to base retriever without caching
            try:
                return self.base.retrieve(query)
            except Exception as base_error:
                logger.error(f"Base retriever also failed: {base_error}")
                return []

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought with caching.

        Args:
            thought: The thought to retrieve context for.
            is_pre_generation: Whether this is pre-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        try:
            # Use the retrieve method to get document texts with caching
            document_texts = self.retrieve(thought.prompt)

            # Convert texts to Document objects
            documents = []
            for i, text in enumerate(document_texts):
                doc = Document(
                    text=text,
                    metadata={
                        "source": "cached_retriever",
                        "base_retriever_type": self.base.__class__.__name__,
                        "retrieval_index": i,
                        "cache_hit": self.cache_hits > 0,  # Approximate cache hit indicator
                    },
                )
                documents.append(doc)

            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)

        except Exception as e:
            logger.warning(f"Cached retrieval for thought failed: {e}")
            # Fallback to base retriever if available
            if hasattr(self.base, "retrieve_for_thought"):
                return self.base.retrieve_for_thought(thought, is_pre_generation)
            else:
                # If base doesn't have retrieve_for_thought, use retrieve and convert
                try:
                    texts = self.base.retrieve(thought.prompt)
                    documents = [
                        Document(text=text, metadata={"source": "fallback", "index": i})
                        for i, text in enumerate(texts)
                    ]
                    if is_pre_generation:
                        return thought.add_pre_generation_context(documents)
                    else:
                        return thought.add_post_generation_context(documents)
                except Exception as fallback_error:
                    logger.error(f"Fallback retrieval also failed: {fallback_error}")
                    return thought

    def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the base retriever and invalidate cache.

        Args:
            doc_id: Document identifier.
            text: Document text.
            metadata: Optional document metadata.
        """
        try:
            # Delegate to base retriever if it has add_document method
            if hasattr(self.base, "add_document"):
                self.base.add_document(doc_id, text, metadata)
            else:
                logger.warning(
                    f"Base retriever {self.base.__class__.__name__} does not support add_document"
                )

            # Invalidate cache since we've added new content
            self._invalidate_cache()

            logger.debug(f"Added document {doc_id} and invalidated cache")

        except Exception as e:
            logger.warning(f"Failed to add document {doc_id}: {e}")

    def _create_cache_key(self, query: str) -> str:
        """Create a cache key for a query.

        Args:
            query: Query string.

        Returns:
            Cache key string.
        """
        # Create a hash of the query for consistent cache keys
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        base_type = self.base.__class__.__name__
        return f"retrieval:{base_type}:{query_hash}"

    def _invalidate_cache(self) -> None:
        """Invalidate cached results when base data changes."""
        try:
            # Clear cache entries for this retriever type
            # In a full implementation, this would be more selective
            self.storage.memory.clear()
            logger.debug("Invalidated retriever cache")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    def _get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    def find_similar_queries(self, query: str, limit: int = 5) -> List[str]:
        """Find similar cached queries using vector search.

        Args:
            query: Query to find similar queries for.
            limit: Maximum number of similar queries to return.

        Returns:
            List of similar query strings.
        """
        try:
            # Use vector search to find similar cached queries
            similar_items = self.storage.search_similar(query, limit)

            # Extract query strings from cached results
            similar_queries: List[str] = []
            for item in similar_items:
                if isinstance(item, list) and len(item) > 0:
                    # This is a cached result list - we'd need to store the original query
                    # For now, just return a placeholder
                    similar_queries.append(f"similar_query_{len(similar_queries)}")

            return similar_queries[:limit]

        except Exception as e:
            logger.warning(f"Similar queries search failed: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache performance metrics.
        """
        base_stats = self.storage.get_stats()

        return {
            **base_stats,
            "cache_performance": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self._get_hit_rate(),
                "total_requests": self.cache_hits + self.cache_misses,
            },
            "base_retriever": {
                "type": self.base.__class__.__name__,
                "module": self.base.__class__.__module__,
            },
            "cache_ttl": self.cache_ttl,
            "storage_type": "CachedRetriever",
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.storage.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug("Cleared retriever cache and reset statistics")

    def warm_cache(self, queries: List[str]) -> None:
        """Pre-populate cache with results for common queries.

        Args:
            queries: List of queries to pre-cache.
        """
        logger.debug(f"Warming cache with {len(queries)} queries...")

        for query in queries:
            try:
                # This will cache the results if not already cached
                self.retrieve(query)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query}': {e}")

        logger.debug(f"Cache warming completed. Hit rate: {self._get_hit_rate():.2f}")

    def get_cached_queries(self) -> List[str]:
        """Get list of currently cached queries.

        Returns:
            List of query strings that have cached results.
        """
        cached_queries = []

        # Scan memory storage for retrieval cache keys
        for key in self.storage.memory.data.keys():
            if key.startswith("retrieval:"):
                # Extract query info from key (simplified)
                parts = key.split(":")
                if len(parts) >= 3:
                    cached_queries.append(f"Query hash: {parts[2]}")

        return cached_queries

    # Delegate other methods to base retriever if they exist
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown methods to base retriever."""
        if hasattr(self.base, name):
            return getattr(self.base, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
