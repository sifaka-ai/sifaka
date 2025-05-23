"""Redis-based retriever for Sifaka.

This module provides a Redis-based retriever that can cache retrieval results
and act as a caching layer for other retrievers. It supports both standalone
operation and as a caching wrapper around other retrievers.

The RedisRetriever can operate in two modes:
1. Standalone: Store and retrieve documents directly from Redis
2. Caching wrapper: Cache results from another retriever for performance

Example:
    ```python
    from sifaka.retrievers.redis import RedisRetriever
    from sifaka.retrievers.base import InMemoryRetriever
    
    # Standalone Redis retriever
    redis_retriever = RedisRetriever()
    redis_retriever.add_document("doc1", "This is a document about AI.")
    
    # As a caching wrapper
    base_retriever = InMemoryRetriever()
    cached_retriever = RedisRetriever(base_retriever=base_retriever)
    ```
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Document, Thought
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class RedisRetriever:
    """Redis-based retriever with caching capabilities.
    
    This retriever can operate in two modes:
    1. Standalone: Store and retrieve documents directly from Redis
    2. Caching wrapper: Cache results from another retriever
    
    Attributes:
        redis_client: Redis client instance.
        base_retriever: Optional base retriever to cache results from.
        cache_ttl: Time-to-live for cached results in seconds.
        key_prefix: Prefix for Redis keys.
        max_results: Maximum number of documents to return.
    """
    
    def __init__(
        self,
        base_retriever: Optional[Retriever] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour default
        key_prefix: str = "sifaka:retriever",
        max_results: int = 10,
        **redis_kwargs: Any,
    ):
        """Initialize the Redis retriever.
        
        Args:
            base_retriever: Optional base retriever to cache results from.
            redis_host: Redis server host (default: localhost).
            redis_port: Redis server port (default: 6379).
            redis_db: Redis database number (default: 0).
            redis_password: Redis password (optional).
            cache_ttl: Cache time-to-live in seconds (default: 3600).
            key_prefix: Prefix for Redis keys (default: "sifaka:retriever").
            max_results: Maximum number of documents to return (default: 10).
            **redis_kwargs: Additional Redis client arguments.
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not available. Please install it with: pip install redis>=5.0.0"
            )
        
        self.base_retriever = base_retriever
        self.cache_ttl = cache_ttl
        self.key_prefix = key_prefix
        self.max_results = max_results
        
        # Initialize Redis client
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                **redis_kwargs,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            raise RetrieverError(f"Failed to connect to Redis: {e}")
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key for a query.
        
        Args:
            query: The query to generate a key for.
            
        Returns:
            A unique cache key for the query.
        """
        # Create a hash of the query for consistent key generation
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"{self.key_prefix}:query:{query_hash}"
    
    def _generate_doc_key(self, doc_id: str) -> str:
        """Generate a key for storing a document.
        
        Args:
            doc_id: The document ID.
            
        Returns:
            A Redis key for the document.
        """
        return f"{self.key_prefix}:doc:{doc_id}"
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Add a document to Redis storage.
        
        Args:
            doc_id: The document ID.
            text: The document text.
            metadata: Optional metadata for the document.
            ttl: Optional time-to-live for the document (uses cache_ttl if None).
        """
        with error_context(
            component="RedisRetriever",
            operation="add document",
            error_class=RetrieverError,
            message_prefix="Failed to add document to Redis",
        ):
            doc_key = self._generate_doc_key(doc_id)
            doc_data = {
                "text": text,
                "metadata": metadata or {},
                "doc_id": doc_id,
            }
            
            # Store document
            self.redis_client.setex(
                doc_key,
                ttl or self.cache_ttl,
                json.dumps(doc_data)
            )
            
            # Add to document index
            index_key = f"{self.key_prefix}:index"
            self.redis_client.sadd(index_key, doc_id)
            
            logger.debug(f"Added document {doc_id} to Redis")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document from Redis storage.
        
        Args:
            doc_id: The document ID.
            
        Returns:
            The document data or None if not found.
        """
        with error_context(
            component="RedisRetriever",
            operation="get document",
            error_class=RetrieverError,
            message_prefix="Failed to get document from Redis",
        ):
            doc_key = self._generate_doc_key(doc_id)
            doc_data = self.redis_client.get(doc_key)
            
            if doc_data:
                return json.loads(doc_data)
            return None
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query to retrieve documents for.
            
        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="RedisRetriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            cache_key = self._generate_cache_key(query)
            
            # Try to get from cache first
            cached_results = self.redis_client.get(cache_key)
            if cached_results:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached_results)[:self.max_results]
            
            logger.debug(f"Cache miss for query: {query[:50]}...")
            
            # If we have a base retriever, use it and cache the results
            if self.base_retriever:
                results = self.base_retriever.retrieve(query)
                
                # Cache the results
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(results)
                )
                
                logger.debug(f"Cached {len(results)} results for query")
                return results[:self.max_results]
            
            # Otherwise, search our own documents
            return self._search_documents(query)
    
    def _search_documents(self, query: str) -> List[str]:
        """Search documents stored in Redis.
        
        Args:
            query: The query to search for.
            
        Returns:
            A list of relevant document texts.
        """
        # Get all document IDs from the index
        index_key = f"{self.key_prefix}:index"
        doc_ids = self.redis_client.smembers(index_key)
        
        if not doc_ids:
            logger.debug("No documents found in Redis index")
            return []
        
        # Simple keyword matching (similar to InMemoryRetriever)
        query_terms = set(query.lower().split())
        results = []
        
        for doc_id in doc_ids:
            doc_data = self.get_document(doc_id)
            if doc_data:
                text = doc_data["text"]
                doc_terms = set(text.lower().split())
                matches = len(query_terms.intersection(doc_terms))
                
                if matches > 0:
                    results.append((text, matches))
        
        # Sort by number of matches (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top results
        top_results = [text for text, _ in results[:self.max_results]]
        logger.debug(f"Found {len(top_results)} matching documents")
        return top_results
    
    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.
        
        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.
            
        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="RedisRetriever",
            operation="retrieval for thought",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents for thought",
        ):
            # Determine the query based on whether this is pre or post-generation
            if is_pre_generation:
                query = thought.prompt
            else:
                # For post-generation, use both the prompt and the generated text
                query = f"{thought.prompt}\n\n{thought.text}"
            
            # Retrieve documents
            document_texts = self.retrieve(query)
            
            # Convert to Document objects
            documents = [
                Document(
                    text=text,
                    metadata={
                        "source": "redis",
                        "query": query,
                        "cached": True,
                    },
                    score=1.0 - (i * 0.1),  # Simple scoring
                )
                for i, text in enumerate(document_texts)
            ]
            
            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cached results.
        
        Args:
            pattern: Optional pattern to match keys (default: all retriever keys).
            
        Returns:
            Number of keys deleted.
        """
        with error_context(
            component="RedisRetriever",
            operation="clear cache",
            error_class=RetrieverError,
            message_prefix="Failed to clear cache",
        ):
            if pattern is None:
                pattern = f"{self.key_prefix}:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys from Redis cache")
                return deleted
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        with error_context(
            component="RedisRetriever",
            operation="get cache stats",
            error_class=RetrieverError,
            message_prefix="Failed to get cache stats",
        ):
            query_keys = self.redis_client.keys(f"{self.key_prefix}:query:*")
            doc_keys = self.redis_client.keys(f"{self.key_prefix}:doc:*")
            
            return {
                "cached_queries": len(query_keys),
                "stored_documents": len(doc_keys),
                "redis_info": self.redis_client.info("memory"),
            }


def create_redis_retriever(
    base_retriever: Optional[Retriever] = None,
    **kwargs: Any,
) -> RedisRetriever:
    """Factory function to create a Redis retriever.
    
    Args:
        base_retriever: Optional base retriever to cache results from.
        **kwargs: Additional arguments for RedisRetriever.
        
    Returns:
        A configured RedisRetriever instance.
    """
    return RedisRetriever(base_retriever=base_retriever, **kwargs)
