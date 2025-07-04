"""Redis storage backend for Sifaka with RediSearch support.

This backend stores Sifaka results and thoughts in Redis with full-text search
capabilities using RediSearch. Falls back to basic search if RediSearch is not available.

## Features:
- Full-text search on content and feedback
- Filter by critic type, confidence scores, dates
- Semantic search support (with Redis Stack)
- Efficient indexing and retrieval
- Backwards compatible with basic Redis

## Requirements:
- Redis with RediSearch module (Redis Stack) for full features
- Falls back to basic search with standard Redis
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import redis.asyncio as redis
    from redis.commands.search.field import TextField, NumericField, TagField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from ..core.models import SifakaResult
from .base import StorageBackend

logger = logging.getLogger(__name__)


class RedisStorage(StorageBackend):
    """Redis storage backend with RediSearch for advanced querying.

    Ideal for:
    - Development and debugging (see thoughts in real-time)
    - Distributed systems (share results across processes)
    - High-performance caching with advanced search
    - Building memory systems for critics

    Example:
        # Basic usage
        storage = RedisStorage()
        result = await improve(text, storage=storage)

        # Search examples
        results = await storage.search("improve clarity")  # Full-text
        results = await storage.search("@critic:{style}")  # By critic
        results = await storage.search("@confidence:[0.8 1.0]")  # High confidence
        results = await storage.search("@critic:{style} improve*")  # Combined

        # Advanced search with filters
        results = await storage.search_advanced(
            query="technical documentation",
            critics=["style", "clarity"],
            min_confidence=0.7,
            limit=20
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "sifaka:",
        ttl: int = 3600,  # 1 hour default TTL
        use_redisearch: bool = True,
        **kwargs,
    ):
        """Initialize Redis storage with optional RediSearch support.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            prefix: Key prefix for all Sifaka data
            ttl: Time-to-live in seconds (0 = no expiration)
            use_redisearch: Try to use RediSearch if available
            **kwargs: Additional Redis connection parameters
        """
        if not HAS_REDIS:
            raise ImportError(
                "Redis support requires 'redis' package. "
                "Install with: pip install redis"
            )

        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.ttl = ttl
        self.use_redisearch = use_redisearch
        self.kwargs = kwargs
        self._client: Optional[redis.Redis] = None
        self._has_redisearch: Optional[bool] = None
        self._index_created = False

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                **self.kwargs,
            )
            # Check for RediSearch support
            if self.use_redisearch:
                await self._check_redisearch_support()
                if self._has_redisearch:
                    await self._ensure_index_exists()
        return self._client

    async def _check_redisearch_support(self) -> None:
        """Check if RediSearch module is available."""
        try:
            modules = await self._client.module_list()
            self._has_redisearch = any(
                module.get(b"name", b"").decode() == "search" for module in modules
            )
            if self._has_redisearch:
                logger.info("RediSearch module detected - enabling advanced search")
            else:
                logger.info("RediSearch not available - using basic search")
        except Exception:
            # MODULE LIST command not available or failed
            self._has_redisearch = False
            logger.info("Could not detect RediSearch - using basic search")

    async def _ensure_index_exists(self) -> None:
        """Create RediSearch index if it doesn't exist."""
        if self._index_created:
            return

        index_name = f"{self.prefix}idx:results"

        try:
            # Check if index exists
            await self._client.ft(index_name).info()
            self._index_created = True
        except Exception:
            # Index doesn't exist, create it
            try:
                await self._create_search_index(index_name)
                self._index_created = True
                logger.info(f"Created RediSearch index: {index_name}")
            except Exception as e:
                logger.warning(f"Failed to create RediSearch index: {e}")
                self._has_redisearch = False

    async def _create_search_index(self, index_name: str) -> None:
        """Create RediSearch index for Sifaka results."""
        # Define index schema
        schema = [
            TextField("$.original_text", as_name="original_text"),
            TextField("$.final_text", as_name="final_text"),
            TextField("$.feedback", as_name="feedback"),  # Combined feedback field
            TagField("$.critics[*]", as_name="critic"),  # All critics used
            NumericField(
                "$.max_confidence", as_name="confidence"
            ),  # Highest confidence
            NumericField("$.iteration_count", as_name="iterations"),
            NumericField("$.timestamp", as_name="timestamp"),  # For date filtering
            TagField("$.result_id", as_name="result_id"),
        ]

        # Create index on JSON documents
        definition = IndexDefinition(
            prefix=[f"{self.prefix}result:"], index_type=IndexType.JSON
        )

        await self._client.ft(index_name).create_index(schema, definition=definition)

    def _make_key(self, key: str) -> str:
        """Create namespaced Redis key."""
        return f"{self.prefix}{key}"

    async def save(self, result: SifakaResult) -> str:
        """Save result to Redis with search indexing.

        Returns result_id for retrieval.
        """
        import uuid

        client = await self._get_client()
        result_id = str(uuid.uuid4())
        key = self._make_key(f"result:{result_id}")

        # Prepare data for storage and indexing
        data = result.model_dump(mode="json")

        # Add fields for better search
        search_data = {
            **data,
            "result_id": result_id,
            "critics": list(set(crit.critic for crit in result.critiques)),
            "feedback": " ".join(crit.feedback for crit in result.critiques),
            "max_confidence": max(
                (crit.confidence for crit in result.critiques), default=0.0
            ),
            "iteration_count": len(result.generations),
            "timestamp": datetime.utcnow().timestamp(),
        }

        # Store as JSON (RediSearch will index if available)
        if self.ttl > 0:
            await client.json().set(key, "$", search_data)
            await client.expire(key, self.ttl)
        else:
            await client.json().set(key, "$", search_data)

        # Also store in a list for easy retrieval
        list_key = self._make_key("results:list")
        await client.lpush(list_key, result_id)
        await client.ltrim(list_key, 0, 99)  # Keep last 100

        # Store thoughts separately for easy access
        await self._save_thoughts(result_id, result)

        return result_id

    async def _save_thoughts(self, result_id: str, result: SifakaResult) -> None:
        """Save thoughts as a simple list."""
        client = await self._get_client()

        # Save thoughts as a JSON list
        thoughts_key = self._make_key(f"thoughts:{result_id}")
        thoughts_data = []

        for i, critique in enumerate(result.critiques):
            thought = {
                "iteration": i + 1,
                "critic": critique.critic,
                "feedback": critique.feedback,
                "confidence": critique.confidence,
                "needs_improvement": critique.needs_improvement,
                "timestamp": (
                    critique.timestamp.isoformat() if critique.timestamp else None
                ),
            }

            # Add tool usage if present
            if hasattr(critique, "tools_used") and critique.tools_used:
                thought["tools_used"] = [
                    {
                        "tool_name": tool.tool_name,
                        "status": tool.status,
                        "result_count": tool.result_count,
                    }
                    for tool in critique.tools_used
                ]

            thoughts_data.append(thought)

        # Store as JSON
        if self.ttl > 0:
            await client.setex(thoughts_key, self.ttl, json.dumps(thoughts_data))
        else:
            await client.set(thoughts_key, json.dumps(thoughts_data))

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load result from Redis."""
        client = await self._get_client()
        key = self._make_key(f"result:{result_id}")

        try:
            # Try JSON.GET first (for RediSearch compatibility)
            data = await client.json().get(key)
            if not data:
                # Fallback to regular GET for backwards compatibility
                data = await client.get(key)
                if data:
                    data = json.loads(data)
        except Exception:
            # Fallback to regular GET
            data = await client.get(key)
            if data:
                data = json.loads(data)

        if not data:
            return None

        # Remove extra fields added for search
        if isinstance(data, dict):
            data.pop("result_id", None)
            data.pop("critics", None)
            data.pop("feedback", None)
            data.pop("max_confidence", None)
            data.pop("iteration_count", None)
            data.pop("timestamp", None)

        return SifakaResult.model_validate(data)

    async def list(self, limit: int = 10, offset: int = 0) -> List[SifakaResult]:
        """List recent results."""
        client = await self._get_client()
        list_key = self._make_key("results:list")

        # Get result IDs
        result_ids = await client.lrange(list_key, offset, offset + limit - 1)

        # Load results
        results = []
        for result_id in result_ids:
            result = await self.load(result_id)
            if result:
                results.append(result)

        return results

    async def delete(self, result_id: str) -> None:
        """Delete a result from Redis."""
        client = await self._get_client()

        # Delete result
        key = self._make_key(f"result:{result_id}")
        await client.delete(key)

        # Delete thoughts
        thoughts_key = self._make_key(f"thoughts:{result_id}")
        await client.delete(thoughts_key)

        # Remove from list
        list_key = self._make_key("results:list")
        await client.lrem(list_key, 0, result_id)

    async def search(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> List[SifakaResult]:
        """Search results using RediSearch or fallback to basic search.

        With RediSearch, supports:
        - Full-text search: "improve clarity"
        - Critic filter: "@critic:{style}"
        - Confidence filter: "@confidence:[0.8 1.0]"
        - Combined: "@critic:{style} improve* @confidence:[0.7 1.0]"
        - Date filter: "@timestamp:[1704067200 +inf]"  # After Jan 1, 2024

        Without RediSearch:
        - Basic text matching in original and final text
        """
        if self._has_redisearch:
            try:
                return await self._search_with_redisearch(query, limit, offset)
            except Exception as e:
                logger.warning(f"RediSearch query failed, falling back: {e}")

        # Fallback to basic search
        return await self._basic_search(query, limit)

    async def _search_with_redisearch(
        self, query: str, limit: int, offset: int
    ) -> List[SifakaResult]:
        """Search using RediSearch."""
        index_name = f"{self.prefix}idx:results"

        # Create query
        q = Query(query).paging(offset, limit).sort_by("timestamp", asc=False)

        # Execute search
        results = await self._client.ft(index_name).search(q)

        # Load full results
        sifaka_results = []
        for doc in results.docs:
            # Extract result_id from document
            result_id = doc.result_id
            result = await self.load(result_id)
            if result:
                sifaka_results.append(result)

        return sifaka_results

    async def _basic_search(self, query: str, limit: int) -> List[SifakaResult]:
        """Fallback basic text search."""
        results = await self.list(limit=100)  # Get recent results

        # Simple text search
        matches = []
        query_lower = query.lower()

        for result in results:
            if (
                query_lower in result.original_text.lower()
                or query_lower in result.final_text.lower()
                or any(
                    query_lower in crit.feedback.lower() for crit in result.critiques
                )
            ):
                matches.append(result)
                if len(matches) >= limit:
                    break

        return matches

    async def search_advanced(
        self,
        query: Optional[str] = None,
        critics: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SifakaResult]:
        """Advanced search with multiple filters.

        Args:
            query: Full-text search query
            critics: Filter by critic types used
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            start_date: Results after this date
            end_date: Results before this date
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of matching SifakaResult objects
        """
        if not self._has_redisearch:
            # Fallback to basic search with query only
            return await self._basic_search(query or "*", limit)

        # Build RediSearch query
        query_parts = []

        if query:
            query_parts.append(query)
        else:
            query_parts.append("*")  # Match all

        if critics:
            critic_filter = " | ".join(f"@critic:{{{c}}}" for c in critics)
            query_parts.append(f"({critic_filter})")

        if min_confidence is not None or max_confidence is not None:
            min_conf = min_confidence or 0.0
            max_conf = max_confidence or 1.0
            query_parts.append(f"@confidence:[{min_conf} {max_conf}]")

        if start_date or end_date:
            start_ts = start_date.timestamp() if start_date else 0
            end_ts = end_date.timestamp() if end_date else "+inf"
            query_parts.append(f"@timestamp:[{start_ts} {end_ts}]")

        full_query = " ".join(query_parts)
        return await self._search_with_redisearch(full_query, limit, offset)

    async def get_thoughts(self, result_id: str) -> List[Dict[str, Any]]:
        """Get thoughts for a result.

        Returns list of thought entries with all details.
        """
        client = await self._get_client()
        thoughts_key = self._make_key(f"thoughts:{result_id}")

        # Get thoughts JSON
        data = await client.get(thoughts_key)
        if not data:
            return []

        return json.loads(data)

    async def cleanup(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search index statistics if RediSearch is available."""
        if not self._has_redisearch:
            return {"available": False, "message": "RediSearch not available"}

        try:
            index_name = f"{self.prefix}idx:results"
            info = await self._client.ft(index_name).info()

            return {
                "available": True,
                "index_name": index_name,
                "num_docs": info.get("num_docs", 0),
                "num_terms": info.get("num_terms", 0),
                "index_size_mb": info.get("inverted_sz_mb", 0),
                "indexing": info.get("indexing", False),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
