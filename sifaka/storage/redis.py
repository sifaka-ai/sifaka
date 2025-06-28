"""Redis storage backend for Sifaka - fast in-memory storage for thoughts.

This backend stores Sifaka results and thoughts in Redis for quick access
and retrieval. Thoughts are stored as JSON for simplicity.
"""

import json
from typing import Optional, List, Dict, Any

try:
    import redis.asyncio as redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from ..core.models import SifakaResult
from .base import StorageBackend


class RedisStorage(StorageBackend):
    """Redis storage backend for fast access to thoughts and results.

    Ideal for:
    - Development and debugging (see thoughts in real-time)
    - Distributed systems (share results across processes)
    - High-performance caching

    Example:
        # Basic usage
        storage = RedisStorage()
        result = await improve(text, storage=storage)

        # Custom Redis connection
        storage = RedisStorage(
            host="localhost",
            port=6379,
            prefix="sifaka:dev:"
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "sifaka:",
        ttl: int = 3600,  # 1 hour default TTL
        **kwargs,
    ):
        """Initialize Redis storage.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            prefix: Key prefix for all Sifaka data
            ttl: Time-to-live in seconds (0 = no expiration)
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
        self.kwargs = kwargs
        self._client: Optional[redis.Redis] = None

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
        return self._client

    def _make_key(self, key: str) -> str:
        """Create namespaced Redis key."""
        return f"{self.prefix}{key}"

    async def save(self, result: SifakaResult) -> str:
        """Save result to Redis.

        Returns result_id for retrieval.
        """
        import uuid

        client = await self._get_client()
        result_id = str(uuid.uuid4())
        key = self._make_key(f"result:{result_id}")

        # Serialize result
        data = result.model_dump(mode="json")

        # Store with optional TTL
        if self.ttl > 0:
            await client.setex(key, self.ttl, json.dumps(data))
        else:
            await client.set(key, json.dumps(data))

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

        data = await client.get(key)
        if not data:
            return None

        return SifakaResult.model_validate(json.loads(data))

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

    async def search(self, query: str, limit: int = 10) -> List[SifakaResult]:
        """Search results by text content.

        Note: This is a simple implementation. For production,
        consider using Redis Search module.
        """
        results = await self.list(limit=100)  # Get recent results

        # Simple text search
        matches = []
        query_lower = query.lower()

        for result in results:
            if (
                query_lower in result.original_text.lower()
                or query_lower in result.final_text.lower()
            ):
                matches.append(result)
                if len(matches) >= limit:
                    break

        return matches

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
