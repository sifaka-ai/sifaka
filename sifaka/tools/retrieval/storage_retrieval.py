"""Storage-specific retrieval tools for Sifaka.

This module provides retrieval tools that work with specific storage backends
like Redis, PostgreSQL, and other persistence implementations.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING

from pydantic_ai.tools import Tool
from sifaka.tools.base import BaseSifakaTool, ToolConfigurationError, register_tool
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from sifaka.storage.base import SifakaBasePersistence

logger = get_logger(__name__)


@register_tool("redis_retrieval")
class RedisRetrievalTool(BaseSifakaTool):
    """Redis-specific retrieval tool."""
    
    def __init__(self, persistence: "SifakaBasePersistence", **kwargs):
        super().__init__(
            name="redis_retrieval",
            description="Retrieve data from Redis storage backend",
            category="retrieval",
            provider="redis",
            requires_auth=False,
            **kwargs
        )
        self.persistence = persistence
    
    def validate_configuration(self) -> None:
        """Validate Redis retrieval configuration."""
        from sifaka.storage.redis import RedisPersistence
        
        if not isinstance(self.persistence, RedisPersistence):
            raise ToolConfigurationError(
                f"RedisRetrievalTool requires RedisPersistence, got {type(self.persistence)}"
            )
    
    async def search_redis_keys(self, pattern: str, limit: int = 50) -> List[str]:
        """Search Redis keys by pattern.
        
        Args:
            pattern: Redis key pattern (supports wildcards)
            limit: Maximum number of keys to return
            
        Returns:
            List of matching Redis keys
        """
        try:
            keys = await self.persistence._list_keys(pattern)
            return keys[:limit]
        except Exception as e:
            logger.error(f"Redis key search failed: {e}")
            return []
    
    async def get_redis_value(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis by key.
        
        Args:
            key: Redis key to retrieve
            
        Returns:
            Value if found, None otherwise
        """
        try:
            value = await self.persistence._get_raw(key)
            if value:
                return {"key": key, "value": value}
            return None
        except Exception as e:
            logger.error(f"Redis value retrieval failed for key '{key}': {e}")
            return None
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create Redis retrieval tools."""
        return [
            Tool(self.search_redis_keys, takes_ctx=False),
            Tool(self.get_redis_value, takes_ctx=False),
        ]


@register_tool("postgres_retrieval")
class PostgreSQLRetrievalTool(BaseSifakaTool):
    """PostgreSQL-specific retrieval tool."""
    
    def __init__(self, persistence: "SifakaBasePersistence", **kwargs):
        super().__init__(
            name="postgres_retrieval",
            description="Retrieve data from PostgreSQL storage backend",
            category="retrieval",
            provider="postgresql",
            requires_auth=True,
            **kwargs
        )
        self.persistence = persistence
    
    def validate_configuration(self) -> None:
        """Validate PostgreSQL retrieval configuration."""
        from sifaka.storage.postgresql import PostgreSQLPersistence
        
        if not isinstance(self.persistence, PostgreSQLPersistence):
            raise ToolConfigurationError(
                f"PostgreSQLRetrievalTool requires PostgreSQLPersistence, got {type(self.persistence)}"
            )
    
    async def search_thoughts_sql(
        self, 
        query: str, 
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search thoughts using SQL-like queries.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Additional filters to apply
            
        Returns:
            List of matching thoughts
        """
        try:
            # This would use the PostgreSQL persistence's search capabilities
            thoughts = await self.persistence.list_thoughts(limit=limit)
            
            # Apply text search filter
            filtered_thoughts = []
            for thought in thoughts:
                if query.lower() in thought.final_text.lower():
                    filtered_thoughts.append(thought.model_dump())
                    if len(filtered_thoughts) >= limit:
                        break
            
            return filtered_thoughts
        except Exception as e:
            logger.error(f"PostgreSQL search failed: {e}")
            return []
    
    async def get_thought_analytics(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics data for a specific thought.
        
        Args:
            thought_id: ID of the thought
            
        Returns:
            Analytics data if found
        """
        try:
            thought = await self.persistence.retrieve_thought(thought_id)
            if thought:
                return {
                    "thought_id": thought_id,
                    "iterations": thought.iteration,
                    "validation_passed": thought.validation_passed(),
                    "critic_count": len(thought.critique_results),
                    "tool_calls": len(thought.tool_calls),
                    "created_at": thought.created_at.isoformat() if thought.created_at else None,
                }
            return None
        except Exception as e:
            logger.error(f"Analytics retrieval failed for thought '{thought_id}': {e}")
            return None
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create PostgreSQL retrieval tools."""
        return [
            Tool(self.search_thoughts_sql, takes_ctx=False),
            Tool(self.get_thought_analytics, takes_ctx=False),
        ]


@register_tool("storage_retrieval")
class GenericStorageRetrievalTool(BaseSifakaTool):
    """Generic storage retrieval tool that works with any persistence backend."""
    
    def __init__(self, persistence: "SifakaBasePersistence", **kwargs):
        super().__init__(
            name="storage_retrieval",
            description="Retrieve data from any Sifaka storage backend",
            category="retrieval",
            provider="generic",
            **kwargs
        )
        self.persistence = persistence
    
    async def search_thoughts_generic(
        self, 
        query: str, 
        limit: int = 50,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search thoughts using generic storage interface.
        
        Args:
            query: Search query
            limit: Maximum number of results
            conversation_id: Optional conversation filter
            
        Returns:
            List of matching thoughts
        """
        try:
            thoughts = await self.persistence.list_thoughts(
                conversation_id=conversation_id,
                limit=limit
            )
            
            # Simple text search
            results = []
            for thought in thoughts:
                if query.lower() in thought.final_text.lower():
                    results.append(thought.model_dump())
            
            return results
        except Exception as e:
            logger.error(f"Generic storage search failed: {e}")
            return []
    
    async def get_thought_by_id(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific thought by ID.
        
        Args:
            thought_id: ID of the thought to retrieve
            
        Returns:
            Thought data if found
        """
        try:
            thought = await self.persistence.retrieve_thought(thought_id)
            return thought.model_dump() if thought else None
        except Exception as e:
            logger.error(f"Thought retrieval failed for ID '{thought_id}': {e}")
            return None
    
    def create_pydantic_tools(self) -> List[Tool]:
        """Create generic storage retrieval tools."""
        return [
            Tool(self.search_thoughts_generic, takes_ctx=False),
            Tool(self.get_thought_by_id, takes_ctx=False),
        ]


def create_storage_retrieval_tools(
    persistence: "SifakaBasePersistence",
    backend_specific: bool = True,
    **kwargs
) -> List[Tool]:
    """Create storage retrieval tools for the given persistence backend.
    
    Args:
        persistence: Storage persistence backend
        backend_specific: Whether to use backend-specific tools when available
        **kwargs: Additional configuration options
        
    Returns:
        List of PydanticAI Tool instances
        
    Example:
        ```python
        from sifaka.storage import RedisPersistence
        
        redis_storage = RedisPersistence(...)
        tools = create_storage_retrieval_tools(redis_storage)
        ```
    """
    if backend_specific:
        # Try to create backend-specific tools
        from sifaka.storage.redis import RedisPersistence
        from sifaka.storage.postgresql import PostgreSQLPersistence
        
        if isinstance(persistence, RedisPersistence):
            tool = RedisRetrievalTool(persistence, **kwargs)
            return tool.create_pydantic_tools()
        elif isinstance(persistence, PostgreSQLPersistence):
            tool = PostgreSQLRetrievalTool(persistence, **kwargs)
            return tool.create_pydantic_tools()
    
    # Fall back to generic tools
    tool = GenericStorageRetrievalTool(persistence, **kwargs)
    return tool.create_pydantic_tools()
