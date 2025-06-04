"""PostgreSQL persistence implementation for Sifaka.

This module provides PostgreSQL-based storage for production-ready persistence
with ACID transactions, complex queries, and scalability.

Note: This is a placeholder implementation. To use PostgreSQL storage:
1. Install: uv add asyncpg
2. Set up PostgreSQL database
3. Run database migrations
"""

import json
from typing import Any, Dict, List, Optional

from .base import SifakaBasePersistence
from sifaka.utils import get_logger

logger = get_logger(__name__)


class PostgreSQLPersistence(SifakaBasePersistence):
    """PostgreSQL-based persistence for production use.
    
    This implementation provides:
    - ACID transactions for data consistency
    - Complex SQL queries for advanced filtering
    - Full-text search capabilities
    - Scalable storage for large datasets
    - Connection pooling for performance
    
    Database Schema:
    ```sql
    CREATE TABLE sifaka_thoughts (
        id UUID PRIMARY KEY,
        key_prefix VARCHAR(100) NOT NULL,
        thought_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        conversation_id UUID,
        parent_thought_id UUID,
        final_text TEXT,
        prompt TEXT NOT NULL,
        INDEX idx_conversation_id (conversation_id),
        INDEX idx_parent_thought_id (parent_thought_id),
        INDEX idx_created_at (created_at),
        INDEX gin_thought_data (thought_data),
        INDEX gin_final_text (final_text gin_trgm_ops)
    );
    
    CREATE TABLE sifaka_snapshots (
        id UUID PRIMARY KEY,
        thought_id UUID NOT NULL,
        node_name VARCHAR(100) NOT NULL,
        snapshot_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(thought_id, node_name)
    );
    ```
    
    Prerequisites:
    - PostgreSQL 12+ with JSONB support
    - pg_trgm extension for full-text search
    - asyncpg Python library
    """
    
    def __init__(
        self,
        connection_string: str,
        key_prefix: str = "sifaka",
        pool_size: int = 10,
        max_pool_size: int = 20
    ):
        """Initialize PostgreSQL persistence.
        
        Args:
            connection_string: PostgreSQL connection string
            key_prefix: Prefix for storage keys
            pool_size: Initial connection pool size
            max_pool_size: Maximum connection pool size
        """
        super().__init__(key_prefix)
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.pool = None
        
        logger.debug(f"Initialized PostgreSQLPersistence with prefix '{key_prefix}'")
    
    async def _ensure_pool(self):
        """Ensure database connection pool is initialized."""
        if self.pool is None:
            try:
                import asyncpg
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=self.pool_size,
                    max_size=self.max_pool_size
                )
                logger.debug("PostgreSQL connection pool created")
            except ImportError:
                raise ImportError(
                    "asyncpg is required for PostgreSQL persistence. "
                    "Install with: uv add asyncpg"
                )
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                raise
    
    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data in PostgreSQL."""
        await self._ensure_pool()
        
        try:
            # Parse the key to extract thought ID
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Regular thought storage
                thought_id = key_parts[2]
                await self._store_thought_data(thought_id, data)
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Snapshot storage
                thought_id = key_parts[2]
                node_name = key_parts[3]
                await self._store_snapshot_data(thought_id, node_name, data)
            else:
                # Generic storage (not implemented for PostgreSQL)
                logger.warning(f"Generic storage not supported for PostgreSQL: {key}")
                
        except Exception as e:
            logger.error(f"Failed to store PostgreSQL data for key {key}: {e}")
            raise
    
    async def _store_thought_data(self, thought_id: str, data: str) -> None:
        """Store thought data in the thoughts table."""
        async with self.pool.acquire() as conn:
            # Parse thought data to extract searchable fields
            thought_data = json.loads(data)
            
            query = """
                INSERT INTO sifaka_thoughts (
                    id, key_prefix, thought_data, conversation_id, 
                    parent_thought_id, final_text, prompt, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    thought_data = EXCLUDED.thought_data,
                    conversation_id = EXCLUDED.conversation_id,
                    parent_thought_id = EXCLUDED.parent_thought_id,
                    final_text = EXCLUDED.final_text,
                    prompt = EXCLUDED.prompt,
                    updated_at = NOW()
            """
            
            await conn.execute(
                query,
                thought_id,
                self.key_prefix,
                thought_data,
                thought_data.get("conversation_id"),
                thought_data.get("parent_thought_id"),
                thought_data.get("final_text"),
                thought_data.get("prompt", "")
            )
    
    async def _store_snapshot_data(self, thought_id: str, node_name: str, data: str) -> None:
        """Store snapshot data in the snapshots table."""
        async with self.pool.acquire() as conn:
            snapshot_data = json.loads(data)
            
            query = """
                INSERT INTO sifaka_snapshots (id, thought_id, node_name, snapshot_data)
                VALUES (gen_random_uuid(), $1, $2, $3)
                ON CONFLICT (thought_id, node_name) DO UPDATE SET
                    snapshot_data = EXCLUDED.snapshot_data,
                    created_at = NOW()
            """
            
            await conn.execute(query, thought_id, node_name, snapshot_data)
    
    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from PostgreSQL."""
        await self._ensure_pool()
        
        try:
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Regular thought retrieval
                thought_id = key_parts[2]
                return await self._retrieve_thought_data(thought_id)
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Snapshot retrieval
                thought_id = key_parts[2]
                node_name = key_parts[3]
                return await self._retrieve_snapshot_data(thought_id, node_name)
            else:
                logger.warning(f"Generic retrieval not supported for PostgreSQL: {key}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve PostgreSQL data for key {key}: {e}")
            return None
    
    async def _retrieve_thought_data(self, thought_id: str) -> Optional[str]:
        """Retrieve thought data from the thoughts table."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT thought_data FROM sifaka_thoughts 
                WHERE id = $1 AND key_prefix = $2
            """
            
            row = await conn.fetchrow(query, thought_id, self.key_prefix)
            if row:
                return json.dumps(row["thought_data"])
            return None
    
    async def _retrieve_snapshot_data(self, thought_id: str, node_name: str) -> Optional[str]:
        """Retrieve snapshot data from the snapshots table."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT snapshot_data FROM sifaka_snapshots 
                WHERE thought_id = $1 AND node_name = $2
            """
            
            row = await conn.fetchrow(query, thought_id, node_name)
            if row:
                return json.dumps(row["snapshot_data"])
            return None
    
    async def _delete_raw(self, key: str) -> bool:
        """Delete data from PostgreSQL."""
        await self._ensure_pool()
        
        try:
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Delete thought
                thought_id = key_parts[2]
                async with self.pool.acquire() as conn:
                    result = await conn.execute(
                        "DELETE FROM sifaka_thoughts WHERE id = $1 AND key_prefix = $2",
                        thought_id, self.key_prefix
                    )
                    return result != "DELETE 0"
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Delete snapshot
                thought_id = key_parts[2]
                node_name = key_parts[3]
                async with self.pool.acquire() as conn:
                    result = await conn.execute(
                        "DELETE FROM sifaka_snapshots WHERE thought_id = $1 AND node_name = $2",
                        thought_id, node_name
                    )
                    return result != "DELETE 0"
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete PostgreSQL data for key {key}: {e}")
            return False
    
    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the pattern from PostgreSQL."""
        await self._ensure_pool()
        
        try:
            keys = []
            
            async with self.pool.acquire() as conn:
                # List thought keys
                if f"{self.key_prefix}:thought:" in pattern:
                    query = "SELECT id FROM sifaka_thoughts WHERE key_prefix = $1"
                    rows = await conn.fetch(query, self.key_prefix)
                    for row in rows:
                        keys.append(f"{self.key_prefix}:thought:{row['id']}")
                
                # List snapshot keys
                if f"{self.key_prefix}:snapshot:" in pattern:
                    query = """
                        SELECT thought_id, node_name FROM sifaka_snapshots s
                        JOIN sifaka_thoughts t ON s.thought_id = t.id
                        WHERE t.key_prefix = $1
                    """
                    rows = await conn.fetch(query, self.key_prefix)
                    for row in rows:
                        keys.append(f"{self.key_prefix}:snapshot:{row['thought_id']}:{row['node_name']}")
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list PostgreSQL keys: {e}")
            return []
    
    async def search_thoughts_by_text(self, query: str, limit: int = 10) -> List[str]:
        """Search thoughts by text content using PostgreSQL full-text search."""
        await self._ensure_pool()
        
        try:
            async with self.pool.acquire() as conn:
                sql_query = """
                    SELECT id FROM sifaka_thoughts 
                    WHERE key_prefix = $1 
                    AND (final_text ILIKE $2 OR prompt ILIKE $2)
                    ORDER BY created_at DESC
                    LIMIT $3
                """
                
                rows = await conn.fetch(sql_query, self.key_prefix, f"%{query}%", limit)
                return [f"{self.key_prefix}:thought:{row['id']}" for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search PostgreSQL thoughts: {e}")
            return []
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.debug("PostgreSQL connection pool closed")
