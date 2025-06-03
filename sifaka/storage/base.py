"""Base persistence implementation for Sifaka using PydanticAI.

This module provides the foundation for all Sifaka storage implementations
by extending PydanticAI's BaseStatePersistence with Sifaka-specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic_graph.persistence import BaseStatePersistence

from sifaka.core.thought import SifakaThought
from sifaka.utils import get_logger

logger = get_logger(__name__)


class SifakaBasePersistence(BaseStatePersistence[SifakaThought, str], ABC):
    """Base persistence class for all Sifaka storage implementations.
    
    This class extends PydanticAI's BaseStatePersistence to provide:
    - SifakaThought-specific serialization and deserialization
    - Thought indexing and search capabilities
    - Conversation and parent-child relationship management
    - Common utilities for all storage backends
    
    All Sifaka storage implementations should inherit from this class.
    """
    
    def __init__(self, key_prefix: str = "sifaka"):
        """Initialize base persistence.
        
        Args:
            key_prefix: Prefix for all storage keys
        """
        self.key_prefix = key_prefix
        logger.debug(f"Initialized {self.__class__.__name__} with prefix '{key_prefix}'")
    
    def _make_key(self, thought_id: str) -> str:
        """Create a prefixed storage key for a thought.
        
        Args:
            thought_id: The thought ID
            
        Returns:
            Prefixed storage key
        """
        return f"{self.key_prefix}:thought:{thought_id}"
    
    def _make_conversation_key(self, conversation_id: str) -> str:
        """Create a prefixed storage key for conversation indexing.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Prefixed conversation key
        """
        return f"{self.key_prefix}:conversation:{conversation_id}"
    
    def _make_index_key(self, index_type: str) -> str:
        """Create a prefixed storage key for indexing.
        
        Args:
            index_type: Type of index (e.g., 'all_thoughts', 'by_date')
            
        Returns:
            Prefixed index key
        """
        return f"{self.key_prefix}:index:{index_type}"
    
    async def serialize_state(self, state: SifakaThought) -> str:
        """Serialize a SifakaThought to string format.
        
        Args:
            state: The thought to serialize
            
        Returns:
            Serialized thought as JSON string
        """
        try:
            return state.model_dump_json()
        except Exception as e:
            logger.error(f"Failed to serialize thought {state.id}: {e}")
            raise
    
    async def deserialize_state(self, data: str) -> SifakaThought:
        """Deserialize a string to SifakaThought.
        
        Args:
            data: Serialized thought data
            
        Returns:
            Deserialized SifakaThought
        """
        try:
            import json
            thought_dict = json.loads(data)
            return SifakaThought.model_validate(thought_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize thought data: {e}")
            raise
    
    # Abstract methods for storage backends to implement
    @abstractmethod
    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data at the given key.
        
        Args:
            key: Storage key
            data: Raw data to store
        """
        pass
    
    @abstractmethod
    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from the given key.
        
        Args:
            key: Storage key
            
        Returns:
            Raw data if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def _delete_raw(self, key: str) -> bool:
        """Delete data at the given key.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if key didn't exist
        """
        pass
    
    @abstractmethod
    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the given pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            List of matching keys
        """
        pass
    
    # Sifaka-specific storage methods
    async def store_thought(self, thought: SifakaThought) -> None:
        """Store a thought with indexing.
        
        Args:
            thought: The thought to store
        """
        try:
            # Serialize and store the thought
            key = self._make_key(thought.id)
            data = await self.serialize_state(thought)
            await self._store_raw(key, data)
            
            # Update indexes
            await self._update_indexes(thought)
            
            logger.debug(f"Stored thought {thought.id}")
            
        except Exception as e:
            logger.error(f"Failed to store thought {thought.id}: {e}")
            raise
    
    async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a thought by ID.
        
        Args:
            thought_id: The thought ID to retrieve
            
        Returns:
            The thought if found, None otherwise
        """
        try:
            key = self._make_key(thought_id)
            data = await self._retrieve_raw(key)
            
            if data is None:
                return None
                
            thought = await self.deserialize_state(data)
            logger.debug(f"Retrieved thought {thought_id}")
            return thought
            
        except Exception as e:
            logger.error(f"Failed to retrieve thought {thought_id}: {e}")
            return None
    
    async def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought by ID.
        
        Args:
            thought_id: The thought ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            key = self._make_key(thought_id)
            deleted = await self._delete_raw(key)
            
            if deleted:
                # TODO: Update indexes to remove the thought
                logger.debug(f"Deleted thought {thought_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete thought {thought_id}: {e}")
            return False
    
    async def list_thoughts(
        self, 
        conversation_id: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[SifakaThought]:
        """List thoughts, optionally filtered by conversation.
        
        Args:
            conversation_id: Filter by conversation ID
            limit: Maximum number of thoughts to return
            
        Returns:
            List of thoughts
        """
        try:
            # Get all thought keys
            pattern = f"{self.key_prefix}:thought:*"
            keys = await self._list_keys(pattern)
            
            thoughts = []
            for key in keys:
                data = await self._retrieve_raw(key)
                if data:
                    try:
                        thought = await self.deserialize_state(data)
                        
                        # Filter by conversation if specified
                        if conversation_id and thought.conversation_id != conversation_id:
                            continue
                            
                        thoughts.append(thought)
                        
                        # Apply limit if specified
                        if limit and len(thoughts) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to deserialize thought from key {key}: {e}")
                        continue
            
            # Sort by creation time (newest first)
            thoughts.sort(key=lambda t: t.created_at, reverse=True)
            
            logger.debug(f"Listed {len(thoughts)} thoughts")
            return thoughts
            
        except Exception as e:
            logger.error(f"Failed to list thoughts: {e}")
            return []
    
    async def _update_indexes(self, thought: SifakaThought) -> None:
        """Update indexes for the given thought.
        
        This is a placeholder for index management. Subclasses can override
        to implement specific indexing strategies.
        
        Args:
            thought: The thought to index
        """
        # Base implementation does nothing
        # Subclasses can override for specific indexing needs
        pass
