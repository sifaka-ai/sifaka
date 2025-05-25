"""Checkpoint storage for Sifaka chains.

This module provides checkpoint storage functionality for chain execution recovery.
Checkpoints capture the state of chain execution at key points to enable recovery
from failures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from sifaka.core.thought import Thought
from sifaka.storage.protocol import Storage
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ChainCheckpoint(BaseModel):
    """Represents a checkpoint in chain execution.
    
    Captures the complete state of a chain at a specific point in execution,
    allowing for recovery and resumption from that point.
    """
    
    checkpoint_id: str = ""
    chain_id: str
    timestamp: datetime
    current_step: str
    iteration: int
    thought: Thought
    performance_data: Dict[str, Any] = {}
    recovery_point: str
    completed_validators: List[str] = []
    completed_critics: List[str] = []
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **data: Any):
        """Initialize checkpoint with auto-generated ID if not provided."""
        if "checkpoint_id" not in data or not data["checkpoint_id"]:
            data["checkpoint_id"] = str(uuid4())
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class CachedCheckpointStorage:
    """Cached storage for chain checkpoints.
    
    Provides efficient storage and retrieval of chain checkpoints with
    caching support for improved performance.
    """
    
    def __init__(self, storage: Storage):
        """Initialize checkpoint storage.
        
        Args:
            storage: The underlying storage backend.
        """
        self.storage = storage
        logger.debug("Initialized CachedCheckpointStorage")
    
    def save_checkpoint(self, checkpoint: ChainCheckpoint) -> None:
        """Save a checkpoint to storage.
        
        Args:
            checkpoint: The checkpoint to save.
        """
        try:
            key = f"checkpoint:{checkpoint.chain_id}:{checkpoint.checkpoint_id}"
            self.storage.set(key, checkpoint.model_dump())
            
            # Also save as latest checkpoint for the chain
            latest_key = f"checkpoint:latest:{checkpoint.chain_id}"
            self.storage.set(latest_key, checkpoint.model_dump())
            
            logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} for chain {checkpoint.chain_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[ChainCheckpoint]:
        """Get a specific checkpoint by ID.
        
        Args:
            checkpoint_id: The checkpoint ID to retrieve.
            
        Returns:
            The checkpoint if found, None otherwise.
        """
        try:
            # Try to find checkpoint by searching all chain checkpoints
            # This is a simplified implementation - in production you'd want better indexing
            for chain_id in self._get_all_chain_ids():
                key = f"checkpoint:{chain_id}:{checkpoint_id}"
                data = self.storage.get(key)
                if data:
                    return ChainCheckpoint(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_latest_checkpoint(self, chain_id: str) -> Optional[ChainCheckpoint]:
        """Get the latest checkpoint for a chain.
        
        Args:
            chain_id: The chain ID to get the latest checkpoint for.
            
        Returns:
            The latest checkpoint if found, None otherwise.
        """
        try:
            key = f"checkpoint:latest:{chain_id}"
            data = self.storage.get(key)
            if data:
                return ChainCheckpoint(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint for chain {chain_id}: {e}")
            return None
    
    def get_chain_checkpoints(self, chain_id: str) -> List[ChainCheckpoint]:
        """Get all checkpoints for a specific chain.
        
        Args:
            chain_id: The chain ID to get checkpoints for.
            
        Returns:
            List of checkpoints for the chain.
        """
        try:
            # This is a simplified implementation
            # In production, you'd want better indexing and querying
            checkpoints = []
            
            # Search for checkpoints with the chain_id pattern
            # This would be more efficient with proper indexing
            results = self.storage.search(f"checkpoint:{chain_id}:", limit=100)
            
            for result in results:
                if isinstance(result, dict):
                    try:
                        checkpoint = ChainCheckpoint(**result)
                        checkpoints.append(checkpoint)
                    except Exception as e:
                        logger.warning(f"Failed to parse checkpoint data: {e}")
            
            # Sort by timestamp
            checkpoints.sort(key=lambda x: x.timestamp)
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to get checkpoints for chain {chain_id}: {e}")
            return []
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_id: The checkpoint ID to delete.
            
        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            # Find and delete the checkpoint
            for chain_id in self._get_all_chain_ids():
                key = f"checkpoint:{chain_id}:{checkpoint_id}"
                if self.storage.get(key):
                    # Note: Storage protocol doesn't have delete method
                    # This would need to be implemented in the storage backend
                    logger.warning(f"Delete not implemented for checkpoint {checkpoint_id}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """Clean up old checkpoints.
        
        Args:
            max_age_days: Maximum age of checkpoints to keep.
            
        Returns:
            Number of checkpoints cleaned up.
        """
        try:
            # This would need proper implementation with storage backend support
            logger.warning("Checkpoint cleanup not fully implemented")
            return 0
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics.
        """
        try:
            # This would need proper implementation with storage backend support
            return {
                "total_checkpoints": 0,
                "storage_size_bytes": 0,
                "oldest_checkpoint": None,
                "newest_checkpoint": None,
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def _get_all_chain_ids(self) -> List[str]:
        """Get all chain IDs that have checkpoints.
        
        Returns:
            List of chain IDs.
        """
        # This is a placeholder implementation
        # In production, you'd want proper indexing
        return []
