"""Enhanced file persistence implementation for Sifaka.

This module extends PydanticAI's FileStatePersistence with Sifaka-specific
features like indexing, search, and backup/restore functionality.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_graph.persistence.file import FileStatePersistence

from .base import SifakaBasePersistence
from sifaka.core.thought import SifakaThought
from sifaka.utils import get_logger

logger = get_logger(__name__)


class SifakaFilePersistence(SifakaBasePersistence):
    """Enhanced file-based persistence with indexing and search.
    
    This implementation extends PydanticAI's FileStatePersistence to provide:
    
    - Individual JSON files per thought for better performance
    - Thought indexing for fast search and retrieval
    - Backup and restore functionality
    - File rotation and cleanup policies
    - Human-readable storage format
    
    Directory structure:
    ```
    storage_dir/
    ├── thoughts/
    │   ├── {thought_id}.json
    │   └── ...
    ├── indexes/
    │   ├── all_thoughts.json
    │   ├── by_conversation.json
    │   └── by_date.json
    ├── snapshots/
    │   ├── {thought_id}_{node}.json
    │   └── ...
    └── backups/
        ├── backup_20240101_120000/
        └── ...
    ```
    
    Attributes:
        storage_dir: Base directory for all storage
        max_backups: Maximum number of backups to keep
        auto_backup: Whether to automatically create backups
    """
    
    def __init__(
        self,
        storage_dir: str = "sifaka_storage",
        key_prefix: str = "sifaka",
        max_backups: int = 10,
        auto_backup: bool = True
    ):
        """Initialize file persistence.
        
        Args:
            storage_dir: Base directory for storage
            key_prefix: Prefix for storage keys (used in indexes)
            max_backups: Maximum number of backups to keep
            auto_backup: Whether to automatically create backups
        """
        super().__init__(key_prefix)
        self.storage_dir = Path(storage_dir)
        self.max_backups = max_backups
        self.auto_backup = auto_backup
        
        # Create directory structure
        self.thoughts_dir = self.storage_dir / "thoughts"
        self.indexes_dir = self.storage_dir / "indexes"
        self.snapshots_dir = self.storage_dir / "snapshots"
        self.backups_dir = self.storage_dir / "backups"
        
        for dir_path in [self.thoughts_dir, self.indexes_dir, self.snapshots_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Initialized SifakaFilePersistence at {self.storage_dir}")
    
    def _get_thought_file(self, thought_id: str) -> Path:
        """Get the file path for a thought.
        
        Args:
            thought_id: The thought ID
            
        Returns:
            Path to the thought file
        """
        return self.thoughts_dir / f"{thought_id}.json"
    
    def _get_snapshot_file(self, thought_id: str, node_name: str) -> Path:
        """Get the file path for a snapshot.
        
        Args:
            thought_id: The thought ID
            node_name: The node name
            
        Returns:
            Path to the snapshot file
        """
        return self.snapshots_dir / f"{thought_id}_{node_name}.json"
    
    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data at the given key.
        
        For file storage, we extract the thought ID from the key
        and store in the appropriate file.
        
        Args:
            key: Storage key (format: prefix:thought:id or prefix:snapshot:id:node)
            data: Raw data to store (JSON string)
        """
        try:
            # Parse the key to determine storage location
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Regular thought storage
                thought_id = key_parts[2]
                file_path = self._get_thought_file(thought_id)
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Snapshot storage
                thought_id = key_parts[2]
                node_name = key_parts[3]
                file_path = self._get_snapshot_file(thought_id, node_name)
            else:
                # Generic file storage
                safe_key = key.replace(":", "_").replace("/", "_")
                file_path = self.storage_dir / f"{safe_key}.json"
            
            # Write data atomically
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                # Pretty-print JSON for human readability
                json_data = json.loads(data)
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Atomic rename
            temp_path.replace(file_path)
            logger.debug(f"File stored: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to store file for key {key}: {e}")
            raise
    
    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from the given key.
        
        Args:
            key: Storage key
            
        Returns:
            Raw data if found, None otherwise
        """
        try:
            # Parse the key to determine file location
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Regular thought retrieval
                thought_id = key_parts[2]
                file_path = self._get_thought_file(thought_id)
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Snapshot retrieval
                thought_id = key_parts[2]
                node_name = key_parts[3]
                file_path = self._get_snapshot_file(thought_id, node_name)
            else:
                # Generic file retrieval
                safe_key = key.replace(":", "_").replace("/", "_")
                file_path = self.storage_dir / f"{safe_key}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                serialized = json.dumps(data)
                logger.debug(f"File retrieved: {file_path}")
                return serialized
                
        except Exception as e:
            logger.warning(f"Failed to retrieve file for key {key}: {e}")
            return None
    
    async def _delete_raw(self, key: str) -> bool:
        """Delete data at the given key.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if key didn't exist
        """
        try:
            # Parse the key to determine file location
            key_parts = key.split(":")
            
            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Regular thought deletion
                thought_id = key_parts[2]
                file_path = self._get_thought_file(thought_id)
            elif len(key_parts) >= 4 and key_parts[1] == "snapshot":
                # Snapshot deletion
                thought_id = key_parts[2]
                node_name = key_parts[3]
                file_path = self._get_snapshot_file(thought_id, node_name)
            else:
                # Generic file deletion
                safe_key = key.replace(":", "_").replace("/", "_")
                file_path = self.storage_dir / f"{safe_key}.json"
            
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"File deleted: {file_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.warning(f"Failed to delete file for key {key}: {e}")
            return False
    
    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the given pattern.
        
        Args:
            pattern: Key pattern to match
            
        Returns:
            List of matching keys
        """
        import fnmatch
        
        keys = []
        
        try:
            # List thought files
            if fnmatch.fnmatch(f"{self.key_prefix}:thought:*", pattern):
                for file_path in self.thoughts_dir.glob("*.json"):
                    thought_id = file_path.stem
                    key = f"{self.key_prefix}:thought:{thought_id}"
                    if fnmatch.fnmatch(key, pattern):
                        keys.append(key)
            
            # List snapshot files
            if fnmatch.fnmatch(f"{self.key_prefix}:snapshot:*", pattern):
                for file_path in self.snapshots_dir.glob("*.json"):
                    # Parse filename: {thought_id}_{node_name}.json
                    filename = file_path.stem
                    if "_" in filename:
                        thought_id, node_name = filename.rsplit("_", 1)
                        key = f"{self.key_prefix}:snapshot:{thought_id}:{node_name}"
                        if fnmatch.fnmatch(key, pattern):
                            keys.append(key)
            
            logger.debug(f"File listed {len(keys)} keys matching pattern: {pattern}")
            return keys
            
        except Exception as e:
            logger.warning(f"Failed to list files with pattern {pattern}: {e}")
            return []
    
    async def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of all stored data.
        
        Args:
            backup_name: Optional backup name (defaults to timestamp)
            
        Returns:
            Path to the created backup directory
        """
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_path = self.backups_dir / backup_name
            
            # Copy all data to backup directory
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            shutil.copytree(self.storage_dir, backup_path, ignore=shutil.ignore_patterns("backups"))
            
            # Clean up old backups if needed
            await self._cleanup_old_backups()
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backups to maintain max_backups limit."""
        try:
            backup_dirs = [d for d in self.backups_dir.iterdir() if d.is_dir()]
            backup_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            for old_backup in backup_dirs[self.max_backups:]:
                shutil.rmtree(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
    
    # PydanticAI BaseStatePersistence interface implementation
    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the current state before executing a node.
        
        Args:
            state: Current thought state
            next_node: Name of the next node to execute
        """
        try:
            # Store the thought with a snapshot key
            snapshot_key = f"{self.key_prefix}:snapshot:{state.id}:{next_node}"
            data = await self.serialize_state(state)
            await self._store_raw(snapshot_key, data)
            
            # Also store as regular thought
            await self.store_thought(state)
            
            # Create backup if auto_backup is enabled
            if self.auto_backup:
                await self.create_backup()
            
            logger.debug(f"File snapshotted state for thought {state.id} before node {next_node}")
            
        except Exception as e:
            logger.error(f"Failed to snapshot state for thought {state.id}: {e}")
            raise
    
    async def load_state(self, state_id: str) -> Optional["SifakaThought"]:
        """Load a previously saved state.
        
        Args:
            state_id: The state ID to load
            
        Returns:
            The loaded state if found, None otherwise
        """
        return await self.retrieve_thought(state_id)
