"""In-memory persistence implementation for Sifaka.

This module provides a simple in-memory storage implementation for development
and testing. Data is not persisted across process restarts.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.core.thought import SifakaThought

from .base import SifakaBasePersistence
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryPersistence(SifakaBasePersistence):
    """Simple in-memory persistence with no durability.

    This implementation stores all data in Python dictionaries and provides
    no persistence across process restarts. Perfect for:

    - Development and testing
    - Simple scripts and experiments
    - Cases where persistence is not needed
    - Fast prototyping

    Attributes:
        data: Dictionary storing all key-value pairs
        indexes: Dictionary storing index data
    """

    def __init__(self, key_prefix: str = "sifaka"):
        """Initialize memory persistence.

        Args:
            key_prefix: Prefix for all storage keys
        """
        super().__init__(key_prefix)
        self.data: Dict[str, str] = {}
        self.indexes: Dict[str, List[str]] = {}
        logger.debug("Initialized MemoryPersistence")

    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data at the given key.

        Args:
            key: Storage key
            data: Raw data to store
        """
        self.data[key] = data
        logger.debug(f"Memory stored key: {key}")

    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from the given key.

        Args:
            key: Storage key

        Returns:
            Raw data if found, None otherwise
        """
        data = self.data.get(key)
        logger.debug(f"Memory retrieved key: {key} -> {'found' if data else 'not found'}")
        return data

    async def _delete_raw(self, key: str) -> bool:
        """Delete data at the given key.

        Args:
            key: Storage key

        Returns:
            True if deleted, False if key didn't exist
        """
        if key in self.data:
            del self.data[key]
            logger.debug(f"Memory deleted key: {key}")
            return True
        else:
            logger.debug(f"Memory key not found for deletion: {key}")
            return False

    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the given pattern.

        Args:
            pattern: Key pattern to match (supports * wildcard)

        Returns:
            List of matching keys
        """
        import fnmatch

        matching_keys = [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]

        logger.debug(f"Memory listed {len(matching_keys)} keys matching pattern: {pattern}")
        return matching_keys

    async def clear(self) -> None:
        """Clear all stored data.

        This removes all thoughts and indexes from memory.
        """
        self.data.clear()
        self.indexes.clear()
        logger.debug("Memory storage cleared")

    async def get_stats(self) -> Dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        thought_keys = await self._list_keys(f"{self.key_prefix}:thought:*")
        conversation_keys = await self._list_keys(f"{self.key_prefix}:conversation:*")
        index_keys = await self._list_keys(f"{self.key_prefix}:index:*")

        return {
            "total_keys": len(self.data),
            "thoughts": len(thought_keys),
            "conversations": len(conversation_keys),
            "indexes": len(index_keys),
        }

    # PydanticAI BaseStatePersistence interface implementation
    async def record_run(self, run_id: str, state: "SifakaThought") -> None:
        """Record a run with its final state.

        Args:
            run_id: Unique identifier for the run
            state: Final state of the run
        """
        try:
            run_key = f"{self.key_prefix}:run:{run_id}"
            data = await self.serialize_state(state)
            await self._store_raw(run_key, data)
            logger.debug(f"Recorded run {run_id}")
        except Exception as e:
            logger.error(f"Failed to record run {run_id}: {e}")
            raise

    async def load_all(self, run_id: str) -> List["SifakaThought"]:
        """Load all states for a given run.

        Args:
            run_id: The run ID to load states for

        Returns:
            List of all states for the run
        """
        try:
            # Load the final run state
            run_key = f"{self.key_prefix}:run:{run_id}"
            data = await self._retrieve_raw(run_key)

            if data is None:
                return []

            state = await self.deserialize_state(data)
            return [state]

        except Exception as e:
            logger.error(f"Failed to load all states for run {run_id}: {e}")
            return []

    async def load_next(self, run_id: str) -> Optional["SifakaThought"]:
        """Load the next state for a given run.

        Args:
            run_id: The run ID to load the next state for

        Returns:
            The next state if available, None otherwise
        """
        try:
            run_key = f"{self.key_prefix}:run:{run_id}"
            data = await self._retrieve_raw(run_key)

            if data is None:
                return None

            return await self.deserialize_state(data)

        except Exception as e:
            logger.error(f"Failed to load next state for run {run_id}: {e}")
            return None

    async def snapshot_end(self, state: "SifakaThought") -> None:
        """Snapshot the final state at the end of execution.

        Args:
            state: Final state to snapshot
        """
        try:
            # Store as final snapshot
            end_key = f"{self.key_prefix}:end:{state.id}"
            data = await self.serialize_state(state)
            await self._store_raw(end_key, data)

            # Also store as regular thought
            await self.store_thought(state)

            logger.debug(f"Snapshotted end state for thought {state.id}")

        except Exception as e:
            logger.error(f"Failed to snapshot end state for thought {state.id}: {e}")
            raise

    async def snapshot_node_if_new(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the state if it's new (not already snapshotted).

        Args:
            state: Current state
            next_node: Name of the next node to execute
        """
        try:
            snapshot_key = f"{self.key_prefix}:snapshot:{state.id}:{next_node}"

            # Check if snapshot already exists
            existing = await self._retrieve_raw(snapshot_key)
            if existing is None:
                # Only snapshot if new
                await self.snapshot_node(state, next_node)
            else:
                logger.debug(
                    f"Snapshot already exists for thought {state.id} before node {next_node}"
                )

        except Exception as e:
            logger.error(f"Failed to check/snapshot state for thought {state.id}: {e}")
            raise

    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the current state before executing a node.

        This is called by PydanticAI's graph execution to save state
        at each node transition for resumability.

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

            logger.debug(f"Snapshotted state for thought {state.id} before node {next_node}")

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

    async def list_snapshots(self, state_id: str) -> List[str]:
        """List all snapshots for a given state.

        Args:
            state_id: The state ID to list snapshots for

        Returns:
            List of snapshot keys
        """
        pattern = f"{self.key_prefix}:snapshot:{state_id}:*"
        return await self._list_keys(pattern)

    async def resume_from_snapshot(
        self, state_id: str, node_name: str
    ) -> Optional["SifakaThought"]:
        """Resume execution from a specific snapshot.

        Args:
            state_id: The state ID to resume
            node_name: The node name to resume from

        Returns:
            The state to resume from if found, None otherwise
        """
        try:
            snapshot_key = f"{self.key_prefix}:snapshot:{state_id}:{node_name}"
            data = await self._retrieve_raw(snapshot_key)

            if data is None:
                return None

            state = await self.deserialize_state(data)
            logger.debug(f"Resumed state {state_id} from node {node_name}")
            return state

        except Exception as e:
            logger.error(f"Failed to resume state {state_id} from node {node_name}: {e}")
            return None
