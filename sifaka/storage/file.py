"""Simple file persistence implementation for Sifaka.

Just saves thoughts to JSON files. That's it.
"""

import json
from pathlib import Path
from typing import List, Optional

from .base import SifakaBasePersistence
from sifaka.core.thought import SifakaThought
from sifaka.utils import get_logger

logger = get_logger(__name__)


class SifakaFilePersistence(SifakaBasePersistence):
    """Simple file-based persistence.

    Just saves thoughts as JSON files in a directory.
    """

    def __init__(self, storage_dir: str = "thoughts", file_prefix: str = ""):
        """Initialize file persistence.

        Args:
            storage_dir: Directory to save thoughts in
            file_prefix: Optional prefix for thought files (e.g., "n_critics_")
        """
        super().__init__("sifaka")
        self.storage_dir = Path(storage_dir)
        self.file_prefix = file_prefix
        self.storage_dir.mkdir(exist_ok=True)
        logger.debug(
            f"Initialized SifakaFilePersistence at {self.storage_dir} with prefix '{file_prefix}'"
        )

    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data at the given key."""
        # For thoughts, save to {prefix}{thought_id}.json
        if ":thought:" in key:
            thought_id = key.split(":")[-1]
            filename = f"{self.file_prefix}{thought_id}.json"
            file_path = self.storage_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                # Try to pretty-print JSON if possible
                try:
                    json_data = json.loads(data)
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    f.write(data)

            logger.debug(f"Saved thought to {file_path}")
        else:
            # For other data (run status, etc.), just ignore it
            # We only care about saving actual thoughts
            pass

    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from the given key."""
        if ":thought:" in key:
            thought_id = key.split(":")[-1]
            filename = f"{self.file_prefix}{thought_id}.json"
            file_path = self.storage_dir / filename

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Try to parse as JSON and re-serialize
                    try:
                        data = json.loads(content)
                        return json.dumps(data)
                    except json.JSONDecodeError:
                        return content

        return None

    async def _delete_raw(self, key: str) -> bool:
        """Delete data at the given key."""
        if ":thought:" in key:
            thought_id = key.split(":")[-1]
            filename = f"{self.file_prefix}{thought_id}.json"
            file_path = self.storage_dir / filename

            if file_path.exists():
                file_path.unlink()
                return True

        return False

    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys matching the given pattern."""
        keys = []

        if ":thought:" in pattern:
            # Look for files with our prefix
            search_pattern = f"{self.file_prefix}*.json"
            for file_path in self.storage_dir.glob(search_pattern):
                # Remove prefix to get thought_id
                filename = file_path.stem
                if filename.startswith(self.file_prefix):
                    thought_id = filename[len(self.file_prefix) :]
                    key = f"{self.key_prefix}:thought:{thought_id}"
                    keys.append(key)

        return keys

    # Required BaseStatePersistence methods - minimal implementation
    def record_run(self, snapshot_id: str):
        """Record the run of a node."""
        return self._RunRecorder()

    class _RunRecorder:
        """Minimal run recorder that does nothing."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    async def load_all(self, run_id: str) -> List["SifakaThought"]:
        """Load all states for a given run."""
        return []

    async def load_next(self, run_id: str) -> Optional["SifakaThought"]:
        """Load the next state for a given run."""
        return None

    async def snapshot_end(self, state: "SifakaThought", end) -> None:
        """Snapshot the final state at the end of execution."""
        await self.store_thought(state)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: "SifakaThought", next_node
    ) -> None:
        """Snapshot the state if it's new."""
        await self.store_thought(state)

    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Snapshot the current state before executing a node."""
        await self.store_thought(state)

    async def load_state(self, state_id: str) -> Optional["SifakaThought"]:
        """Load a previously saved state."""
        return await self.retrieve_thought(state_id)
