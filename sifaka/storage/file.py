"""Simple file persistence implementation for Sifaka.

Just saves thoughts to JSON files. That's it.
"""

import json
from pathlib import Path
from typing import List, Optional

from pydantic_graph.persistence import BaseStatePersistence
from sifaka.core.thought import SifakaThought
from sifaka.utils import get_logger

logger = get_logger(__name__)


class SifakaFilePersistence(BaseStatePersistence[SifakaThought, str]):
    """Simple file-based persistence.

    Just saves thoughts as JSON files in a directory.
    """

    def __init__(self, storage_dir: str = "thoughts", file_prefix: str = ""):
        """Initialize file persistence.

        Args:
            storage_dir: Directory to save thoughts in
            file_prefix: Optional prefix for thought files (e.g., "n_critics_")
        """
        self.storage_dir = Path(storage_dir)
        self.file_prefix = file_prefix
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"Saving thoughts to: {self.storage_dir.absolute()}")

    def _get_file_path(self, thought_id: str) -> Path:
        """Get the file path for a thought ID."""
        filename = f"{self.file_prefix}{thought_id}.json"
        return self.storage_dir / filename

    async def store_thought(self, thought: SifakaThought) -> None:
        """Store a thought as a JSON file."""
        try:
            file_path = self._get_file_path(thought.id)

            # Convert to dict and handle numpy types
            thought_dict = thought.model_dump()

            # Convert any numpy types and datetime objects to JSON-serializable types
            def convert_types(obj):
                import datetime

                if hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, "tolist"):  # numpy array
                    return obj.tolist()
                elif isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                elif isinstance(obj, datetime.date):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj

            thought_dict = convert_types(thought_dict)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(thought_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved thought {thought.id} to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save thought {thought.id}: {e}")
            raise

    async def retrieve_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a thought from a JSON file."""
        try:
            file_path = self._get_file_path(thought_id)

            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                thought_dict = json.load(f)

            return SifakaThought.model_validate(thought_dict)

        except Exception as e:
            logger.error(f"Failed to load thought {thought_id}: {e}")
            return None

    # Required BaseStatePersistence methods - just save thoughts when needed
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
        """Save the final thought."""
        await self.store_thought(state)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: "SifakaThought", next_node
    ) -> None:
        """Save the thought."""
        await self.store_thought(state)

    async def snapshot_node(self, state: "SifakaThought", next_node: str) -> None:
        """Save the thought."""
        await self.store_thought(state)

    async def load_state(self, state_id: str) -> Optional["SifakaThought"]:
        """Load a previously saved state."""
        return await self.retrieve_thought(state_id)
