"""
JSON-based persistence implementation for Sifaka.

This module provides a file-based JSON storage backend for thoughts,
with support for querying, indexing, and concurrent access.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

# Try to import fcntl for file locking (Unix-like systems)
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from sifaka.core.thought import Thought
from sifaka.utils.error_handling import error_context
from sifaka.utils.logging import get_logger
from .base import ThoughtStorage, ThoughtQuery, ThoughtQueryResult, PersistenceError

logger = get_logger(__name__)


class JSONThoughtStorage(ThoughtStorage):
    """JSON file-based storage for thoughts.

    This implementation stores thoughts as individual JSON files in a
    directory structure organized by date. It supports concurrent
    access through file locking and provides basic querying capabilities.

    Directory structure:
        storage_dir/
        └── thoughts/
            ├── YYYY-MM-DD/
            │   ├── {thought_id}.json
            │   └── ...
            └── ...

    Features:
        - Individual thought storage as JSON files
        - Date-based directory organization
        - File locking for concurrent access
        - Basic querying through file scanning
        - Configurable file size limits

    Note:
        This is a simple implementation suitable for development and small-scale
        usage. For production use with large datasets, consider implementing
        proper indexing or using a database backend.

    Attributes:
        storage_dir: Root directory for storage
        auto_create_dirs: Whether to automatically create directories
        enable_indexing: Currently unused (placeholder for future indexing)
        max_file_size_mb: Maximum size for individual files
    """

    def __init__(
        self,
        storage_dir: str,
        auto_create_dirs: bool = True,
        enable_indexing: bool = True,
        max_file_size_mb: int = 10,
    ):
        """Initialize JSON storage.

        Args:
            storage_dir: Root directory for storage
            auto_create_dirs: Whether to automatically create directories
            enable_indexing: Whether to maintain search indexes
            max_file_size_mb: Maximum size for individual files
        """
        self.storage_dir = Path(storage_dir)
        self.auto_create_dirs = auto_create_dirs
        self.enable_indexing = enable_indexing
        self.max_file_size_mb = max_file_size_mb

        # Directory structure
        self.thoughts_dir = self.storage_dir / "thoughts"

        if auto_create_dirs:
            self._create_directories()

    def _create_directories(self) -> None:
        """Create the directory structure."""
        with error_context(
            component="JSONThoughtStorage",
            operation="directory creation",
            error_class=PersistenceError,
        ):
            self.thoughts_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created storage directories in {self.storage_dir}")

    @contextmanager
    def _file_lock(self, file_path: Path) -> Any:
        """Context manager for file locking."""
        if HAS_FCNTL:
            # Use fcntl for proper file locking on Unix-like systems
            lock_file = file_path.with_suffix(file_path.suffix + ".lock")
            try:
                with open(lock_file, "w") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    yield
            finally:
                if lock_file.exists():
                    lock_file.unlink()
        else:
            # Fallback for systems without fcntl (like Windows)
            # This is not as robust but provides basic protection
            lock_file = file_path.with_suffix(file_path.suffix + ".lock")
            try:
                # Check if lock file exists
                if lock_file.exists():
                    # Wait a bit and try again
                    time.sleep(0.1)
                    if lock_file.exists():
                        logger.warning(f"Lock file exists for {file_path}, proceeding anyway")

                # Create lock file
                lock_file.touch()
                yield
            finally:
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except Exception:
                        pass  # Ignore errors when cleaning up lock file

    def _get_thought_file_path(self, thought: Thought) -> Path:
        """Get the file path for a thought."""
        date_str = thought.timestamp.strftime("%Y-%m-%d")
        date_dir = self.thoughts_dir / date_str
        if self.auto_create_dirs:
            date_dir.mkdir(exist_ok=True)
        return date_dir / f"{thought.id}.json"

    def _serialize_thought(self, thought: Thought) -> Dict[str, Any]:
        """Serialize a thought to a dictionary."""
        # Use Pydantic's model_dump with custom serialization for datetime
        data = thought.model_dump(mode="json")

        # Ensure timestamp is ISO format string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        return data

    def _deserialize_thought(self, data: Dict[str, Any]) -> Thought:
        """Deserialize a thought from a dictionary."""
        # Convert timestamp string back to datetime if needed
        if isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            except ValueError:
                # Handle legacy formats or parsing errors
                logger.warning(f"Could not parse timestamp: {data.get('timestamp')}")
                data["timestamp"] = datetime.now()

        return Thought.model_validate(data)

    def save_thought(self, thought: Thought) -> None:
        """Save a thought to storage."""
        with error_context(
            component="JSONThoughtStorage",
            operation="save_thought",
            error_class=PersistenceError,
            message_prefix=f"Failed to save thought {thought.id}",
        ):
            file_path = self._get_thought_file_path(thought)

            with self._file_lock(file_path):
                # Serialize the thought
                data = self._serialize_thought(thought)

                # Write to file
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.debug(f"Saved thought {thought.id} to {file_path}")

                # Note: Indexing is not implemented in this simple storage backend

    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Retrieve a thought by ID."""
        with error_context(
            component="JSONThoughtStorage",
            operation="get_thought",
            error_class=PersistenceError,
            message_prefix=f"Failed to retrieve thought {thought_id}",
        ):
            # Try to find the thought file
            thought_file = self._find_thought_file(thought_id)
            if not thought_file:
                return None

            with open(thought_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return self._deserialize_thought(data)

    def _find_thought_file(self, thought_id: str) -> Optional[Path]:
        """Find the file path for a thought ID."""
        # Search through date directories
        for date_dir in self.thoughts_dir.iterdir():
            if date_dir.is_dir():
                thought_file = date_dir / f"{thought_id}.json"
                if thought_file.exists():
                    return thought_file
        return None

    def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought from storage."""
        with error_context(
            component="JSONThoughtStorage",
            operation="delete_thought",
            error_class=PersistenceError,
            message_prefix=f"Failed to delete thought {thought_id}",
        ):
            thought_file = self._find_thought_file(thought_id)
            if not thought_file:
                return False

            with self._file_lock(thought_file):
                thought_file.unlink()
                logger.debug(f"Deleted thought {thought_id}")

                return True

    def query_thoughts(self, query: Optional[ThoughtQuery] = None) -> ThoughtQueryResult:
        """Query thoughts based on criteria."""
        start_time = time.time()

        with error_context(
            component="JSONThoughtStorage",
            operation="query_thoughts",
            error_class=PersistenceError,
            message_prefix="Failed to query thoughts",
        ):
            if query is None:
                query = ThoughtQuery()

            # Collect all thoughts
            all_thoughts = []
            for date_dir in self.thoughts_dir.iterdir():
                if date_dir.is_dir():
                    for thought_file in date_dir.glob("*.json"):
                        try:
                            with open(thought_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            thought = self._deserialize_thought(data)
                            all_thoughts.append(thought)
                        except Exception as e:
                            logger.warning(f"Failed to load thought from {thought_file}: {e}")
                            continue

            # Apply filters
            filtered_thoughts = self._apply_filters(all_thoughts, query)

            # Apply sorting
            sorted_thoughts = self._apply_sorting(filtered_thoughts, query)

            # Apply pagination
            total_count = len(sorted_thoughts)
            offset = query.offset or 0
            limit = query.limit or 100
            paginated_thoughts = sorted_thoughts[offset : offset + limit]

            execution_time = (time.time() - start_time) * 1000

            return ThoughtQueryResult(
                thoughts=paginated_thoughts,
                total_count=total_count,
                query=query,
                execution_time_ms=execution_time,
                metadata={"storage_type": "json", "files_scanned": len(all_thoughts)},
            )

    def _apply_filters(self, thoughts: List[Thought], query: ThoughtQuery) -> List[Thought]:
        """Apply query filters to thoughts."""
        filtered = thoughts

        # ID filters
        if query.thought_ids:
            filtered = [t for t in filtered if t.id in query.thought_ids]

        if query.chain_ids:
            filtered = [t for t in filtered if t.chain_id and t.chain_id in query.chain_ids]

        if query.parent_ids:
            filtered = [t for t in filtered if t.parent_id and t.parent_id in query.parent_ids]

        # Content filters
        if query.prompts:
            filtered = [t for t in filtered if any(prompt in t.prompt for prompt in query.prompts)]

        if query.text_contains:
            filtered = [t for t in filtered if t.text and query.text_contains in t.text]

        # Iteration filters
        if query.min_iteration is not None:
            filtered = [t for t in filtered if t.iteration >= query.min_iteration]

        if query.max_iteration is not None:
            filtered = [t for t in filtered if t.iteration <= query.max_iteration]

        # Date filters
        if query.start_date:
            filtered = [t for t in filtered if t.timestamp >= query.start_date]

        if query.end_date:
            filtered = [t for t in filtered if t.timestamp <= query.end_date]

        # Feature filters
        if query.has_validation_results is not None:
            if query.has_validation_results:
                filtered = [t for t in filtered if t.validation_results]
            else:
                filtered = [t for t in filtered if not t.validation_results]

        if query.has_critic_feedback is not None:
            if query.has_critic_feedback:
                filtered = [t for t in filtered if t.critic_feedback]
            else:
                filtered = [t for t in filtered if not t.critic_feedback]

        if query.has_context is not None:
            if query.has_context:
                filtered = [
                    t for t in filtered if t.pre_generation_context or t.post_generation_context
                ]
            else:
                filtered = [
                    t
                    for t in filtered
                    if not t.pre_generation_context and not t.post_generation_context
                ]

        return filtered

    def _apply_sorting(self, thoughts: List[Thought], query: ThoughtQuery) -> List[Thought]:
        """Apply sorting to thoughts."""
        sort_by = query.sort_by or "timestamp"
        sort_order = query.sort_order or "desc"
        reverse = sort_order == "desc"

        if sort_by == "timestamp":
            return sorted(thoughts, key=lambda t: t.timestamp, reverse=reverse)
        elif sort_by == "iteration":
            return sorted(thoughts, key=lambda t: t.iteration, reverse=reverse)
        elif sort_by == "id":
            return sorted(thoughts, key=lambda t: t.id, reverse=reverse)
        else:
            # Default to timestamp if unknown sort field
            return sorted(thoughts, key=lambda t: t.timestamp, reverse=reverse)

    def get_thought_history(self, thought_id: str) -> List[Thought]:
        """Get the complete history of a thought."""
        with error_context(
            component="JSONThoughtStorage",
            operation="get_thought_history",
            error_class=PersistenceError,
            message_prefix=f"Failed to get history for thought {thought_id}",
        ):
            # Get the thought
            thought = self.get_thought(thought_id)
            if not thought:
                return []

            # Collect all thoughts in the history chain
            history = []
            current = thought

            # Add the current thought
            history.append(current)

            # Follow the parent chain backwards
            while current.parent_id:
                parent = self.get_thought(current.parent_id)
                if not parent:
                    break
                history.append(parent)
                current = parent

            # Reverse to get chronological order (oldest first)
            history.reverse()

            return history

    def get_chain_thoughts(self, chain_id: str) -> List[Thought]:
        """Get all thoughts in a chain."""
        with error_context(
            component="JSONThoughtStorage",
            operation="get_chain_thoughts",
            error_class=PersistenceError,
            message_prefix=f"Failed to get thoughts for chain {chain_id}",
        ):
            query = ThoughtQuery(chain_ids=[chain_id], sort_by="timestamp", sort_order="asc")
            result = self.query_thoughts(query)
            return result.thoughts

    def count_thoughts(self, query: Optional[ThoughtQuery] = None) -> int:
        """Count thoughts matching query criteria."""
        with error_context(
            component="JSONThoughtStorage",
            operation="count_thoughts",
            error_class=PersistenceError,
            message_prefix="Failed to count thoughts",
        ):
            if query is None:
                query = ThoughtQuery()

            # Use query_thoughts but only count, don't return data
            result = self.query_thoughts(query)
            return result.total_count

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the storage backend."""
        with error_context(
            component="JSONThoughtStorage",
            operation="health_check",
            error_class=PersistenceError,
            message_prefix="Health check failed",
        ):
            health_info = {
                "status": "healthy",
                "storage_type": "json",
                "storage_dir": str(self.storage_dir),
                "directories_exist": True,
                "writable": True,
                "total_thoughts": 0,
                "total_size_mb": 0.0,
                "oldest_thought": None,
                "newest_thought": None,
            }

            try:
                # Check if thoughts directory exists
                if not self.thoughts_dir.exists():
                    health_info["directories_exist"] = False
                    health_info["status"] = "unhealthy"

                # Check if storage is writable
                test_file = self.storage_dir / ".health_check"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception:
                    health_info["writable"] = False
                    health_info["status"] = "unhealthy"

                # Count thoughts and calculate storage size
                total_thoughts = 0
                total_size = 0
                oldest_timestamp = None
                newest_timestamp = None

                for date_dir in self.thoughts_dir.iterdir():
                    if date_dir.is_dir():
                        for thought_file in date_dir.glob("*.json"):
                            total_thoughts += 1
                            total_size += thought_file.stat().st_size

                            # Get timestamp from file for oldest/newest tracking
                            try:
                                with open(thought_file, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                timestamp_str = data.get("timestamp")
                                if timestamp_str:
                                    timestamp = datetime.fromisoformat(timestamp_str)
                                    if oldest_timestamp is None or timestamp < oldest_timestamp:
                                        oldest_timestamp = timestamp
                                    if newest_timestamp is None or timestamp > newest_timestamp:
                                        newest_timestamp = timestamp
                            except Exception:
                                continue

                health_info["total_thoughts"] = total_thoughts
                health_info["total_size_mb"] = round(total_size / (1024 * 1024), 2)
                health_info["oldest_thought"] = (
                    oldest_timestamp.isoformat() if oldest_timestamp else None
                )
                health_info["newest_thought"] = (
                    newest_timestamp.isoformat() if newest_timestamp else None
                )

            except Exception as e:
                health_info["status"] = "unhealthy"
                health_info["error"] = str(e)

            return health_info
