"""
Tracing utilities for Sifaka.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from typing_extensions import NotRequired, TypedDict

from .logging import get_logger

logger = get_logger(__name__)

class TraceEvent(TypedDict):
    """A trace event."""

    type: str
    timestamp: str
    data: Dict[str, Any]
    component: NotRequired[str]

class TraceData(TypedDict):
    """Type definition for trace data."""

    id: str
    start_time: str
    end_time: Optional[str]
    events: List[TraceEvent]

@runtime_checkable
class TraceStorage(Protocol):
    """Protocol for trace storage backends."""

    def save_trace(self, trace_id: str, data: TraceData) -> None:
        """Save a trace to storage."""
        ...

    def load_trace(self, trace_id: str) -> TraceData:
        """Load a trace from storage."""
        ...

    def delete_trace(self, trace_id: str) -> None:
        """Delete a trace from storage."""
        ...

@runtime_checkable
class TraceFormatter(Protocol):
    """Protocol for trace formatters."""

    def format_trace(self, data: TraceData) -> str:
        """Format a trace for output."""
        ...

@dataclass(frozen=True)
class TraceConfig:
    """Immutable configuration for tracers."""

    trace_dir: Optional[Path] = None
    in_memory: bool = True
    max_events: int = 1000
    auto_flush: bool = False
    component_name: str = "sifaka"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_events < 1:
            raise ValueError("max_events must be positive")
        if not isinstance(self.in_memory, bool):
            raise ValueError("in_memory must be a boolean")

class FileTraceStorage(TraceStorage):
    """File-based trace storage implementation."""

    def __init__(self, trace_dir: Path) -> None:
        """Initialize the file storage."""
        self.trace_dir = trace_dir
        if not self.trace_dir.exists():
            self.trace_dir.mkdir(parents=True)
            logger.info("Created trace directory: %s", self.trace_dir)

    def save_trace(self, trace_id: str, data: TraceData) -> None:
        """Save a trace to a JSON file."""
        trace_path = self.trace_dir / f"{trace_id}.json"
        with trace_path.open("w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved trace to %s", trace_path)

    def load_trace(self, trace_id: str) -> TraceData:
        """Load a trace from a JSON file."""
        trace_path = self.trace_dir / f"{trace_id}.json"
        if not trace_path.exists():
            raise ValueError(f"Trace file not found: {trace_path}")
        with trace_path.open("r") as f:
            return json.load(f)

    def delete_trace(self, trace_id: str) -> None:
        """Delete a trace file."""
        trace_path = self.trace_dir / f"{trace_id}.json"
        if trace_path.exists():
            trace_path.unlink()
            logger.debug("Deleted trace file: %s", trace_path)

class MemoryTraceStorage(TraceStorage):
    """In-memory trace storage implementation."""

    def __init__(self) -> None:
        """Initialize the memory storage."""
        self.traces: Dict[str, TraceData] = {}

    def save_trace(self, trace_id: str, data: TraceData) -> None:
        """Save a trace to memory."""
        self.traces[trace_id] = data
        logger.debug("Saved trace to memory: %s", trace_id)

    def load_trace(self, trace_id: str) -> TraceData:
        """Load a trace from memory."""
        if trace_id not in self.traces:
            raise ValueError(f"Trace not found: {trace_id}")
        return self.traces[trace_id]

    def delete_trace(self, trace_id: str) -> None:
        """Delete a trace from memory."""
        if trace_id in self.traces:
            del self.traces[trace_id]
            logger.debug("Deleted trace from memory: %s", trace_id)

class JSONTraceFormatter(TraceFormatter):
    """JSON trace formatter implementation."""

    def format_trace(self, data: TraceData) -> str:
        """Format a trace as JSON."""
        return json.dumps(data, indent=2)

T = TypeVar("T", bound=TraceStorage)

@dataclass
class Tracer(Generic[T]):
    """
    Tracer for Sifaka.

    A tracer records events for debugging and auditing.
    """

    config: TraceConfig = field(default_factory=TraceConfig)
    storage: T = field(init=False)
    formatter: TraceFormatter = field(default_factory=JSONTraceFormatter)
    _active_traces: Dict[str, TraceData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the tracer with appropriate storage."""
        if self.config.trace_dir:
            self.storage = FileTraceStorage(self.config.trace_dir)  # type: ignore
        else:
            self.storage = MemoryTraceStorage()  # type: ignore

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """Start a new trace."""
        if trace_id is None:
            trace_id = datetime.now().strftime("%Y%m%d%H%M%S")

        trace_data: TraceData = {
            "id": trace_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "events": [],
        }

        self._active_traces[trace_id] = trace_data
        logger.debug("Started trace: %s", trace_id)
        return trace_id

    def add_event(self, trace_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to a trace."""
        if trace_id not in self._active_traces:
            logger.warning("Trace %s not found, creating new trace", trace_id)
            self.start_trace(trace_id)

        event: TraceEvent = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "component": self.config.component_name,
        }

        trace = self._active_traces[trace_id]
        trace["events"].append(event)

        if self.config.auto_flush and len(trace["events"]) >= self.config.max_events:
            self.end_trace(trace_id)

        logger.debug("Added event to trace %s: %s", trace_id, event_type)

    def end_trace(self, trace_id: str) -> TraceData:
        """End a trace and save it to storage."""
        if trace_id not in self._active_traces:
            raise ValueError(f"Trace {trace_id} not found")

        trace = self._active_traces[trace_id]
        trace["end_time"] = datetime.now().isoformat()

        self.storage.save_trace(trace_id, trace)
        del self._active_traces[trace_id]

        logger.debug("Ended and saved trace: %s", trace_id)
        return trace

    def get_trace(self, trace_id: str) -> TraceData:
        """Get a trace by ID."""
        if trace_id in self._active_traces:
            return self._active_traces[trace_id]
        return self.storage.load_trace(trace_id)

    def clear_traces(self) -> None:
        """Clear all traces."""
        self._active_traces.clear()
        logger.info("Cleared all active traces")
