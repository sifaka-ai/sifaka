"""
Tracing utilities for Sifaka.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from .logging import get_logger

logger = get_logger(__name__)


class TraceEvent(TypedDict):
    """
    A trace event.

    Attributes:
        component: The component that generated the event
        event_type: The type of event
        data: The event data
    """

    component: str
    event_type: str
    data: Dict[str, Any]


class TraceData(TypedDict):
    """Type definition for trace data."""

    id: str
    start_time: str
    end_time: Optional[str]
    events: List[TraceEvent]


@dataclass
class Tracer:
    """
    Tracer for Sifaka.

    A tracer records the steps of the reflection process for debugging and auditing.

    Attributes:
        trace_dir: Directory to save trace files
        in_memory: Whether to keep traces in memory
        traces: Dictionary of trace data
    """

    trace_dir: Optional[Path] = None
    in_memory: bool = True
    traces: Dict[str, TraceData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the tracer."""
        if self.trace_dir and not self.trace_dir.exists():
            self.trace_dir.mkdir(parents=True)
            logger.info("Created trace directory: %s", self.trace_dir)

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """
        Start a new trace.

        Args:
            trace_id: ID for the trace

        Returns:
            The trace ID
        """
        if trace_id is None:
            trace_id = datetime.now().strftime("%Y%m%d%H%M%S")

        self.traces[trace_id] = {
            "id": trace_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "events": [],
        }

        logger.debug("Started trace: %s", trace_id)
        return trace_id

    def add_event(self, trace_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Add an event to a trace.

        Args:
            trace_id: The trace ID
            event_type: The type of event
            data: The event data

        Raises:
            ValueError: If the trace ID is not found
        """
        if trace_id not in self.traces:
            logger.warning("Trace %s not found, creating new trace", trace_id)
            self.start_trace(trace_id)

        event: TraceEvent = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        self.traces[trace_id]["events"].append(event)
        logger.debug("Added event to trace %s: %s", trace_id, event_type)

    def end_trace(self, trace_id: str) -> TraceData:
        """
        End a trace and save it if a trace directory is configured.

        Args:
            trace_id: The trace ID

        Returns:
            The trace data

        Raises:
            ValueError: If the trace ID is not found
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace {trace_id} not found")

        self.traces[trace_id]["end_time"] = datetime.now().isoformat()

        if self.trace_dir:
            trace_path = self.trace_dir / f"{trace_id}.json"
            with trace_path.open("w") as f:
                json.dump(self.traces[trace_id], f, indent=2)
            logger.info("Saved trace to %s", trace_path)

        trace_data = self.traces[trace_id]

        if not self.in_memory:
            del self.traces[trace_id]
            logger.debug("Removed trace from memory: %s", trace_id)

        return trace_data

    def get_trace(self, trace_id: str) -> TraceData:
        """
        Get a trace by ID.

        Args:
            trace_id: The trace ID

        Returns:
            The trace data

        Raises:
            ValueError: If the trace ID is not found
        """
        if trace_id not in self.traces:
            raise ValueError(f"Trace {trace_id} not found")

        return self.traces[trace_id]

    def clear_traces(self) -> None:
        """Clear all traces from memory."""
        self.traces.clear()
        logger.info("Cleared all traces from memory")
