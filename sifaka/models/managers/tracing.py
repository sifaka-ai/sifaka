"""
Tracing manager for model providers.

This module provides the TracingManager class which is responsible for
managing tracing functionality for model providers.
"""

from datetime import datetime
from typing import Any, Dict, Optional

# Import configuration directly to avoid circular dependencies
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.tracing import Tracer
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class TracingManager:
    """
    Manages tracing functionality for model providers.

    This class is responsible for creating and managing tracers,
    and providing a consistent interface for tracing operations.

    Lifecycle:
    1. Initialization: Set up the manager with a model name, configuration, and tracer
    2. Usage: Record trace events during model operations
    3. Cleanup: Flush any pending trace events when no longer needed

    Examples:
        ```python
        # Create a tracing manager
        manager = TracingManager(
            model_name="claude-3-opus",
            config=ModelConfig(trace_enabled=True),
            tracer=None  # Will create a default tracer when needed
        )

        # Record a trace event
        manager.trace_event("generate", {"prompt_tokens": 10, "response_tokens": 20})
        ```
    """

    def __init__(self, model_name: str, config: ModelConfig, tracer: Optional[Tracer] = None):
        """
        Initialize a TracingManager instance.

        Args:
            model_name: The name of the model to trace operations for
            config: The model configuration
            tracer: Optional tracer to use
        """
        self._model_name = model_name
        self._config = config
        self._tracer = tracer or (Tracer() if config.trace_enabled else None)

    def trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record a trace event if tracing is enabled.

        Args:
            event_type: The type of event to record
            data: The data to record with the event
        """
        if self._tracer and self._config.trace_enabled:
            trace_id = datetime.now().strftime(f"{self._model_name}_%Y%m%d%H%M%S")
            self._tracer.add_event(trace_id, event_type, data)

    def is_tracing_enabled(self) -> bool:
        """
        Check if tracing is enabled.

        Returns:
            True if tracing is enabled, False otherwise
        """
        return self._config.trace_enabled and self._tracer is not None
