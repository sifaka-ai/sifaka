import time
import threading
from typing import Dict, Any

class SifakaMonitor:
    """Monitors and collects metrics about Sifaka's performance and behavior."""

    def __init__(self):
        """Initialize the monitor with empty metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'total_tokens_processed': 0,
            'total_cost': 0.0,
            'errors': []
        }
        self._start_time = time.time()
        self._last_reset = self._start_time
        self._request_times = []
        self._lock = threading.Lock()

    def _get_metrics(self) -> Dict[str, Any]:
        """Get a copy of the current metrics."""
        with self._lock:
            return self.metrics.copy()