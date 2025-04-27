"""
Performance monitoring for Sifaka.
"""

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    count: int = 0
    total_time: float = 0.0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_measurement(self, time: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_time += time
        self.times.append(time)

        if self.min_time is None or time < self.min_time:
            self.min_time = time
        if self.max_time is None or time > self.max_time:
            self.max_time = time

    @property
    def avg_time(self) -> float:
        """Calculate average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """Calculate average of recent times."""
        return statistics.mean(self.times) if self.times else 0.0


class PerformanceMonitor:
    """
    Monitor performance metrics for Sifaka.

    This class tracks:
    - Rule validation times
    - Model generation times
    - Critique times
    - Success/failure rates
    - Cache hit rates
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.lock = threading.Lock()

        # Timing statistics
        self.rule_times: Dict[str, TimingStats] = {}
        self.generation_times = TimingStats()
        self.critique_times = TimingStats()
        self.validation_times = TimingStats()

        # Success/failure tracking
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Start time
        self.start_time = time.time()

    def record_rule_time(self, rule_name: str, elapsed: float) -> None:
        """Record time taken for a rule validation."""
        with self.lock:
            if rule_name not in self.rule_times:
                self.rule_times[rule_name] = TimingStats()
            self.rule_times[rule_name].add_measurement(elapsed)

    def record_generation_time(self, elapsed: float) -> None:
        """Record time taken for model generation."""
        with self.lock:
            self.generation_times.add_measurement(elapsed)

    def record_critique_time(self, elapsed: float) -> None:
        """Record time taken for critique."""
        with self.lock:
            self.critique_times.add_measurement(elapsed)

    def record_validation_time(self, elapsed: float) -> None:
        """Record time taken for validation."""
        with self.lock:
            self.validation_times.add_measurement(elapsed)

    def record_attempt(self, success: bool) -> None:
        """Record a validation attempt."""
        with self.lock:
            self.total_attempts += 1
            if success:
                self.successful_attempts += 1
            else:
                self.failed_attempts += 1

    def record_cache_access(self, hit: bool) -> None:
        """Record a cache access."""
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        with self.lock:
            uptime = time.time() - self.start_time

            # Calculate cache hit rate
            total_cache_accesses = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                self.cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0
            )

            # Calculate success rate
            success_rate = (
                self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0.0
            )

            return {
                "uptime_seconds": uptime,
                "rule_validation": {
                    name: {
                        "count": stats.count,
                        "avg_time_ms": stats.avg_time * 1000,
                        "recent_avg_time_ms": stats.recent_avg_time * 1000,
                        "min_time_ms": (
                            stats.min_time * 1000 if stats.min_time is not None else None
                        ),
                        "max_time_ms": (
                            stats.max_time * 1000 if stats.max_time is not None else None
                        ),
                    }
                    for name, stats in self.rule_times.items()
                },
                "model_generation": {
                    "count": self.generation_times.count,
                    "avg_time_ms": self.generation_times.avg_time * 1000,
                    "recent_avg_time_ms": self.generation_times.recent_avg_time * 1000,
                },
                "critique": {
                    "count": self.critique_times.count,
                    "avg_time_ms": self.critique_times.avg_time * 1000,
                    "recent_avg_time_ms": self.critique_times.recent_avg_time * 1000,
                },
                "validation": {
                    "count": self.validation_times.count,
                    "avg_time_ms": self.validation_times.avg_time * 1000,
                    "recent_avg_time_ms": self.validation_times.recent_avg_time * 1000,
                },
                "attempts": {
                    "total": self.total_attempts,
                    "successful": self.successful_attempts,
                    "failed": self.failed_attempts,
                    "success_rate": success_rate,
                },
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": cache_hit_rate,
                },
            }

    def log_metrics(self) -> None:
        """Log current performance metrics."""
        metrics = self.get_metrics()
        logger.info("Performance metrics:")
        for name, value in metrics.items():
            if "time" in name:
                logger.info(f"  {name}: {value:.2f}ms")
            else:
                logger.info(f"  {name}: {value:.2f}")
