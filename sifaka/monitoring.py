"""
Performance monitoring for Sifaka.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics
import threading

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for timing measurements."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: deque = deque(maxlen=1000)  # Keep last 1000 measurements

    def add(self, timing: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_time += timing
        self.min_time = min(self.min_time, timing)
        self.max_time = max(self.max_time, timing)
        self.times.append(timing)

    @property
    def avg_time(self) -> float:
        """Calculate average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Calculate median time."""
        return statistics.median(self.times) if self.times else 0.0

    @property
    def p95_time(self) -> float:
        """Calculate 95th percentile time."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]


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
        self.rule_times = {}  # Dict[str, TimingStats]
        self.generation_times = TimingStats()
        self.critique_times = TimingStats()

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
            self.rule_times[rule_name].add(elapsed)

    def record_generation_time(self, elapsed: float) -> None:
        """Record time taken for model generation."""
        with self.lock:
            self.generation_times.add(elapsed)

    def record_critique_time(self, elapsed: float) -> None:
        """Record time taken for critique."""
        with self.lock:
            self.critique_times.add(elapsed)

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
                        "median_time_ms": stats.median_time * 1000,
                        "p95_time_ms": stats.p95_time * 1000,
                        "min_time_ms": stats.min_time * 1000,
                        "max_time_ms": stats.max_time * 1000,
                    }
                    for name, stats in self.rule_times.items()
                },
                "model_generation": {
                    "count": self.generation_times.count,
                    "avg_time_ms": self.generation_times.avg_time * 1000,
                    "median_time_ms": self.generation_times.median_time * 1000,
                    "p95_time_ms": self.generation_times.p95_time * 1000,
                },
                "critique": {
                    "count": self.critique_times.count,
                    "avg_time_ms": self.critique_times.avg_time * 1000,
                    "median_time_ms": self.critique_times.median_time * 1000,
                    "p95_time_ms": self.critique_times.p95_time * 1000,
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
