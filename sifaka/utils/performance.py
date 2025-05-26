"""Performance monitoring and optimization utilities for Sifaka.

This module provides tools for monitoring and optimizing performance across
the Sifaka framework, including timing, caching, and metrics collection.

Features:
- Performance timing decorators and context managers
- Memory usage monitoring
- Cache hit/miss statistics
- Chain execution metrics
- Component-level performance tracking

Example:
    ```python
    from sifaka.utils.performance import timer, PerformanceMonitor

    # Use timer decorator
    @timer("model_generation")
    def generate_text(prompt):
        return model.generate(prompt)

    # Use timer context manager
    with timer("validation"):
        result = validator.validate(text)

    # Get performance stats
    monitor = PerformanceMonitor.get_instance()
    stats = monitor.get_stats()
    print(f"Average generation time: {stats['model_generation']['avg_time']:.3f}s")
    ```
"""

import functools
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, TypeVar, Union

from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PerformanceMetric:
    """Container for performance metrics."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    recent_times: "deque[float]" = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """Average of recent execution times."""
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0

    def add_measurement(self, duration: float) -> None:
        """Add a new timing measurement."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_time": self.total_time,
            "call_count": self.call_count,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "recent_avg_time": self.recent_avg_time,
        }


class PerformanceMonitor:
    """Thread-safe performance monitoring singleton."""

    _instance: Optional["PerformanceMonitor"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the performance monitor."""
        self._metrics: Dict[str, PerformanceMetric] = {}
        self._enabled = True
        self._lock = threading.Lock()
        logger.debug("Initialized performance monitor")

    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
        logger.debug("Performance monitoring enabled")

    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
        logger.debug("Performance monitoring disabled")

    def is_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self._enabled

    def record_timing(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        if not self._enabled:
            return

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetric(name)
            self._metrics[name].add_measurement(duration)

    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a specific metric."""
        with self._lock:
            return self._metrics.get(name)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance statistics."""
        with self._lock:
            return {name: metric.to_dict() for name, metric in self._metrics.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance statistics."""
        stats = self.get_stats()

        if not stats:
            return {"total_operations": 0, "total_time": 0.0}

        total_operations = sum(metric["call_count"] for metric in stats.values())
        total_time = sum(metric["total_time"] for metric in stats.values())

        # Find slowest operations
        slowest = sorted(
            [(name, metric["avg_time"]) for name, metric in stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Find most frequent operations
        most_frequent = sorted(
            [(name, metric["call_count"]) for name, metric in stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_operations": total_operations,
            "total_time": total_time,
            "avg_time_per_operation": (
                total_time / total_operations if total_operations > 0 else 0.0
            ),
            "slowest_operations": slowest,
            "most_frequent_operations": most_frequent,
            "unique_operations": len(stats),
        }

    def clear(self) -> None:
        """Clear all performance metrics."""
        with self._lock:
            self._metrics.clear()
        logger.debug("Cleared all performance metrics")

    def print_summary(self) -> None:
        """Print a formatted summary of performance statistics."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("🚀 SIFAKA PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"📊 Total Operations: {summary['total_operations']:,}")
        print(f"⏱️  Total Time: {summary['total_time']:.3f}s")
        print(f"📈 Average Time/Op: {summary['avg_time_per_operation']:.3f}s")
        print(f"🔧 Unique Operations: {summary['unique_operations']}")

        if summary["slowest_operations"]:
            print(f"\n🐌 Slowest Operations:")
            for name, avg_time in summary["slowest_operations"]:
                print(f"   {name}: {avg_time:.3f}s avg")

        if summary["most_frequent_operations"]:
            print(f"\n🔥 Most Frequent Operations:")
            for name, count in summary["most_frequent_operations"]:
                print(f"   {name}: {count:,} calls")

        print("=" * 60)


def timer(name: str) -> Callable[[F], F]:
    """Decorator for timing operations.

    Can be used as a decorator:
        @timer("my_operation")
        def my_function():
            pass

    Args:
        name: Name of the operation being timed.

    Returns:
        Decorator function.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                PerformanceMonitor.get_instance().record_timing(name, duration)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def time_operation(name: str) -> Generator[None, None, None]:
    """Context manager for timing operations.

    Args:
        name: Name of the operation being timed.

    Yields:
        None

    Example:
        with time_operation("database_query"):
            result = db.query("SELECT * FROM table")
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        PerformanceMonitor.get_instance().record_timing(name, duration)


class CacheStats:
    """Statistics for cache performance."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        with self._lock:
            self.evictions += 1

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.total_requests
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as a percentage."""
        return 100.0 - self.hit_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


# Global cache statistics registry
_cache_stats: Dict[str, CacheStats] = {}
_cache_stats_lock = threading.Lock()


def get_cache_stats(name: str) -> CacheStats:
    """Get or create cache statistics for a named cache."""
    with _cache_stats_lock:
        if name not in _cache_stats:
            _cache_stats[name] = CacheStats()
        return _cache_stats[name]


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all registered caches."""
    with _cache_stats_lock:
        return {name: stats.to_dict() for name, stats in _cache_stats.items()}


def clear_cache_stats() -> None:
    """Clear all cache statistics."""
    with _cache_stats_lock:
        for stats in _cache_stats.values():
            stats.reset()


# Convenience functions
def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return PerformanceMonitor.get_instance()


def enable_performance_monitoring() -> None:
    """Enable global performance monitoring."""
    PerformanceMonitor.get_instance().enable()


def disable_performance_monitoring() -> None:
    """Disable global performance monitoring."""
    PerformanceMonitor.get_instance().disable()


def print_performance_summary() -> None:
    """Print a summary of all performance statistics."""
    monitor = PerformanceMonitor.get_instance()
    monitor.print_summary()

    # Also print cache statistics if any exist
    cache_stats = get_all_cache_stats()
    if cache_stats:
        print(f"\n💾 CACHE STATISTICS")
        print("-" * 40)
        for name, stats in cache_stats.items():
            print(f"{name}:")
            print(f"  Hits: {stats['hits']:,} ({stats['hit_rate']:.1f}%)")
            print(f"  Misses: {stats['misses']:,} ({stats['miss_rate']:.1f}%)")
            print(f"  Evictions: {stats['evictions']:,}")
        print("-" * 40)
