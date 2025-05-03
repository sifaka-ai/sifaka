"""
Performance monitoring for Sifaka.

This module provides components for monitoring and analyzing performance metrics. It includes:
- TimingStats: A class for tracking timing statistics
- PerformanceMonitor: A class for monitoring various performance aspects

The monitoring system tracks:
1. Timing statistics for different operations
2. Success/failure rates
3. Cache hit/miss rates
4. Performance metrics over time

Example:
    ```python
    from sifaka.monitoring import PerformanceMonitor

    # Create a monitor
    monitor = PerformanceMonitor()

    # Record timing for an operation
    start = time.time()
    # ... perform operation ...
    monitor.record_rule_time("my_rule", time.time() - start)

    # Get and log metrics
    metrics = monitor.get_metrics()
    monitor.log_metrics()
    ```
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
    """
    Statistics for timing measurements.

    This class provides functionality to track and analyze timing statistics,
    including:
    - Total count of measurements
    - Total time across all measurements
    - Minimum and maximum times
    - Recent time history
    - Average calculations

    Attributes:
        count: Total number of measurements
        total_time: Sum of all measurement times
        min_time: Minimum measurement time
        max_time: Maximum measurement time
        times: Deque of recent measurements (max 100)

    Example:
        ```python
        stats = TimingStats()
        stats.add_measurement(1.5)
        stats.add_measurement(2.0)
        print(f"Average: {stats.avg_time}")
        print(f"Recent average: {stats.recent_avg_time}")
        ```
    """

    count: int = 0
    total_time: float = 0.0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_measurement(self, time: float) -> None:
        """
        Add a timing measurement.

        This method:
        1. Updates the total count
        2. Adds to the total time
        3. Updates min/max times if needed
        4. Adds to the recent times deque

        Args:
            time: The time measurement to add in seconds

        Raises:
            ValueError: If time is negative
        """
        if time < 0:
            raise ValueError("Time measurement cannot be negative")
        self.count += 1
        self.total_time += time
        self.times.append(time)

        if self.min_time is None or time < self.min_time:
            self.min_time = time
        if self.max_time is None or time > self.max_time:
            self.max_time = time

    @property
    def avg_time(self) -> float:
        """
        Calculate average time across all measurements.

        Returns:
            Average time in seconds, or 0.0 if no measurements
        """
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """
        Calculate average of recent times (last 100 measurements).

        Returns:
            Average of recent times in seconds, or 0.0 if no measurements
        """
        return statistics.mean(self.times) if self.times else 0.0


class PerformanceMonitor:
    """
    Performance monitoring class for tracking various metrics.

    This class is responsible for:
    1. Tracking timing statistics for different operations
    2. Monitoring success/failure rates
    3. Tracking cache performance
    4. Providing access to aggregated metrics

    The monitor follows a simple workflow:
    1. Initialize the monitor
    2. Record various metrics during operation
    3. Access metrics or log them as needed

    Example:
        ```python
        monitor = PerformanceMonitor()
        monitor.record_rule_time("my_rule", 1.5)
        monitor.record_attempt(True)
        metrics = monitor.get_metrics()
        ```
    """

    def __init__(self):
        """
        Initialize a PerformanceMonitor instance.

        This method sets up:
        1. Thread-safe storage for metrics
        2. Initial counters and statistics
        3. Lock for thread safety
        """
        self._lock = threading.Lock()
        self._rule_times: Dict[str, TimingStats] = {}
        self._generation_time = TimingStats()
        self._critique_time = TimingStats()
        self._validation_time = TimingStats()
        self._attempts = {"success": 0, "failure": 0}
        self._cache_stats = {"hits": 0, "misses": 0}

    def record_rule_time(self, rule_name: str, elapsed: float) -> None:
        """
        Record timing for a specific rule.

        Args:
            rule_name: Name of the rule being timed
            elapsed: Time elapsed in seconds

        Raises:
            ValueError: If rule_name is empty or elapsed is negative
        """
        if not rule_name:
            raise ValueError("Rule name cannot be empty")
        if elapsed < 0:
            raise ValueError("Elapsed time cannot be negative")

        with self._lock:
            if rule_name not in self._rule_times:
                self._rule_times[rule_name] = TimingStats()
            self._rule_times[rule_name].add_measurement(elapsed)

    def record_generation_time(self, elapsed: float) -> None:
        """
        Record timing for generation operations.

        Args:
            elapsed: Time elapsed in seconds

        Raises:
            ValueError: If elapsed is negative
        """
        if elapsed < 0:
            raise ValueError("Elapsed time cannot be negative")
        with self._lock:
            self._generation_time.add_measurement(elapsed)

    def record_critique_time(self, elapsed: float) -> None:
        """
        Record timing for critique operations.

        Args:
            elapsed: Time elapsed in seconds

        Raises:
            ValueError: If elapsed is negative
        """
        if elapsed < 0:
            raise ValueError("Elapsed time cannot be negative")
        with self._lock:
            self._critique_time.add_measurement(elapsed)

    def record_validation_time(self, elapsed: float) -> None:
        """
        Record timing for validation operations.

        Args:
            elapsed: Time elapsed in seconds

        Raises:
            ValueError: If elapsed is negative
        """
        if elapsed < 0:
            raise ValueError("Elapsed time cannot be negative")
        with self._lock:
            self._validation_time.add_measurement(elapsed)

    def record_attempt(self, success: bool) -> None:
        """
        Record a success or failure attempt.

        Args:
            success: True if the attempt was successful, False otherwise
        """
        with self._lock:
            if success:
                self._attempts["success"] += 1
            else:
                self._attempts["failure"] += 1

    def record_cache_access(self, hit: bool) -> None:
        """
        Record a cache hit or miss.

        Args:
            hit: True if it was a cache hit, False for a miss
        """
        with self._lock:
            if hit:
                self._cache_stats["hits"] += 1
            else:
                self._cache_stats["misses"] += 1

    def get_metrics(self) -> Dict:
        """
        Get all current performance metrics.

        This method aggregates all metrics into a single dictionary,
        including:
        1. Rule timing statistics
        2. Generation, critique, and validation times
        3. Success/failure rates
        4. Cache hit/miss rates

        Returns:
            Dictionary containing all performance metrics
        """
        with self._lock:
            metrics = {
                "rule_times": {
                    name: {
                        "count": stats.count,
                        "total_time": stats.total_time,
                        "avg_time": stats.avg_time,
                        "min_time": stats.min_time,
                        "max_time": stats.max_time,
                        "recent_avg_time": stats.recent_avg_time,
                    }
                    for name, stats in self._rule_times.items()
                },
                "generation_time": {
                    "count": self._generation_time.count,
                    "total_time": self._generation_time.total_time,
                    "avg_time": self._generation_time.avg_time,
                    "min_time": self._generation_time.min_time,
                    "max_time": self._generation_time.max_time,
                    "recent_avg_time": self._generation_time.recent_avg_time,
                },
                "critique_time": {
                    "count": self._critique_time.count,
                    "total_time": self._critique_time.total_time,
                    "avg_time": self._critique_time.avg_time,
                    "min_time": self._critique_time.min_time,
                    "max_time": self._critique_time.max_time,
                    "recent_avg_time": self._critique_time.recent_avg_time,
                },
                "validation_time": {
                    "count": self._validation_time.count,
                    "total_time": self._validation_time.total_time,
                    "avg_time": self._validation_time.avg_time,
                    "min_time": self._validation_time.min_time,
                    "max_time": self._validation_time.max_time,
                    "recent_avg_time": self._validation_time.recent_avg_time,
                },
                "attempts": self._attempts.copy(),
                "cache_stats": self._cache_stats.copy(),
            }
            return metrics

    def log_metrics(self) -> None:
        """
        Log all current performance metrics.

        This method logs all metrics using the configured logger,
        including:
        1. Rule timing statistics
        2. Generation, critique, and validation times
        3. Success/failure rates
        4. Cache hit/miss rates

        The metrics are logged at INFO level with appropriate formatting.
        """
        metrics = self.get_metrics()
        logger.info("Performance Metrics:")
        logger.info("Rule Times:")
        for name, stats in metrics["rule_times"].items():
            logger.info(f"  {name}:")
            logger.info(f"    Count: {stats['count']}")
            logger.info(f"    Total Time: {stats['total_time']:.2f}s")
            logger.info(f"    Average Time: {stats['avg_time']:.2f}s")
            logger.info(f"    Min Time: {stats['min_time']:.2f}s")
            logger.info(f"    Max Time: {stats['max_time']:.2f}s")
            logger.info(f"    Recent Average: {stats['recent_avg_time']:.2f}s")

        logger.info("Generation Time:")
        logger.info(f"  Count: {metrics['generation_time']['count']}")
        logger.info(f"  Total Time: {metrics['generation_time']['total_time']:.2f}s")
        logger.info(f"  Average Time: {metrics['generation_time']['avg_time']:.2f}s")
        logger.info(f"  Min Time: {metrics['generation_time']['min_time']:.2f}s")
        logger.info(f"  Max Time: {metrics['generation_time']['max_time']:.2f}s")
        logger.info(f"  Recent Average: {metrics['generation_time']['recent_avg_time']:.2f}s")

        logger.info("Critique Time:")
        logger.info(f"  Count: {metrics['critique_time']['count']}")
        logger.info(f"  Total Time: {metrics['critique_time']['total_time']:.2f}s")
        logger.info(f"  Average Time: {metrics['critique_time']['avg_time']:.2f}s")
        logger.info(f"  Min Time: {metrics['critique_time']['min_time']:.2f}s")
        logger.info(f"  Max Time: {metrics['critique_time']['max_time']:.2f}s")
        logger.info(f"  Recent Average: {metrics['critique_time']['recent_avg_time']:.2f}s")

        logger.info("Validation Time:")
        logger.info(f"  Count: {metrics['validation_time']['count']}")
        logger.info(f"  Total Time: {metrics['validation_time']['total_time']:.2f}s")
        logger.info(f"  Average Time: {metrics['validation_time']['avg_time']:.2f}s")
        logger.info(f"  Min Time: {metrics['validation_time']['min_time']:.2f}s")
        logger.info(f"  Max Time: {metrics['validation_time']['max_time']:.2f}s")
        logger.info(f"  Recent Average: {metrics['validation_time']['recent_avg_time']:.2f}s")

        logger.info("Attempts:")
        logger.info(f"  Success: {metrics['attempts']['success']}")
        logger.info(f"  Failure: {metrics['attempts']['failure']}")

        logger.info("Cache Stats:")
        logger.info(f"  Hits: {metrics['cache_stats']['hits']}")
        logger.info(f"  Misses: {metrics['cache_stats']['misses']}")
