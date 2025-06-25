"""Performance metrics collection example for Sifaka.

This example shows how to collect detailed performance metrics including:
- Memory usage tracking
- CPU profiling
- Request timing
- Resource utilization
- Bottleneck identification
"""

import time
import psutil
import gc
import tracemalloc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import cProfile
import pstats
import io

from sifaka import improve_sync, improve_async
from sifaka.core.middleware import BaseMiddleware
from sifaka.core.models import SifakaResult


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    duration_seconds: float
    memory_used_mb: float
    memory_peak_mb: float
    cpu_percent: float
    tokens_per_second: float
    iterations_per_minute: float
    gc_collections: Dict[int, int]
    thread_count: int
    profile_stats: Optional[str] = None


class PerformanceMonitoringMiddleware(BaseMiddleware):
    """Middleware to collect detailed performance metrics."""

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_profiling: bool = False,
        enable_gc_stats: bool = True,
        profile_top_n: int = 10,
    ):
        """Initialize performance monitoring.

        Args:
            enable_memory_tracking: Track memory allocation
            enable_cpu_profiling: Enable CPU profiling (adds overhead)
            enable_gc_stats: Track garbage collection
            profile_top_n: Number of top functions to show in profile
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_gc_stats = enable_gc_stats
        self.profile_top_n = profile_top_n

        # Start memory tracking if enabled
        if self.enable_memory_tracking:
            tracemalloc.start()

        # Storage for metrics
        self.metrics_history: List[PerformanceMetrics] = []

    async def pre_improve(self, text: str, **kwargs) -> Dict[str, Any]:
        """Start performance monitoring."""
        context = {
            "start_time": time.time(),
            "start_memory": None,
            "gc_stats_start": None,
            "profiler": None,
            "process": psutil.Process(),
            "text_length": len(text),
        }

        # Track memory
        if self.enable_memory_tracking:
            context["start_memory"] = tracemalloc.get_traced_memory()

        # Track GC
        if self.enable_gc_stats:
            context["gc_stats_start"] = gc.get_stats()

        # Start CPU profiling
        if self.enable_cpu_profiling:
            profiler = cProfile.Profile()
            profiler.enable()
            context["profiler"] = profiler

        # CPU usage baseline
        context["process"].cpu_percent()  # First call to initialize

        return context

    async def post_improve(
        self,
        result: SifakaResult,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
    ):
        """Collect performance metrics after improvement."""
        # Calculate duration
        duration = time.time() - context["start_time"]

        # Memory metrics
        memory_used_mb = 0
        memory_peak_mb = 0
        if self.enable_memory_tracking and context["start_memory"]:
            current, peak = tracemalloc.get_traced_memory()
            start_current, start_peak = context["start_memory"]
            memory_used_mb = (current - start_current) / 1024 / 1024
            memory_peak_mb = peak / 1024 / 1024

        # CPU usage
        cpu_percent = context["process"].cpu_percent()

        # GC collections
        gc_collections = {}
        if self.enable_gc_stats:
            for i in range(gc.get_count().__len__()):
                gc_collections[i] = gc.get_count()[i]

        # Profile stats
        profile_stats = None
        if self.enable_cpu_profiling and context["profiler"]:
            context["profiler"].disable()
            s = io.StringIO()
            ps = pstats.Stats(context["profiler"], stream=s)
            ps.strip_dirs().sort_stats("cumulative").print_stats(self.profile_top_n)
            profile_stats = s.getvalue()

        # Calculate derived metrics
        tokens_per_second = result.total_tokens / duration if not error else 0
        iterations_per_minute = (result.iterations / duration * 60) if not error else 0

        # Create metrics object
        metrics = PerformanceMetrics(
            duration_seconds=duration,
            memory_used_mb=memory_used_mb,
            memory_peak_mb=memory_peak_mb,
            cpu_percent=cpu_percent,
            tokens_per_second=tokens_per_second,
            iterations_per_minute=iterations_per_minute,
            gc_collections=gc_collections,
            thread_count=threading.active_count(),
            profile_stats=profile_stats,
        )

        # Store metrics
        self.metrics_history.append(metrics)

        # Log metrics
        self._log_metrics(metrics, result, error)

    def _log_metrics(
        self,
        metrics: PerformanceMetrics,
        result: SifakaResult,
        error: Optional[Exception],
    ):
        """Log performance metrics."""
        print("\n=== Performance Metrics ===")
        print(f"Duration: {metrics.duration_seconds:.2f}s")
        print(f"Memory Used: {metrics.memory_used_mb:.2f} MB")
        print(f"Memory Peak: {metrics.memory_peak_mb:.2f} MB")
        print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
        print(f"Tokens/sec: {metrics.tokens_per_second:.1f}")
        print(f"Iterations/min: {metrics.iterations_per_minute:.1f}")
        print(f"Thread Count: {metrics.thread_count}")
        print(f"GC Collections: {metrics.gc_collections}")

        if metrics.profile_stats and self.enable_cpu_profiling:
            print(f"\n=== CPU Profile (Top {self.profile_top_n}) ===")
            print(metrics.profile_stats)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all monitored operations."""
        if not self.metrics_history:
            return {}

        durations = [m.duration_seconds for m in self.metrics_history]
        memory_used = [m.memory_used_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_percent for m in self.metrics_history]
        tokens_per_sec = [m.tokens_per_second for m in self.metrics_history]

        return {
            "total_operations": len(self.metrics_history),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations),
            },
            "memory_mb": {
                "min": min(memory_used),
                "max": max(memory_used),
                "avg": sum(memory_used) / len(memory_used),
            },
            "cpu_percent": {
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "avg": sum(cpu_usage) / len(cpu_usage),
            },
            "throughput": {
                "tokens_per_second_avg": sum(tokens_per_sec) / len(tokens_per_sec)
            },
        }


class ResourceLimiter:
    """Limit resource usage for Sifaka operations."""

    def __init__(
        self,
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
    ):
        """Initialize resource limiter.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            max_duration_seconds: Maximum operation duration
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_duration_seconds = max_duration_seconds
        self.process = psutil.Process()

    @contextmanager
    def limit(self):
        """Context manager to enforce resource limits."""
        start_time = time.time()

        # Monitor in background thread
        stop_monitoring = threading.Event()
        violation = threading.Event()

        def monitor():
            while not stop_monitoring.is_set():
                # Check memory
                if self.max_memory_mb:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    if memory_mb > self.max_memory_mb:
                        violation.set()
                        break

                # Check CPU
                if self.max_cpu_percent:
                    cpu = self.process.cpu_percent(interval=0.1)
                    if cpu > self.max_cpu_percent:
                        violation.set()
                        break

                # Check duration
                if self.max_duration_seconds:
                    if time.time() - start_time > self.max_duration_seconds:
                        violation.set()
                        break

                time.sleep(0.1)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        try:
            yield
        finally:
            stop_monitoring.set()
            monitor_thread.join()

            if violation.is_set():
                raise ResourceError("Resource limit exceeded")


class ResourceError(Exception):
    """Raised when resource limits are exceeded."""

    pass


# Memory profiling utilities
def profile_memory_usage(func):
    """Decorator to profile memory usage of a function."""

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_current, start_peak = tracemalloc.get_traced_memory()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nMemory usage for {func.__name__}:")
        print(f"  Current: {(current - start_current) / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

        return result

    return wrapper


# Performance benchmarking
class PerformanceBenchmark:
    """Run performance benchmarks for Sifaka."""

    def __init__(self, performance_middleware: PerformanceMonitoringMiddleware):
        self.middleware = performance_middleware

    async def benchmark_critics(self, text: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark different critics."""
        critics = ["reflexion", "self_refine", "chain_of_thought", "constitutional_ai"]
        results = {}

        for critic in critics:
            print(f"\nBenchmarking {critic}...")
            critic_metrics = []

            for i in range(iterations):
                try:
                    result = await improve_async(
                        text,
                        critics=[critic],
                        max_iterations=2,
                        middleware=[self.middleware],
                    )

                    if self.middleware.metrics_history:
                        critic_metrics.append(self.middleware.metrics_history[-1])

                except Exception as e:
                    print(f"  Error in iteration {i+1}: {e}")

            if critic_metrics:
                results[critic] = {
                    "avg_duration": sum(m.duration_seconds for m in critic_metrics)
                    / len(critic_metrics),
                    "avg_memory": sum(m.memory_used_mb for m in critic_metrics)
                    / len(critic_metrics),
                    "avg_tokens_per_sec": sum(
                        m.tokens_per_second for m in critic_metrics
                    )
                    / len(critic_metrics),
                }

        return results

    def benchmark_text_sizes(
        self, sizes: List[int] = [100, 500, 1000, 5000]
    ) -> Dict[int, Any]:
        """Benchmark performance with different text sizes."""
        results = {}
        base_text = "This is a sample text that will be repeated. "

        for size in sizes:
            # Create text of approximate size
            text = base_text * (size // len(base_text))
            print(f"\nBenchmarking text size: {len(text)} chars")

            try:
                result = improve_sync(
                    text,
                    critics=["reflexion"],
                    max_iterations=2,
                    middleware=[self.middleware],
                )

                if self.middleware.metrics_history:
                    metrics = self.middleware.metrics_history[-1]
                    results[size] = {
                        "duration": metrics.duration_seconds,
                        "memory": metrics.memory_used_mb,
                        "tokens_per_sec": metrics.tokens_per_second,
                    }

            except Exception as e:
                print(f"  Error: {e}")
                results[size] = {"error": str(e)}

        return results


# Example usage
if __name__ == "__main__":
    # Create performance monitoring middleware
    perf_middleware = PerformanceMonitoringMiddleware(
        enable_memory_tracking=True, enable_cpu_profiling=True, enable_gc_stats=True
    )

    # Example 1: Monitor a single improvement
    print("=== Example 1: Single Improvement Monitoring ===")
    text = "AI is transforming how we work and live."

    result = improve_sync(
        text, critics=["reflexion"], max_iterations=2, middleware=[perf_middleware]
    )

    print(f"\nOriginal: {text}")
    print(f"Improved: {result.final_text}")

    # Example 2: Resource-limited operation
    print("\n=== Example 2: Resource-Limited Operation ===")
    limiter = ResourceLimiter(
        max_memory_mb=500, max_cpu_percent=80, max_duration_seconds=30
    )

    try:
        with limiter.limit():
            result = improve_sync(
                "Climate change requires urgent action from all of us.",
                critics=["self_refine"],
                max_iterations=3,
                middleware=[perf_middleware],
            )
            print("Operation completed within resource limits")
    except ResourceError as e:
        print(f"Resource limit exceeded: {e}")

    # Example 3: Memory profiling
    print("\n=== Example 3: Memory Profiling ===")

    @profile_memory_usage
    def memory_intensive_improvement():
        texts = [
            "Python is a versatile programming language.",
            "Machine learning models learn from data.",
            "Cloud computing provides scalable resources.",
        ] * 10  # Process multiple texts

        results = []
        for text in texts:
            result = improve_sync(text, critics=["chain_of_thought"], max_iterations=1)
            results.append(result)

        return results

    # Run memory profiled function
    memory_intensive_improvement()

    # Example 4: Performance benchmarking
    print("\n=== Example 4: Performance Benchmarking ===")
    benchmark = PerformanceBenchmark(perf_middleware)

    # Benchmark text sizes
    size_results = benchmark.benchmark_text_sizes([100, 500, 1000])

    print("\nText Size Benchmark Results:")
    for size, metrics in size_results.items():
        if "error" not in metrics:
            print(
                f"  {size} chars: {metrics['duration']:.2f}s, "
                f"{metrics['memory']:.2f}MB, "
                f"{metrics['tokens_per_sec']:.1f} tokens/sec"
            )

    # Get summary statistics
    print("\n=== Summary Statistics ===")
    summary = perf_middleware.get_summary_stats()
    print(f"Total operations: {summary.get('total_operations', 0)}")
    if "duration" in summary:
        print(
            f"Duration - Min: {summary['duration']['min']:.2f}s, "
            f"Max: {summary['duration']['max']:.2f}s, "
            f"Avg: {summary['duration']['avg']:.2f}s"
        )

    # Example 5: Async performance comparison
    print("\n=== Example 5: Sync vs Async Performance ===")

    async def compare_sync_async():
        text = "Renewable energy is the key to a sustainable future."

        # Sync version
        start = time.time()
        sync_result = improve_sync(text, critics=["reflexion"])
        sync_duration = time.time() - start

        # Async version
        start = time.time()
        async_result = await improve_async(text, critics=["reflexion"])
        async_duration = time.time() - start

        print(f"Sync duration: {sync_duration:.2f}s")
        print(f"Async duration: {async_duration:.2f}s")
        print(f"Speedup: {sync_duration / async_duration:.2f}x")

    # Run async comparison
    # asyncio.run(compare_sync_async())
