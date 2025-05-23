#!/usr/bin/env python3
"""
Performance Monitoring Example for Sifaka

This example demonstrates how to use Sifaka's performance monitoring
capabilities to track timing and identify bottlenecks.

Run this example:
    python examples/mock/performance_example.py
"""

import time
from sifaka.utils.performance import (
    timer,
    time_operation,
    PerformanceMonitor,
    print_performance_summary,
    get_cache_stats,
)


# Simple performance demonstration functions


@timer("simple_operation")
def simple_operation() -> str:
    """A simple operation to time."""
    time.sleep(0.1)
    return "Operation completed"


@timer("math_operation")
def math_operation() -> int:
    """A math operation to time."""
    result = sum(i * i for i in range(1000))
    return result


def main() -> None:
    """Run the performance monitoring example."""
    print("🚀 Sifaka Performance Monitoring Example")
    print("=" * 50)

    try:
        # Enable performance monitoring
        monitor = PerformanceMonitor.get_instance()
        monitor.enable()

        # Run some operations
        print("🔧 Running timed operations...")
        result1 = simple_operation()
        print(f"  ✅ {result1}")

        result2 = math_operation()
        print(f"  ✅ Math result: {result2}")

        # Use context manager for timing
        with time_operation("context_operation"):
            time.sleep(0.05)
            print("  ✅ Context operation completed")

        print("\n" + "=" * 50)
        print("📊 PERFORMANCE REPORT")
        print("=" * 50)

        # Print performance summary
        print_performance_summary()

        print("\n🎉 Performance monitoring example completed successfully!")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        raise
    finally:
        # Clean up
        monitor = PerformanceMonitor.get_instance()
        monitor.clear()


if __name__ == "__main__":
    main()
