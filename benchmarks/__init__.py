"""Performance benchmarks for Sifaka.

This package provides comprehensive benchmarking tools for:
- Core operations (improve, critique, validate)
- Scalability testing
- Memory usage analysis
- Storage performance
- Concurrent operations

Usage:
    python -m pytest benchmarks/ -v
    python -m benchmarks.performance_benchmarks run
"""

from .performance_benchmarks import PerformanceBenchmark

__all__ = ["PerformanceBenchmark"]
