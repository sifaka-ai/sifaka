#!/usr/bin/env python3

"""
Simplified benchmark for measuring rule performance.
"""

import gc
import statistics
import time
from typing import Dict, Any
import numpy as np
import psutil
from tqdm import tqdm

from sifaka.rules.prohibited_content import ProhibitedContentRule
from sifaka.rules.length import LengthRule


def measure_memory(func):
    """Measure memory usage of a function."""
    process = psutil.Process()
    gc.collect()

    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    result = func()
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    return result, {
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_diff_mb": mem_after - mem_before,
    }


def benchmark_rule(rule, test_data: list[str], num_samples: int = 100) -> Dict[str, Any]:
    """Benchmark a single rule."""
    latencies = []
    results = []

    # Warm up
    for _ in range(2):
        rule.validate(test_data[0])

    for text in tqdm(test_data[:num_samples], desc=f"Benchmarking {rule.name}"):
        start_time = time.perf_counter()
        result = rule.validate(text)
        end_time = time.perf_counter()

        latencies.append(end_time - start_time)
        results.append(result)

    stats = {
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": num_samples / sum(latencies),
        "sample_size": num_samples,
    }

    # Add memory stats
    _, mem_stats = measure_memory(lambda: rule.validate(test_data[0]))
    stats.update(mem_stats)

    return stats


def main():
    # Generate test data
    test_data = [
        "This is a short test.",
        "Another simple example that might be too long.",
        "Testing rule performance with various inputs.",
        "Some bad content that should be flagged.",
        "Text with worse content for testing.",
        "A very long text that goes on and on and might exceed length limits.",
    ] * 20  # Multiply to get more samples

    # Initialize rules with proper configuration
    rules = {
        "length": LengthRule(
            name="length_rule",
            description="Validates text length",
            config={"min_length": 10, "max_length": 100},
        ),
        "prohibited_content": ProhibitedContentRule(
            name="prohibited_content_rule",
            description="Checks for prohibited content",
            config={"prohibited_terms": {"bad", "worse", "worst"}},
        ),
    }

    # Run benchmarks
    print("\nRunning rule benchmarks...")
    results = {}
    for name, rule in rules.items():
        try:
            results[name] = benchmark_rule(rule, test_data)
        except Exception as e:
            results[name] = {"error": str(e)}

    # Print results
    print("\nBenchmark Results:")
    print("=" * 80)
    for rule_name, stats in results.items():
        print(f"\n{rule_name}:")
        print("-" * 40)
        if "error" in stats:
            print(f"Error: {stats['error']}")
            continue
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")


if __name__ == "__main__":
    main()
