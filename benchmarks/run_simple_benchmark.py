#!/usr/bin/env python3

"""
Simplified benchmark runner that focuses on basic rule benchmarks.
"""

from benchmark_rules import RuleBenchmark
from sifaka.rules.content.prohibited import create_prohibited_content_rule
from sifaka.rules.formatting.length import create_length_rule


def main():
    # Initialize benchmark suite with minimal configuration
    benchmark = RuleBenchmark(
        num_samples=100, warm_up_rounds=2  # Reduced sample size for quicker testing
    )

    # Override the rules with our minimal set
    benchmark.rules = {}  # Clear existing rules

    # Use factory functions to create rules
    benchmark.rules["length"] = create_length_rule(
        min_chars=10,
        max_chars=1000,
        rule_id="length_rule"
    )

    benchmark.rules["prohibited_content"] = create_prohibited_content_rule(
        terms=["bad", "worse", "worst"],
        case_sensitive=False,
        name="prohibited_content_rule"
    )

    # Run benchmarks
    print("\nRunning rule benchmarks...")
    results = benchmark.run_all_benchmarks()

    # Print results
    print("\nBenchmark Results:")
    print("=" * 80)
    for rule_name, stats in results.items():
        print(f"\n{rule_name}:")
        print("-" * 40)
        if isinstance(stats, dict) and "error" in stats:
            print(f"Error: {stats['error']}")
            continue
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")


if __name__ == "__main__":
    main()
