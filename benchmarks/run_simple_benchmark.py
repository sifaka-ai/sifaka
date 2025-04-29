#!/usr/bin/env python3

"""
Simplified benchmark runner that focuses on basic rule benchmarks.
"""

from benchmark_rules import RuleBenchmark
from sifaka.rules.content.prohibited import ProhibitedContentRule
from sifaka.rules.length import LengthRule
from sifaka.rules.base import RuleConfig


def main():
    # Initialize benchmark suite with minimal configuration
    benchmark = RuleBenchmark(
        num_samples=100, warm_up_rounds=2  # Reduced sample size for quicker testing
    )

    # Override the rules with our minimal set
    benchmark.rules = {}  # Clear existing rules

    # Add length rule
    benchmark.rules["length"] = LengthRule(
        name="length_rule",
        description="Validates text length",
        config={"min_length": 10, "max_length": 1000},
    )

    # Add prohibited content rule
    benchmark.rules["prohibited_content"] = ProhibitedContentRule(
        name="prohibited_content_rule",
        description="Checks for prohibited content",
        config=RuleConfig(params={"terms": ["bad", "worse", "worst"], "case_sensitive": False}),
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
