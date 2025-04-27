#!/usr/bin/env python3

import sys
import traceback
from sifaka.benchmarks.benchmark_classifiers import ClassifierBenchmark, print_benchmark_results


def main():
    try:
        print("Starting Sifaka Classifier Benchmarks...")

        # Initialize benchmark suite with smaller sample size for quicker testing
        benchmark = ClassifierBenchmark(num_samples=100, warm_up_rounds=2)

        # Run benchmarks without API key first (local classifiers only)
        print("\nRunning local classifier benchmarks...")
        results = benchmark.run_all_benchmarks()

        # Print results
        print_benchmark_results(results)

        print("\nBenchmarks completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError during benchmark execution: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
