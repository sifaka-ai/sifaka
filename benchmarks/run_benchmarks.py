#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the parent directory to the Python path to find the sifaka package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from benchmark_classifiers import (
    ClassifierBenchmark,
    print_benchmark_results as print_classifier_results,
)
from benchmark_rules import RuleBenchmark, print_benchmark_results as print_rule_results


def main():
    # Initialize benchmark suites
    print("Initializing benchmark suites...")
    classifier_benchmark = ClassifierBenchmark(num_samples=1000, warm_up_rounds=3)
    rule_benchmark = RuleBenchmark(num_samples=1000, warm_up_rounds=3)

    # Run classifier benchmarks
    print("\nRunning classifier benchmarks...")
    classifier_results = classifier_benchmark.run_all_benchmarks()
    print_classifier_results(classifier_results)

    # Run rule benchmarks
    print("\nRunning rule benchmarks...")
    rule_results = rule_benchmark.run_all_benchmarks()
    print_rule_results(rule_results)


if __name__ == "__main__":
    main()
