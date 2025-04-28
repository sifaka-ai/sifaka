"""
Benchmarking suite for Sifaka rules.

This module provides tools to measure:
1. Individual rule performance
2. Rule pipeline performance
3. Memory usage
4. Throughput and latency
5. Rule validation effectiveness
"""

import gc
import statistics
import time
from typing import Any, Callable, Dict, List

import numpy as np
import psutil
from tqdm import tqdm

from sifaka.rules.base import Rule, RuleConfig, RuleResult, RulePriority
from sifaka.rules.length import LengthRule, LengthConfig, DefaultLengthValidator
from sifaka.rules.prohibited_content import ProhibitedContentRule
from sifaka.rules.pattern_rules import RepetitionRule, SymmetryRule
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class RuleBenchmark:
    """Benchmark suite for rule performance measurement."""

    def __init__(self, num_samples: int = 1000, warm_up_rounds: int = 3):
        """
        Initialize the benchmark suite.

        Args:
            num_samples: Number of text samples to use
            warm_up_rounds: Number of warm-up rounds before measurement
        """
        self.num_samples = num_samples
        self.warm_up_rounds = warm_up_rounds
        self._generate_test_data()
        self.rules = self._init_rules()
        self._warm_up_rules()

    def _init_rules(self) -> Dict[str, Rule]:
        """Initialize rules with proper validators and configs."""
        rules = {}

        # Initialize LengthRule with config
        length_config = LengthConfig(
            min_length=10, max_length=1000, unit="characters", cache_size=100, priority=1, cost=1.0
        )
        length_validator = DefaultLengthValidator(length_config)
        rules["length"] = LengthRule(
            name="length_rule",
            description="Validates text length",
            validator=length_validator,
            config={"min_length": 10, "max_length": 1000},
        )

        # Initialize ProhibitedContentRule with config
        rules["prohibited_content"] = ProhibitedContentRule(
            name="prohibited_content_rule",
            description="Checks for prohibited content",
            config={
                "prohibited_terms": ["bad", "terrible", "awful"],
                "case_sensitive": False,
                "cache_size": 100,
                "priority": 1,
                "cost": 1.0,
            },
        )

        # Initialize SymmetryRule with config
        rules["symmetry"] = SymmetryRule(
            name="symmetry_rule",
            description="Checks text symmetry",
            config=RuleConfig(
                priority=RulePriority.MEDIUM,
                cache_size=100,
                cost=1.0,
                metadata={
                    "mirror_mode": "horizontal",
                    "preserve_whitespace": True,
                    "preserve_case": True,
                    "ignore_punctuation": False,
                    "symmetry_threshold": 0.7,
                },
            ),
        )

        # Initialize RepetitionRule with config
        rules["repetition"] = RepetitionRule(
            name="repetition_rule",
            description="Checks for text repetition",
            config=RuleConfig(
                priority=RulePriority.MEDIUM,
                cache_size=100,
                cost=1.0,
                metadata={
                    "pattern_type": "repeat",
                    "pattern_length": 2,
                    "case_sensitive": True,
                    "allow_overlap": False,
                },
            ),
        )

        return rules

    def _generate_test_data(self) -> None:
        """Generate test data of varying complexity."""
        # Simple texts
        simple_texts = [
            "This is a short test.",
            "Another simple example.",
            "Testing rule performance.",
            "Basic text content.",
        ]

        # Complex texts with potential rule violations
        complex_texts = [
            "This text contains repeated words words and might trigger pattern rules.",
            "A very very very long sentence that goes on and on and might exceed length limits.",
            "Text with potential formatting issues.  Double spaces.   Triple spaces.",
            "Code-like content: def function(): pass # Python rule test",
        ]

        # Generate variations
        self.test_data = []
        for _ in range(self.num_samples):
            text_type = np.random.choice(["simple", "complex"])
            base_texts = simple_texts if text_type == "simple" else complex_texts
            selected_text = np.random.choice(base_texts)
            self.test_data.append(selected_text)

    def _warm_up_rules(self) -> None:
        """Warm up all rules."""
        for rule in self.rules.values():
            for _ in range(self.warm_up_rounds):
                rule.validate(self.test_data[0])

    def _measure_memory(self, func: Callable) -> Dict[str, float]:
        """Measure memory usage of a function."""
        process = psutil.Process()
        gc.collect()  # Force garbage collection

        # Measure before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run function
        func()

        # Measure after
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_diff_mb": mem_after - mem_before,
        }

    def benchmark_single_rule(self, rule: Rule, sample_size: int = 100) -> Dict[str, Any]:
        """
        Benchmark a single rule.

        Args:
            rule: Rule instance to benchmark
            sample_size: Number of samples to use

        Returns:
            Dictionary with benchmark results
        """
        # Warm up
        for _ in range(self.warm_up_rounds):
            rule.validate(self.test_data[0])

        # Measure performance
        latencies = []
        results = []

        for text in tqdm(self.test_data[:sample_size], desc=f"Benchmarking {rule.name}"):
            start_time = time.perf_counter()
            result = rule.validate(text)
            end_time = time.perf_counter()

            latencies.append(end_time - start_time)
            results.append(result)

        # Calculate statistics
        stats = {
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies),
            "throughput": sample_size / sum(latencies),
            "sample_size": sample_size,
            "validation_rate": sum(1 for r in results if r.passed) / sample_size,
        }

        # Measure memory usage
        mem_stats = self._measure_memory(
            lambda: [rule.validate(text) for text in self.test_data[:10]]
        )
        stats.update(mem_stats)

        return stats

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks and return results.

        Returns:
            Dictionary with all benchmark results
        """
        results = {}
        for name, rule in self.rules.items():
            results[name] = self.benchmark_single_rule(rule)
        return results


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print("\n=== Sifaka Rule Benchmarks ===\n")

    for rule_name, stats in results.items():
        print(f"\n{rule_name.upper()} Rule:")
        print("-" * 40)
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"P99 Latency: {stats['p99_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")
        print(f"Validation Rate: {stats['validation_rate']*100:.1f}%")


if __name__ == "__main__":
    # Initialize benchmark suite
    benchmark = RuleBenchmark(num_samples=1000)

    # Run benchmarks
    results = benchmark.run_all_benchmarks()

    # Print results
    print_benchmark_results(results)
