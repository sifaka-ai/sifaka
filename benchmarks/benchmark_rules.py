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

from sifaka.rules.base import Rule
from sifaka.rules.length import LengthRule
from sifaka.rules.prohibited_content import ProhibitedContentRule
from sifaka.rules.safety import ToxicityRule, BiasRule
from sifaka.rules.classifier_rule import ClassifierRule
from sifaka.rules.formatting import ParagraphRule, StyleRule
from sifaka.rules.pattern_rules import RepetitionRule, SymmetryRule
from sifaka.rules.domain import PythonRule

from sifaka.classifiers.sentiment import SentimentClassifier
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
        self._init_rules()

    def _generate_test_data(self) -> None:
        """Generate test data of varying complexity and characteristics."""
        # Simple texts
        simple_texts = [
            "The cat sat on the mat.",
            "It was a sunny day.",
            "The bird flew by quickly.",
            "She likes to read books.",
        ]

        # Medium complexity texts
        medium_texts = [
            "The photosynthesis process enables plants to convert sunlight into energy.",
            "The implementation of quantum algorithms requires sophisticated understanding.",
            "Climate change affects global weather patterns significantly.",
            "Modern architecture combines aesthetics with functional design principles.",
        ]

        # Complex texts
        complex_texts = [
            "The ontological implications of quantum entanglement challenge our understanding.",
            "Post-modern deconstructionist theory exemplifies paradigm shifts in analysis.",
            "The epistemological framework underlying cognitive science remains debatable.",
            "Theoretical physics postulates multiple dimensional spaces beyond observation.",
        ]

        # Texts with potential issues for rules to catch
        problematic_texts = [
            "This text repeats repeats words unnecessarily.",
            "This text has some inappropriate or offensive language like damn and hell.",
            "This text is extremely short.",
            "This extremely long text goes on and on with unnecessary details and redundant information that doesn't add value but simply extends the length beyond what is reasonable for efficient communication in most contexts, creating cognitive burden for readers who must wade through excessive verbiage.",
            "This text has some factual errors: the earth is flat and the moon is made of cheese.",
            "I absolutely hate this terrible product! It's the worst thing ever made!!",
            "def broken_code():\n  print('Hello world\n  return None",
            "The patient should take 100mg of medication X twice daily with food.",
            "According to the ruling in Brown v. Board of Education, segregation was constitutional.",
        ]

        # Generate variations with different lengths and combinations
        self.test_data = []
        for _ in range(self.num_samples):
            text_type = np.random.choice(["simple", "medium", "complex", "problematic"])
            if text_type == "simple":
                base_texts = simple_texts
            elif text_type == "medium":
                base_texts = medium_texts
            elif text_type == "complex":
                base_texts = complex_texts
            else:
                base_texts = problematic_texts

            # Randomly combine 1-3 sentences
            num_sentences = np.random.randint(1, 4)
            selected_texts = np.random.choice(base_texts, num_sentences)
            self.test_data.append(" ".join(selected_texts))

    def _init_rules(self) -> None:
        """Initialize all rules."""
        # Content-based rules
        self.length_rule = LengthRule(
            name="benchmark_length",
            description="Validates text length",
            config={"min_length": 10, "max_length": 1000},
        )

        self.prohibited_content_rule = ProhibitedContentRule(
            name="benchmark_prohibited",
            description="Checks for prohibited terms",
            config={"prohibited_terms": ["inappropriate", "offensive", "damn", "hell"]},
        )

        # Formatting rules
        self.paragraph_rule = ParagraphRule(
            name="benchmark_paragraph",
            description="Validates paragraph structure",
            config={"max_paragraph_length": 150},
        )

        self.style_rule = StyleRule(
            name="benchmark_style",
            description="Validates text style",
            config={"allowed_styles": ["formal", "informative"]},
        )

        # Pattern rules
        self.repetition_rule = RepetitionRule(
            name="benchmark_repetition",
            description="Detects repeated patterns",
            config={"max_repetitions": 2},
        )

        self.symmetry_rule = SymmetryRule(
            name="benchmark_symmetry",
            description="Checks for text symmetry",
            config={"symmetry_threshold": 0.7},
        )

        # Safety rules
        self.toxicity_rule = ToxicityRule(
            name="benchmark_toxicity",
            description="Checks for toxic content",
            config={"toxicity_threshold": 0.7},
        )

        self.bias_rule = BiasRule(
            name="benchmark_bias", description="Checks for bias", config={"bias_threshold": 0.7}
        )

        # Domain-specific rules
        self.python_rule = PythonRule(
            name="benchmark_python",
            description="Python code validation",
            config={"strict_mode": True},
        )

        # Classifier-based rules
        self.sentiment_classifier = SentimentClassifier(
            name="benchmark_sentiment_classifier", description="Sentiment analysis"
        )

        self.classifier_rule = ClassifierRule(
            name="benchmark_classifier_rule",
            description="Rule using sentiment classifier",
            classifier=self.sentiment_classifier,
            config={"threshold": 0.7, "valid_labels": ["positive", "neutral"]},
        )

        # Warm up all rules
        rules = [
            self.length_rule,
            self.prohibited_content_rule,
            self.paragraph_rule,
            self.style_rule,
            self.repetition_rule,
            self.symmetry_rule,
            self.toxicity_rule,
            self.bias_rule,
            self.python_rule,
            self.classifier_rule,
        ]

        for rule in rules:
            # Call validate on a sample text to warm up the rule
            for _ in range(self.warm_up_rounds):
                try:
                    rule.validate(self.test_data[0])
                except Exception as e:
                    logger.warning(f"Error warming up rule {rule.name}: {str(e)}")

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
            try:
                rule.validate(self.test_data[0])
            except Exception as e:
                logger.warning(f"Error warming up rule {rule.name}: {str(e)}")
                return {
                    "error": f"Rule validation failed during warm-up: {str(e)}",
                    "rule_name": rule.name,
                }

        # Measure performance
        latencies = []
        results = []
        passed_count = 0
        failed_count = 0
        error_count = 0

        for text in tqdm(self.test_data[:sample_size], desc=f"Benchmarking {rule.name}"):
            try:
                start_time = time.perf_counter()
                result = rule.validate(text)
                end_time = time.perf_counter()

                latencies.append(end_time - start_time)
                results.append(result)

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Error validating with rule {rule.name}: {str(e)}")

        # Calculate statistics
        if not latencies:
            return {"error": "No successful validations performed", "rule_name": rule.name}

        stats = {
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "throughput": sample_size / sum(latencies),
            "sample_size": sample_size,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "error_count": error_count,
            "pass_rate": (
                passed_count / (sample_size - error_count) if (sample_size - error_count) > 0 else 0
            ),
        }

        # Measure memory usage
        mem_stats = self._measure_memory(
            lambda: [rule.validate(text) for text in self.test_data[:10]]
        )
        stats.update(mem_stats)

        return stats

    def benchmark_rule_pipeline(self, rules: List[Rule], sample_size: int = 50) -> Dict[str, Any]:
        """
        Benchmark a pipeline of rules.

        Args:
            rules: List of rules to benchmark as a pipeline
            sample_size: Number of samples to use

        Returns:
            Dictionary with benchmark results
        """
        # Warm up
        for _ in range(self.warm_up_rounds):
            text = self.test_data[0]
            for rule in rules:
                try:
                    rule.validate(text)
                except Exception:
                    break

        # Measure performance
        latencies = []
        all_passed_count = 0
        partial_passed_count = 0
        all_failed_count = 0

        for text in tqdm(self.test_data[:sample_size], desc="Benchmarking rule pipeline"):
            start_time = time.perf_counter()

            # Run all rules and collect results
            rule_results = []
            for rule in rules:
                try:
                    result = rule.validate(text)
                    rule_results.append(result)
                except Exception:
                    rule_results.append(None)

            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            # Calculate pass/fail statistics
            valid_results = [r for r in rule_results if r is not None]
            if not valid_results:
                continue

            passed_results = [r for r in valid_results if r.passed]

            if len(passed_results) == len(valid_results):
                all_passed_count += 1
            elif len(passed_results) > 0:
                partial_passed_count += 1
            else:
                all_failed_count += 1

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
            "all_passed_count": all_passed_count,
            "partial_passed_count": partial_passed_count,
            "all_failed_count": all_failed_count,
            "success_rate": (all_passed_count + 0.5 * partial_passed_count) / sample_size,
        }

        # Measure memory usage
        mem_stats = self._measure_memory(
            lambda: [[rule.validate(text) for rule in rules] for text in self.test_data[:5]]
        )
        stats.update(mem_stats)

        return stats

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks and return results.

        Returns:
            Dictionary with all benchmark results
        """
        results = {
            "length_rule": self.benchmark_single_rule(self.length_rule),
            "prohibited_content_rule": self.benchmark_single_rule(self.prohibited_content_rule),
            "paragraph_rule": self.benchmark_single_rule(self.paragraph_rule),
            "style_rule": self.benchmark_single_rule(self.style_rule),
            "repetition_rule": self.benchmark_single_rule(self.repetition_rule),
            "symmetry_rule": self.benchmark_single_rule(self.symmetry_rule),
            "toxicity_rule": self.benchmark_single_rule(self.toxicity_rule),
            "bias_rule": self.benchmark_single_rule(self.bias_rule),
            "python_rule": self.benchmark_single_rule(self.python_rule),
            "classifier_rule": self.benchmark_single_rule(self.classifier_rule),
        }

        # Run pipeline benchmarks
        content_pipeline = [self.length_rule, self.prohibited_content_rule, self.repetition_rule]

        format_pipeline = [self.paragraph_rule, self.style_rule, self.symmetry_rule]

        safety_pipeline = [self.toxicity_rule, self.bias_rule, self.classifier_rule]

        results["content_pipeline"] = self.benchmark_rule_pipeline(content_pipeline)
        results["format_pipeline"] = self.benchmark_rule_pipeline(format_pipeline)
        results["safety_pipeline"] = self.benchmark_rule_pipeline(safety_pipeline)

        return results


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print("\n=== Sifaka Rule Benchmarks ===\n")

    # Print individual rule results
    print("\n== Individual Rules ==")
    for rule_name, stats in results.items():
        if rule_name.endswith("_pipeline"):
            continue

        if "error" in stats:
            print(f"\n{rule_name.upper()}:")
            print("-" * 40)
            print(f"ERROR: {stats['error']}")
            continue

        print(f"\n{rule_name.upper()}:")
        print("-" * 40)
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")
        print(f"Pass Rate: {stats['pass_rate']*100:.1f}%")

    # Print pipeline results
    print("\n== Rule Pipelines ==")
    for pipeline_name, stats in results.items():
        if not pipeline_name.endswith("_pipeline"):
            continue

        print(f"\n{pipeline_name.upper()}:")
        print("-" * 40)
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")


def main():
    """Run benchmarks with example usage."""
    # Initialize benchmark suite
    benchmark = RuleBenchmark(num_samples=500)

    # Run benchmarks
    results = benchmark.run_all_benchmarks()

    # Print results
    print_benchmark_results(results)


if __name__ == "__main__":
    main()
