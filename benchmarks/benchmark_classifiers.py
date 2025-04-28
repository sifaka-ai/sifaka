"""
Benchmarking suite for Sifaka classifiers.

This module provides tools to measure:
1. Individual classifier performance
2. Memory usage
3. Throughput and latency
4. Caching effectiveness
"""

import asyncio
import cProfile
import gc
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
from tqdm import tqdm

from sifaka.classifiers.language import LanguageClassifier, LanguageConfig
from sifaka.classifiers.profanity import ProfanityClassifier, ProfanityConfig
from sifaka.classifiers.readability import ReadabilityClassifier, ReadabilityConfig
from sifaka.classifiers.sentiment import SentimentClassifier, SentimentThresholds
from sifaka.rules.content import ContentAnalyzer
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ClassifierBenchmark:
    """Benchmark suite for classifier performance measurement."""

    def __init__(
        self,
        num_samples: int = 1000,
        warm_up_rounds: int = 3,
        classifiers: Optional[List[str]] = None,
    ):
        """
        Initialize the benchmark suite.

        Args:
            num_samples: Number of text samples to use
            warm_up_rounds: Number of warm-up rounds before measurement
            classifiers: List of classifier names to benchmark. If None, all classifiers are used.
        """
        self.num_samples = num_samples
        self.warm_up_rounds = warm_up_rounds
        self.classifiers_to_run = classifiers or [
            "readability",
            "sentiment",
            "language",
            "profanity",
        ]
        self._generate_test_data()
        self._init_classifiers()

    def _generate_test_data(self) -> None:
        """Generate test data of varying complexity."""
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

        # Generate variations with different lengths and combinations
        self.test_data = []
        for _ in range(self.num_samples):
            text_type = np.random.choice(["simple", "medium", "complex"])
            if text_type == "simple":
                base_texts = simple_texts
            elif text_type == "medium":
                base_texts = medium_texts
            else:
                base_texts = complex_texts

            # Randomly combine 1-3 sentences
            num_sentences = np.random.randint(1, 4)
            selected_texts = np.random.choice(base_texts, num_sentences)
            self.test_data.append(" ".join(selected_texts))

    def _init_classifiers(self) -> None:
        """Initialize all classifiers."""
        self.classifiers = {}

        if "readability" in self.classifiers_to_run:
            readability_config = ReadabilityConfig(
                min_confidence=0.5,
                grade_level_bounds={
                    "elementary": (0.0, 6.0),
                    "middle": (6.0, 9.0),
                    "high": (9.0, 12.0),
                    "college": (12.0, 16.0),
                    "graduate": (16.0, float("inf")),
                },
            )
            self.classifiers["readability"] = ReadabilityClassifier(
                name="benchmark_readability",
                description="Benchmark readability classifier",
                readability_config=readability_config,
            )

        if "sentiment" in self.classifiers_to_run:
            self.classifiers["sentiment"] = SentimentClassifier(
                name="benchmark_sentiment",
                description="Benchmark sentiment classifier",
                thresholds=SentimentThresholds(positive=0.05, negative=-0.05),
            )

        if "language" in self.classifiers_to_run:
            lang_config = LanguageConfig(
                min_confidence=0.1, seed=0, fallback_lang="en", fallback_confidence=0.0
            )
            self.classifiers["language"] = LanguageClassifier(
                name="benchmark_language",
                description="Benchmark language classifier",
                lang_config=lang_config,
            )

        if "profanity" in self.classifiers_to_run:
            profanity_config = ProfanityConfig(
                custom_words={"bad", "inappropriate", "offensive"},
                censor_char="*",
                min_confidence=0.5,
            )
            self.classifiers["profanity"] = ProfanityClassifier(
                name="benchmark_profanity",
                description="Benchmark profanity classifier",
                profanity_config=profanity_config,
            )

        # Warm up all classifiers
        for classifier in self.classifiers.values():
            try:
                classifier.warm_up()
            except Exception as e:
                logger.warning(f"Failed to warm up {classifier.name}: {str(e)}")

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

    def benchmark_single_classifier(
        self, classifier: Any, sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark a single classifier.

        Args:
            classifier: Classifier instance to benchmark
            sample_size: Number of samples to use

        Returns:
            Dictionary with benchmark results
        """
        try:
            # Warm up
            for _ in range(self.warm_up_rounds):
                classifier.classify(self.test_data[0])

            # Measure performance
            latencies = []
            results = []
            errors = 0

            for text in tqdm(
                self.test_data[:sample_size],
                desc=f"Benchmarking {classifier.name}",
                leave=False,
            ):
                try:
                    start_time = time.perf_counter()
                    result = classifier.classify(text)
                    end_time = time.perf_counter()

                    latencies.append(end_time - start_time)
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        f"Error in {classifier.name} for text: {text[:50]}... - {str(e)}"
                    )
                    errors += 1

            if not latencies:
                return {
                    "error": f"All {sample_size} classifications failed",
                    "classifier": classifier.name,
                }

            # Calculate statistics
            stats = {
                "classifier": classifier.name,
                "mean_latency": statistics.mean(latencies),
                "median_latency": statistics.median(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_dev": statistics.stdev(latencies),
                "throughput": (sample_size - errors) / sum(latencies),
                "sample_size": sample_size,
                "errors": errors,
                "success_rate": (sample_size - errors) / sample_size,
            }

            # Measure memory usage
            mem_stats = self._measure_memory(
                lambda: [classifier.classify(text) for text in self.test_data[:10]]
            )
            stats.update(mem_stats)

            # Measure cache effectiveness (if applicable)
            if hasattr(classifier, "_word_cache"):
                stats["cache_size"] = len(classifier._word_cache)
                stats["cache_hits"] = getattr(classifier, "_cache_hits", 0)
                stats["cache_misses"] = getattr(classifier, "_cache_misses", 0)

            return stats
        except Exception as e:
            return {
                "error": str(e),
                "classifier": classifier.name,
            }

    def benchmark_pipeline(self, api_key: str, sample_size: int = 50) -> Dict[str, Any]:
        """
        Benchmark the entire classifier pipeline.

        Args:
            api_key: API key for LLM provider
            sample_size: Number of samples to use

        Returns:
            Dictionary with benchmark results
        """
        analyzer = ContentAnalyzer(api_key=api_key)

        # Warm up
        for _ in range(self.warm_up_rounds):
            asyncio.run(analyzer.analyze_content(self.test_data[0]))

        # Measure performance
        latencies = []
        results = []

        async def run_benchmarks():
            for text in tqdm(self.test_data[:sample_size], desc="Benchmarking pipeline"):
                start_time = time.perf_counter()
                result = await analyzer.analyze_content(text)
                end_time = time.perf_counter()

                latencies.append(end_time - start_time)
                results.append(result)

        asyncio.run(run_benchmarks())

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
        }

        # Measure memory usage
        mem_stats = self._measure_memory(
            lambda: asyncio.run(analyzer.analyze_content(self.test_data[0]))
        )
        stats.update(mem_stats)

        return stats

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run benchmarks for all initialized classifiers.

        Returns:
            Dictionary with benchmark results for each classifier
        """
        results = {}
        for name, classifier in self.classifiers.items():
            try:
                results[name] = self.benchmark_single_classifier(classifier)
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "classifier": name,
                }
        return results


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """
    Print benchmark results in a formatted way.

    Args:
        results: Dictionary with benchmark results
    """
    print("\nBenchmark Results:")
    print("=" * 80)

    for classifier_name, stats in results.items():
        print(f"\n{classifier_name}:")
        print("-" * 40)

        if "error" in stats:
            print(f"Error: {stats['error']}")
            continue

        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
        print(f"Latency (seconds):")
        print(f"  Mean: {stats['mean_latency']:.4f}")
        print(f"  Median: {stats['median_latency']:.4f}")
        print(f"  P95: {stats['p95_latency']:.4f}")
        print(f"  P99: {stats['p99_latency']:.4f}")
        print(f"Memory usage:")
        print(f"  Before: {stats['memory_before_mb']:.1f} MB")
        print(f"  After: {stats['memory_after_mb']:.1f} MB")
        print(f"  Difference: {stats['memory_diff_mb']:.1f} MB")

        if "cache_size" in stats:
            print(f"Cache statistics:")
            print(f"  Size: {stats['cache_size']}")
            if "cache_hits" in stats:
                print(f"  Hits: {stats['cache_hits']}")
                print(f"  Misses: {stats['cache_misses']}")


if __name__ == "__main__":
    # Initialize benchmark suite
    benchmark = ClassifierBenchmark(num_samples=1000)

    # Run benchmarks
    results = benchmark.run_all_benchmarks()

    # Print results
    print_benchmark_results(results)

    # Optional: Run with profiler
    profiler = cProfile.Profile()
    profiler.enable()
    benchmark.benchmark_single_classifier(benchmark.classifiers["readability"], sample_size=100)
    profiler.disable()
    profiler.print_stats(sort="cumulative")
