"""
Benchmarking suite for Sifaka classifiers.

This module provides tools to measure:
1. Individual classifier performance
2. Memory usage
3. Throughput and latency
4. Caching effectiveness
"""

import gc
import time
import statistics
from typing import Dict, Any, Callable
from memory_profiler import profile
import psutil
import numpy as np
from tqdm import tqdm

from sifaka.classifiers.readability import ReadabilityClassifier
from sifaka.classifiers.sentiment import SentimentClassifier
from sifaka.classifiers.language import LanguageClassifier
from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class ClassifierBenchmark:
    """Benchmark suite for classifier performance measurement."""

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
        self.readability = ReadabilityClassifier(
            name="benchmark_readability",
            description="Benchmark readability classifier",
        )
        self.sentiment = SentimentClassifier(
            name="benchmark_sentiment",
            description="Benchmark sentiment classifier",
        )
        self.language = LanguageClassifier(
            name="benchmark_language",
            description="Benchmark language classifier",
        )
        self.profanity = ProfanityClassifier(
            name="benchmark_profanity",
            description="Benchmark profanity classifier",
        )

        # Warm up all classifiers
        classifiers = [self.readability, self.sentiment, self.language, self.profanity]
        for classifier in classifiers:
            classifier.warm_up()

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
        # Warm up
        for _ in range(self.warm_up_rounds):
            classifier.classify(self.test_data[0])

        # Measure performance
        latencies = []
        results = []

        for text in tqdm(self.test_data[:sample_size], desc=f"Benchmarking {classifier.name}"):
            start_time = time.perf_counter()
            result = classifier.classify(text)
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
        }

        # Measure memory usage
        mem_stats = self._measure_memory(
            lambda: [classifier.classify(text) for text in self.test_data[:10]]
        )
        stats.update(mem_stats)

        # Measure cache effectiveness (if applicable)
        if hasattr(classifier, "_word_cache"):
            stats["cache_size"] = len(classifier._word_cache)

        return stats

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
        Run all benchmarks and return results.

        Returns:
            Dictionary with all benchmark results
        """
        results = {
            "readability": self.benchmark_single_classifier(self.readability),
            "sentiment": self.benchmark_single_classifier(self.sentiment),
            "language": self.benchmark_single_classifier(self.language),
            "profanity": self.benchmark_single_classifier(self.profanity),
        }

        return results


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print("\n=== Sifaka Classifier Benchmarks ===\n")

    for classifier_name, stats in results.items():
        print(f"\n{classifier_name.upper()} Classifier:")
        print("-" * 40)
        print(f"Throughput: {stats['throughput']:.2f} texts/second")
        print(f"Mean Latency: {stats['mean_latency']*1000:.2f}ms")
        print(f"P95 Latency: {stats['p95_latency']*1000:.2f}ms")
        print(f"P99 Latency: {stats['p99_latency']*1000:.2f}ms")
        print(f"Memory Usage: {stats['memory_diff_mb']:.2f}MB")
        if "cache_size" in stats:
            print(f"Cache Size: {stats['cache_size']} entries")


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
    benchmark.benchmark_single_classifier(benchmark.readability, sample_size=100)
    profiler.disable()
    profiler.print_stats(sort="cumulative")
