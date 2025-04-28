"""
Visualization tools for benchmark results.

This module provides functions to create:
1. Latency distribution plots
2. Memory usage comparisons
3. Throughput comparisons
4. Cache effectiveness visualizations
"""

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BenchmarkVisualizer:
    """Visualizer for benchmark results."""

    def __init__(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """
        Initialize the visualizer.

        Args:
            results: Benchmark results dictionary
            output_dir: Directory to save visualizations
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def plot_latency_distribution(self) -> None:
        """Create violin plots of latency distributions."""
        plt.figure(figsize=(12, 6))

        data = []
        for classifier, stats in self.results.items():
            if classifier != "pipeline":  # Exclude pipeline from this comparison
                data.append(
                    {
                        "Classifier": classifier,
                        "Mean Latency (ms)": stats["mean_latency"] * 1000,
                        "P95 Latency (ms)": stats["p95_latency"] * 1000,
                        "P99 Latency (ms)": stats["p99_latency"] * 1000,
                    }
                )

        df = pd.DataFrame(data)
        df_melted = pd.melt(
            df,
            id_vars=["Classifier"],
            var_name="Metric",
            value_name="Latency (ms)",
        )

        sns.violinplot(
            data=df_melted,
            x="Classifier",
            y="Latency (ms)",
            hue="Metric",
        )

        plt.title("Latency Distribution by Classifier")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_distribution.png")
        plt.close()

    def plot_memory_usage(self) -> None:
        """Create bar plots of memory usage."""
        plt.figure(figsize=(10, 6))

        classifiers = []
        memory_usage = []

        for classifier, stats in self.results.items():
            classifiers.append(classifier)
            memory_usage.append(stats["memory_diff_mb"])

        sns.barplot(x=classifiers, y=memory_usage)
        plt.title("Memory Usage by Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Memory Usage (MB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_usage.png")
        plt.close()

    def plot_throughput_comparison(self) -> None:
        """Create bar plots of throughput."""
        plt.figure(figsize=(10, 6))

        classifiers = []
        throughput = []

        for classifier, stats in self.results.items():
            classifiers.append(classifier)
            throughput.append(stats["throughput"])

        sns.barplot(x=classifiers, y=throughput)
        plt.title("Throughput by Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Texts per Second")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "throughput.png")
        plt.close()

    def plot_cache_effectiveness(self) -> None:
        """Create visualization of cache effectiveness."""
        cache_data = []

        for classifier, stats in self.results.items():
            if "cache_size" in stats:
                cache_data.append(
                    {
                        "Classifier": classifier,
                        "Cache Size": stats["cache_size"],
                        "Cache Hit Rate": stats.get("cache_hit_rate", 0),
                    }
                )

        if not cache_data:
            return

        df = pd.DataFrame(cache_data)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="Cache Size",
            y="Cache Hit Rate",
            hue="Classifier",
            s=100,
        )
        plt.title("Cache Effectiveness")
        plt.tight_layout()
        plt.savefig(self.output_dir / "cache_effectiveness.png")
        plt.close()

    def create_summary_report(self) -> None:
        """Create a markdown summary report."""
        report = ["# Benchmark Results Summary\n"]

        # Overall statistics
        report.append("## Overall Statistics\n")
        for classifier, stats in self.results.items():
            report.append(f"### {classifier.upper()}\n")
            report.append(f"- Throughput: {stats['throughput']:.2f} texts/second")
            report.append(f"- Mean Latency: {stats['mean_latency']*1000:.2f}ms")
            report.append(f"- P95 Latency: {stats['p95_latency']*1000:.2f}ms")
            report.append(f"- Memory Usage: {stats['memory_diff_mb']:.2f}MB")
            if "cache_size" in stats:
                report.append(f"- Cache Size: {stats['cache_size']} entries")
            report.append("\n")

        # Save report
        with open(self.output_dir / "summary.md", "w") as f:
            f.write("\n".join(report))

    def save_results_json(self) -> None:
        """Save raw results as JSON."""
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def create_all_visualizations(self) -> None:
        """Create all visualizations and reports."""
        self.plot_latency_distribution()
        self.plot_memory_usage()
        self.plot_throughput_comparison()
        self.plot_cache_effectiveness()
        self.create_summary_report()
        self.save_results_json()

def main():
    """Example usage of the visualizer."""
    # Load results from a benchmark run
    from benchmark_classifiers import ClassifierBenchmark

    # Run benchmarks
    benchmark = ClassifierBenchmark(num_samples=100)
    results = benchmark.run_all_benchmarks()

    # Create visualizations
    visualizer = BenchmarkVisualizer(results)
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
