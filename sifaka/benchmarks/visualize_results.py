"""
Visualization module for benchmark results.

This module provides tools to create:
1. Latency distribution plots
2. Memory usage comparisons
3. Throughput comparisons
4. Cache effectiveness visualizations
5. Summary reports in markdown format
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkVisualizer:
    """Creates visualizations and reports from benchmark results."""

    def __init__(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """
        Initialize the visualizer.

        Args:
            results: Dictionary containing benchmark results
            output_dir: Directory to save visualizations and reports
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style for all plots
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def plot_latency_distribution(self) -> None:
        """Create violin plots showing latency distribution for each classifier."""
        plt.figure(figsize=(12, 6))

        # Prepare data
        data = []
        for name, stats in self.results.items():
            if name != "pipeline":  # Exclude pipeline for better scale
                data.append(
                    {
                        "classifier": name,
                        "mean": stats["mean_latency"] * 1000,  # Convert to ms
                        "p95": stats["p95_latency"] * 1000,
                        "p99": stats["p99_latency"] * 1000,
                    }
                )

        df = pd.DataFrame(data)

        # Create plot
        sns.violinplot(data=df, x="classifier", y="mean")
        plt.title("Latency Distribution by Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=45)

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "latency_distribution.png"))
        plt.close()

    def plot_memory_usage(self) -> None:
        """Create bar plots showing memory usage for each classifier."""
        plt.figure(figsize=(12, 6))

        # Prepare data
        names = []
        memory_usage = []
        for name, stats in self.results.items():
            names.append(name)
            memory_usage.append(stats["memory_diff_mb"])

        # Create plot
        plt.bar(names, memory_usage)
        plt.title("Memory Usage by Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Memory Usage (MB)")
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(memory_usage):
            plt.text(i, v, f"{v:.1f}MB", ha="center", va="bottom")

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "memory_usage.png"))
        plt.close()

    def plot_throughput_comparison(self) -> None:
        """Create bar plots comparing throughput of each classifier."""
        plt.figure(figsize=(12, 6))

        # Prepare data
        names = []
        throughput = []
        for name, stats in self.results.items():
            names.append(name)
            throughput.append(stats["throughput"])

        # Create plot
        plt.bar(names, throughput)
        plt.title("Throughput Comparison")
        plt.xlabel("Classifier")
        plt.ylabel("Texts per Second")
        plt.xticks(rotation=45)

        # Add value labels
        for i, v in enumerate(throughput):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "throughput_comparison.png"))
        plt.close()

    def plot_cache_effectiveness(self) -> None:
        """Create visualization of cache effectiveness for applicable classifiers."""
        cache_data = []
        for name, stats in self.results.items():
            if "cache_size" in stats:
                cache_data.append({"classifier": name, "cache_size": stats["cache_size"]})

        if not cache_data:
            logger.info("No cache data available for visualization")
            return

        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(cache_data)
        sns.barplot(data=df, x="classifier", y="cache_size")
        plt.title("Cache Size by Classifier")
        plt.xlabel("Classifier")
        plt.ylabel("Number of Cached Entries")
        plt.xticks(rotation=45)

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cache_effectiveness.png"))
        plt.close()

    def create_markdown_report(self) -> None:
        """Create a markdown report summarizing benchmark results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""# Sifaka Classifier Benchmark Report
Generated: {timestamp}

## Summary
This report presents the performance metrics for various Sifaka classifiers.

## Performance Metrics

"""
        for name, stats in self.results.items():
            report += f"""### {name.upper()} Classifier
- Throughput: {stats['throughput']:.2f} texts/second
- Mean Latency: {stats['mean_latency']*1000:.2f}ms
- Median Latency: {stats['median_latency']*1000:.2f}ms
- P95 Latency: {stats['p95_latency']*1000:.2f}ms
- P99 Latency: {stats['p99_latency']*1000:.2f}ms
- Memory Usage: {stats['memory_diff_mb']:.2f}MB
"""
            if "cache_size" in stats:
                report += f"- Cache Size: {stats['cache_size']} entries\n"
            report += "\n"

        report += """## Visualizations
The following visualizations are available in the benchmark_results directory:
1. latency_distribution.png - Distribution of latencies for each classifier
2. memory_usage.png - Memory usage comparison
3. throughput_comparison.png - Throughput comparison
4. cache_effectiveness.png - Cache size comparison (if applicable)

## Notes
- All latency measurements are in milliseconds
- Memory usage is measured in megabytes
- Throughput is measured in texts per second
"""

        # Save report
        with open(os.path.join(self.output_dir, "benchmark_report.md"), "w") as f:
            f.write(report)

    def save_raw_results(self) -> None:
        """Save raw benchmark results as JSON."""
        with open(os.path.join(self.output_dir, "raw_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def create_all_visualizations(self) -> None:
        """Create all visualizations and reports."""
        logger.info("Creating benchmark visualizations...")
        self.plot_latency_distribution()
        self.plot_memory_usage()
        self.plot_throughput_comparison()
        self.plot_cache_effectiveness()
        self.create_markdown_report()
        self.save_raw_results()
        logger.info(f"Benchmark results saved to {self.output_dir}/")


if __name__ == "__main__":
    # Example usage
    from sifaka.benchmarks.benchmark_classifiers import ClassifierBenchmark

    # Run benchmarks
    benchmark = ClassifierBenchmark(num_samples=1000)
    results = benchmark.run_all_benchmarks()

    # Create visualizations
    visualizer = BenchmarkVisualizer(results)
    visualizer.create_all_visualizations()
