#!/usr/bin/env python3
"""Benchmark runner for Sifaka performance tests.

This script provides a convenient way to run performance benchmarks
with various configurations and generate reports.

Usage:
    python benchmark_runner.py --help
    python benchmark_runner.py --quick
    python benchmark_runner.py --full --output results.json
    python benchmark_runner.py --category core --iterations 5
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from performance_benchmarks import PerformanceBenchmark


class BenchmarkRunner:
    """Orchestrates benchmark execution and reporting."""

    def __init__(self, iterations: int = 3, output_file: Optional[str] = None):
        self.iterations = iterations
        self.output_file = output_file
        self.results = []

    async def run_quick_benchmark(self) -> Dict:
        """Run a quick benchmark suite."""
        print("Running quick benchmark suite...")

        from unittest.mock import AsyncMock, MagicMock, patch

        from sifaka import improve
        from sifaka.core.config import Config

        benchmark = PerformanceBenchmark()

        # Mock LLM client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Good. SUGGESTIONS: Continue."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            config = Config.fast()

            # Run basic benchmarks
            benchmarks = [
                ("basic_improve", "Basic improve operation"),
                ("single_critic", "Single critic evaluation"),
                ("memory_usage", "Memory usage test"),
            ]

            for name, description in benchmarks:
                print(f"  Running {description}...")

                total_time = 0
                total_memory = 0

                for i in range(self.iterations):
                    start_metrics = benchmark.start_measurement()
                    await improve(f"Test text {i}", config=config)
                    metrics = benchmark.end_measurement(start_metrics)

                    total_time += metrics["execution_time"]
                    total_memory += metrics["memory_usage"]

                avg_time = total_time / self.iterations
                avg_memory = total_memory / self.iterations

                benchmark.add_result(
                    name,
                    {
                        "execution_time": avg_time,
                        "memory_usage": avg_memory,
                        "cpu_usage": 0,
                    },
                    {"iterations": self.iterations, "description": description},
                )

                print(f"    Average time: {avg_time:.3f}s")
                print(f"    Average memory: {avg_memory:.1f}MB")

        return benchmark.get_results_summary()

    def run_pytest_benchmarks(self, category: Optional[str] = None) -> Dict:
        """Run pytest-based benchmarks."""
        print("Running pytest benchmarks...")

        import subprocess

        cmd = ["python", "-m", "pytest", "benchmarks/performance_benchmarks.py", "-v"]

        if category:
            cmd.extend(["-k", category])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )

            return {
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }
        except Exception as e:
            return {"command": " ".join(cmd), "error": str(e), "success": False}

    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable report."""
        report = []
        report.append("=" * 60)
        report.append("SIFAKA PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Iterations: {self.iterations}")
        report.append("")

        if "total_tests" in results:
            report.append(f"Total Tests: {results['total_tests']}")
            report.append(
                f"Average Execution Time: {results['avg_execution_time']:.3f}s"
            )
            report.append(f"Max Execution Time: {results['max_execution_time']:.3f}s")
            report.append(f"Average Memory Usage: {results['avg_memory_usage']:.1f}MB")
            report.append(f"Max Memory Usage: {results['max_memory_usage']:.1f}MB")
            report.append("")

            report.append("Individual Results:")
            report.append("-" * 40)

            for result in results.get("results", []):
                name = result["test_name"]
                metrics = result["metrics"]
                metadata = result.get("metadata", {})

                report.append(f"Test: {name}")
                report.append(f"  Time: {metrics['execution_time']:.3f}s")
                report.append(f"  Memory: {metrics['memory_usage']:.1f}MB")

                if metadata:
                    report.append(f"  Metadata: {metadata}")

                report.append("")

        return "\n".join(report)

    def save_results(self, results: Dict) -> None:
        """Save results to file."""
        if self.output_file:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Sifaka performance benchmarks")

    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark suite"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full pytest benchmark suite"
    )
    parser.add_argument("--category", type=str, help="Run specific benchmark category")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations for each test"
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument(
        "--report", action="store_true", help="Generate human-readable report"
    )

    args = parser.parse_args()

    if not any([args.quick, args.full]):
        args.quick = True  # Default to quick

    runner = BenchmarkRunner(iterations=args.iterations, output_file=args.output)

    try:
        if args.quick:
            results = asyncio.run(runner.run_quick_benchmark())
        elif args.full:
            results = runner.run_pytest_benchmarks(args.category)

        # Generate report
        if args.report and "total_tests" in results:
            report = runner.generate_report(results)
            print("\n" + report)

        # Save results
        if args.output:
            runner.save_results(results)

        # Print summary
        if "total_tests" in results:
            print(f"\nBenchmark completed: {results['total_tests']} tests")
            print(f"Average time: {results['avg_execution_time']:.3f}s")
            print(f"Average memory: {results['avg_memory_usage']:.1f}MB")
        elif "success" in results:
            print(
                f"\nPytest benchmark {'succeeded' if results['success'] else 'failed'}"
            )
            if not results["success"]:
                print("Error output:", results.get("stderr", "No error output"))

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
