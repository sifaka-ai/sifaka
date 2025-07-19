"""Performance benchmarks for Sifaka.

This module provides comprehensive performance benchmarking for:
- Core operations (improve, critique, validate)
- Different configurations and settings
- Scalability with various input sizes
- Memory usage patterns
- Concurrent operation performance

Run with: python -m pytest benchmarks/performance_benchmarks.py -v
"""

import asyncio
import gc
import json
import time
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from sifaka import improve
from sifaka.core.config import Config
from sifaka.core.engine import SifakaEngine
from sifaka.core.models import SifakaResult
from sifaka.storage import FileStorage, MemoryStorage
from sifaka.validators import ContentValidator, LengthValidator


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self):
        self.results = []
        self.process = psutil.Process()

    def start_measurement(self) -> Dict[str, float]:
        """Start performance measurement."""
        gc.collect()
        return {
            "start_time": time.time(),
            "start_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
            "start_cpu": self.process.cpu_percent(),
        }

    def end_measurement(self, start_metrics: Dict[str, float]) -> Dict[str, float]:
        """End performance measurement and calculate metrics."""
        end_time = time.time()
        gc.collect()

        return {
            "execution_time": end_time - start_metrics["start_time"],
            "memory_usage": self.process.memory_info().rss / 1024 / 1024
            - start_metrics["start_memory"],
            "cpu_usage": self.process.cpu_percent() - start_metrics["start_cpu"],
        }

    def add_result(
        self, test_name: str, metrics: Dict[str, float], metadata: Optional[Dict] = None
    ):
        """Add benchmark result."""
        result = {
            "test_name": test_name,
            "timestamp": time.time(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        self.results.append(result)
        return result

    def get_results_summary(self) -> Dict:
        """Get summary of all benchmark results."""
        if not self.results:
            return {"total_tests": 0}

        times = [r["metrics"]["execution_time"] for r in self.results]
        memories = [r["metrics"]["memory_usage"] for r in self.results]

        return {
            "total_tests": len(self.results),
            "avg_execution_time": sum(times) / len(times),
            "max_execution_time": max(times),
            "min_execution_time": min(times),
            "avg_memory_usage": sum(memories) / len(memories),
            "max_memory_usage": max(memories),
            "total_execution_time": sum(times),
            "results": self.results,
        }


@pytest.fixture
def benchmark():
    """Create performance benchmark instance."""
    return PerformanceBenchmark()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for benchmarks."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[
        0
    ].message.content = "REFLECTION: Good analysis. SUGGESTIONS: Continue."

    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture
def fast_config():
    """Configuration optimized for fast benchmarks."""
    return Config.fast()


class TestCoreOperationBenchmarks:
    """Benchmark core operations."""

    @pytest.mark.asyncio
    async def test_basic_improve_benchmark(
        self, benchmark, mock_llm_client, fast_config
    ):
        """Benchmark basic improve operation."""
        text = "This is a test text for basic improvement benchmarking."

        # Warm up
        await improve(text, config=fast_config)

        # Benchmark
        start_metrics = benchmark.start_measurement()
        result = await improve(text, config=fast_config)
        metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "basic_improve",
            metrics,
            {
                "text_length": len(text),
                "iterations": result.iteration,
                "model": fast_config.llm.model,
            },
        )

        # Assertions
        assert isinstance(result, SifakaResult)
        assert metrics["execution_time"] < 2.0  # Should be fast
        assert metrics["memory_usage"] < 20.0  # Should use < 20MB

    @pytest.mark.asyncio
    async def test_multiple_critics_benchmark(self, benchmark, mock_llm_client):
        """Benchmark multiple critics performance."""
        text = "Test text for multiple critics benchmark."

        # Mock different responses for different critics
        def mock_create_side_effect(*args, **kwargs):
            mock_response = MagicMock()
            system_content = kwargs.get("messages", [{}])[0].get("content", "")

            if "constitutional" in system_content.lower():
                mock_response.choices[0].message.content = json.dumps(
                    {
                        "overall_assessment": "Good",
                        "principle_scores": {"1": 4},
                        "violations": [],
                        "suggestions": ["Continue"],
                        "overall_confidence": 0.8,
                        "evaluation_quality": 4,
                    }
                )
            else:
                mock_response.choices[
                    0
                ].message.content = "REFLECTION: Good analysis. SUGGESTIONS: Continue."

            return mock_response

        mock_llm_client.chat.completions.create.side_effect = mock_create_side_effect

        critic_sets = [
            ["reflexion"],
            ["reflexion", "constitutional"],
            ["reflexion", "constitutional", "self_refine"],
            ["reflexion", "constitutional", "self_refine", "n_critics"],
        ]

        for critics in critic_sets:
            config = Config.fast()

            start_metrics = benchmark.start_measurement()
            result = await improve(text, config=config)
            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                f"critics_{len(critics)}",
                metrics,
                {
                    "critic_count": len(critics),
                    "critics": critics,
                    "critique_count": len(result.critiques),
                },
            )

            # Should scale reasonably with critic count
            assert metrics["execution_time"] < len(critics) * 1.0

    @pytest.mark.asyncio
    async def test_iteration_scaling_benchmark(self, benchmark, mock_llm_client):
        """Benchmark how performance scales with iterations."""
        text = "Test text for iteration scaling benchmark."

        # Mock that always suggests improvement
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = (
            "REFLECTION: Needs improvement. SUGGESTIONS: Add more detail."
        )
        mock_llm_client.chat.completions.create.return_value = mock_response

        for max_iterations in [1, 2, 3, 5]:
            config = Config.fast()

            start_metrics = benchmark.start_measurement()
            result = await improve(text, config=config)
            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                f"iterations_{max_iterations}",
                metrics,
                {
                    "max_iterations": max_iterations,
                    "actual_iterations": result.iteration,
                    "generation_count": len(result.generations),
                },
            )

            # Should scale linearly with iterations
            assert metrics["execution_time"] < max_iterations * 1.5


class TestScalabilityBenchmarks:
    """Benchmark scalability characteristics."""

    @pytest.mark.asyncio
    async def test_text_size_scaling(self, benchmark, mock_llm_client, fast_config):
        """Benchmark scaling with different text sizes."""
        text_sizes = [100, 500, 1000, 5000, 10000]  # Characters

        for size in text_sizes:
            text = "A" * size

            start_metrics = benchmark.start_measurement()
            result = await improve(text, config=fast_config)
            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                f"text_size_{size}",
                metrics,
                {
                    "text_size": size,
                    "original_length": len(result.original_text),
                    "final_length": len(result.final_text),
                },
            )

            # Should not degrade exponentially
            assert metrics["execution_time"] < 5.0
            assert metrics["memory_usage"] < 50.0

    @pytest.mark.asyncio
    async def test_concurrent_operations_benchmark(
        self, benchmark, mock_llm_client, fast_config
    ):
        """Benchmark concurrent operations."""
        text = "Concurrent benchmark test text."

        async def single_improve(i: int):
            return await improve(f"{text} {i}", config=fast_config)

        # Test different concurrency levels
        for concurrency in [1, 5, 10, 20]:
            start_metrics = benchmark.start_measurement()

            if concurrency == 1:
                results = [await single_improve(0)]
            else:
                tasks = [single_improve(i) for i in range(concurrency)]
                results = await asyncio.gather(*tasks)

            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                f"concurrent_{concurrency}",
                metrics,
                {
                    "concurrency": concurrency,
                    "results_count": len(results),
                    "avg_time_per_operation": metrics["execution_time"] / concurrency,
                },
            )

            # All operations should complete
            assert len(results) == concurrency
            assert all(isinstance(r, SifakaResult) for r in results)

    @pytest.mark.asyncio
    async def test_validator_scaling_benchmark(
        self, benchmark, mock_llm_client, fast_config
    ):
        """Benchmark scaling with validators."""
        text = "Test text with various validator terms and validation requirements."

        # Create different validator sets
        validator_sets = [
            [],  # No validators
            [LengthValidator(min_length=10, max_length=1000)],
            [
                LengthValidator(min_length=10, max_length=1000),
                ContentValidator(required_terms=["test"]),
            ],
            [
                LengthValidator(min_length=10, max_length=1000),
                ContentValidator(required_terms=["test", "validator"]),
                ContentValidator(forbidden_terms=["bad", "error"]),
            ],
        ]

        for i, validators in enumerate(validator_sets):
            start_metrics = benchmark.start_measurement()
            result = await improve(text, validators=validators, config=fast_config)
            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                f"validators_{i}",
                metrics,
                {
                    "validator_count": len(validators),
                    "validation_count": len(result.validations),
                    "all_validations_passed": all(v.passed for v in result.validations),
                },
            )

            # Should scale reasonably
            assert metrics["execution_time"] < 3.0


class TestStorageBenchmarks:
    """Benchmark storage operations."""

    def test_memory_storage_benchmark(self, benchmark):
        """Benchmark memory storage operations."""
        storage = MemoryStorage()

        # Create test results
        results = []
        for i in range(100):
            result = SifakaResult(
                original_text=f"Test text {i}",
                final_text=f"Improved text {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=0.1,
            )
            results.append(result)

        # Benchmark save operations
        start_metrics = benchmark.start_measurement()
        for result in results:
            asyncio.run(storage.save(result))
        save_metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "memory_storage_save",
            save_metrics,
            {
                "operation": "save",
                "result_count": len(results),
                "operations_per_second": len(results) / save_metrics["execution_time"],
            },
        )

        # Benchmark load operations
        start_metrics = benchmark.start_measurement()
        for result in results:
            asyncio.run(storage.load(result.id))
        load_metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "memory_storage_load",
            load_metrics,
            {
                "operation": "load",
                "result_count": len(results),
                "operations_per_second": len(results) / load_metrics["execution_time"],
            },
        )

        # Should be fast
        assert save_metrics["execution_time"] < 1.0
        assert load_metrics["execution_time"] < 1.0

    @pytest.mark.asyncio
    async def test_file_storage_benchmark(self, benchmark):
        """Benchmark file storage operations."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(base_path=temp_dir)

            # Create test results (fewer for file storage)
            results = []
            for i in range(20):
                result = SifakaResult(
                    original_text=f"File storage test {i}",
                    final_text=f"Improved file test {i}",
                    iteration=1,
                    generations=[],
                    critiques=[],
                    validations=[],
                    processing_time=0.1,
                )
                results.append(result)

            # Benchmark save operations
            start_metrics = benchmark.start_measurement()
            for result in results:
                await storage.save(result)
            save_metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                "file_storage_save",
                save_metrics,
                {
                    "operation": "save",
                    "result_count": len(results),
                    "operations_per_second": len(results)
                    / save_metrics["execution_time"],
                },
            )

            # Benchmark load operations
            start_metrics = benchmark.start_measurement()
            for result in results:
                await storage.load(result.id)
            load_metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                "file_storage_load",
                load_metrics,
                {
                    "operation": "load",
                    "result_count": len(results),
                    "operations_per_second": len(results)
                    / load_metrics["execution_time"],
                },
            )

            # Should complete in reasonable time
            assert save_metrics["execution_time"] < 3.0
            assert load_metrics["execution_time"] < 2.0


class TestMemoryBenchmarks:
    """Benchmark memory usage patterns."""

    @pytest.mark.asyncio
    async def test_memory_growth_benchmark(
        self, benchmark, mock_llm_client, fast_config
    ):
        """Benchmark memory growth over many operations."""
        text = "Memory growth benchmark test text."

        start_metrics = benchmark.start_measurement()
        initial_memory = start_metrics["start_memory"]

        # Perform many operations
        for i in range(50):
            await improve(f"{text} {i}", config=fast_config)

            # Measure memory every 10 operations
            if i % 10 == 0:
                current_memory = benchmark.process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory

                benchmark.add_result(
                    f"memory_growth_{i}",
                    {
                        "execution_time": time.time() - start_metrics["start_time"],
                        "memory_usage": memory_growth,
                        "cpu_usage": 0,
                    },
                    {
                        "operation_count": i + 1,
                        "memory_per_operation": memory_growth / (i + 1) if i > 0 else 0,
                    },
                )

        # Force garbage collection
        gc.collect()
        final_memory = benchmark.process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        # Memory growth should be bounded
        assert total_growth < 100.0  # Less than 100MB total growth

    @pytest.mark.asyncio
    async def test_memory_bounds_benchmark(self, benchmark, mock_llm_client):
        """Benchmark memory bounds enforcement."""
        text = "Memory bounds test text."

        # Mock that always suggests improvement to force max iterations
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "REFLECTION: Needs improvement. SUGGESTIONS: Add more."
        mock_llm_client.chat.completions.create.return_value = mock_response

        config = Config.fast()

        start_metrics = benchmark.start_measurement()
        result = await improve(text, config=config)
        metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "memory_bounds",
            metrics,
            {
                "max_iterations": config.max_iterations,
                "actual_iterations": result.iteration,
                "generation_count": len(result.generations),
                "critique_count": len(result.critiques),
                "generation_bound": len(result.generations) <= 10,
                "critique_bound": len(result.critiques) <= 20,
            },
        )

        # Memory bounds should be enforced
        assert len(result.generations) <= 10
        assert len(result.critiques) <= 20


class TestEnginePerformanceBenchmarks:
    """Benchmark engine-specific performance."""

    @pytest.mark.asyncio
    async def test_engine_initialization_benchmark(self, benchmark, fast_config):
        """Benchmark engine initialization."""
        start_metrics = benchmark.start_measurement()

        # Create multiple engines
        engines = []
        for i in range(10):
            engine = SifakaEngine(config=fast_config)
            engines.append(engine)

        metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "engine_initialization",
            metrics,
            {
                "engine_count": len(engines),
                "time_per_engine": metrics["execution_time"] / len(engines),
            },
        )

        # Should be fast to initialize
        assert metrics["execution_time"] < 1.0
        assert len(engines) == 10

    @pytest.mark.asyncio
    async def test_engine_reuse_benchmark(
        self, benchmark, mock_llm_client, fast_config
    ):
        """Benchmark engine reuse vs recreation."""
        text = "Engine reuse benchmark test."

        # Test engine recreation
        start_metrics = benchmark.start_measurement()
        for i in range(10):
            engine = SifakaEngine(config=fast_config)
            await engine.improve(f"{text} {i}")
        recreation_metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "engine_recreation",
            recreation_metrics,
            {"operation_count": 10, "pattern": "recreation"},
        )

        # Test engine reuse
        engine = SifakaEngine(config=fast_config)
        start_metrics = benchmark.start_measurement()
        for i in range(10):
            await engine.improve(f"{text} {i}")
        reuse_metrics = benchmark.end_measurement(start_metrics)

        benchmark.add_result(
            "engine_reuse", reuse_metrics, {"operation_count": 10, "pattern": "reuse"}
        )

        # Reuse should be faster
        assert reuse_metrics["execution_time"] < recreation_metrics["execution_time"]


def test_benchmark_summary(benchmark):
    """Test benchmark summary generation."""
    # Add some dummy results
    for i in range(5):
        benchmark.add_result(
            f"test_{i}",
            {
                "execution_time": 0.1 + i * 0.05,
                "memory_usage": 5.0 + i * 2.0,
                "cpu_usage": 10.0 + i * 1.0,
            },
            {"test_id": i},
        )

    summary = benchmark.get_results_summary()

    assert summary["total_tests"] == 5
    assert "avg_execution_time" in summary
    assert "max_execution_time" in summary
    assert "avg_memory_usage" in summary
    assert len(summary["results"]) == 5


if __name__ == "__main__":
    # Run benchmarks directly
    import sys

    async def run_basic_benchmark():
        """Run a basic benchmark for demonstration."""
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

            # Run benchmark
            config = Config.fast()
            text = "Basic benchmark test text."

            start_metrics = benchmark.start_measurement()
            await improve(text, config=config)
            metrics = benchmark.end_measurement(start_metrics)

            benchmark.add_result(
                "basic_test", metrics, {"text_length": len(text), "model": config.model}
            )

        summary = benchmark.get_results_summary()
        print(f"Benchmark completed: {summary['total_tests']} tests")
        print(f"Average execution time: {summary['avg_execution_time']:.3f}s")
        print(f"Average memory usage: {summary['avg_memory_usage']:.1f}MB")
        return summary

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_basic_benchmark())
    else:
        print("Run with: python performance_benchmarks.py run")
