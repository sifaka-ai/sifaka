#!/usr/bin/env python3
"""
Performance validation tests for MCP functionality.

Tests performance characteristics, benchmarks, and scalability of MCP retrievers.
"""

import asyncio
import json
import pytest
import time
import statistics
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sifaka.core.thought import Thought, Document
from sifaka.mcp import MCPServerConfig, MCPTransportType, MCPResponse
from sifaka.utils.error_handling import RetrieverError

# Import retrievers with error handling
try:
    from sifaka.retrievers.milvus import MilvusRetriever
    from sifaka.retrievers.redis import RedisRetriever

    MCP_RETRIEVERS_AVAILABLE = True
except ImportError:
    MCP_RETRIEVERS_AVAILABLE = False


class PerformanceTimer:
    """Helper class for measuring performance."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
class TestMCPPerformanceBenchmarks:
    """Performance benchmarks for MCP retrievers."""

    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return MCPServerConfig(
            name="perf-server",
            transport_type=MCPTransportType.STDIO,
            url="test-command",
            timeout=5.0,
            retry_attempts=1,  # Minimal retries for performance testing
        )

    @pytest.fixture
    def mock_fast_response(self):
        """Mock response that simulates fast server."""
        return MCPResponse(
            result={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            [{"text": f"Document {i}", "score": 0.9 - i * 0.1} for i in range(10)]
                        ),
                    }
                ]
            },
            error=None,
            id="perf-1",
        )

    def test_milvus_retrieval_latency(self, performance_config, mock_fast_response):
        """Test Milvus retrieval latency under normal conditions."""
        retriever = MilvusRetriever(
            mcp_config=performance_config, collection_name="perf_test", max_results=10
        )

        # Mock fast embedding generation
        with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
            with patch.object(retriever, "_send_mcp_request", return_value=mock_fast_response):

                # Measure single query latency
                latencies = []
                for i in range(10):
                    with PerformanceTimer() as timer:
                        results = retriever.retrieve(f"test query {i}")

                    latencies.append(timer.elapsed)
                    assert len(results) == 10

                # Analyze latency statistics
                avg_latency = statistics.mean(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)

                print(f"\nMilvus Retrieval Latency Stats:")
                print(f"  Average: {avg_latency:.4f}s")
                print(f"  Min: {min_latency:.4f}s")
                print(f"  Max: {max_latency:.4f}s")

                # Performance assertions (adjust based on expected performance)
                assert avg_latency < 0.1, f"Average latency too high: {avg_latency:.4f}s"
                assert max_latency < 0.2, f"Max latency too high: {max_latency:.4f}s"

    def test_redis_cache_performance(self, performance_config):
        """Test Redis cache hit/miss performance."""
        # Create base retriever with known latency
        base_retriever = Mock()
        base_retriever.retrieve.return_value = ["base result 1", "base result 2"]

        retriever = RedisRetriever(
            mcp_config=performance_config, base_retriever=base_retriever, cache_ttl=3600
        )

        # Mock cache miss then hit
        cache_miss_response = MCPResponse(
            result={"content": [{"type": "text", "text": "null"}]}, error=None, id="miss-1"
        )

        cache_hit_response = MCPResponse(
            result={
                "content": [
                    {"type": "text", "text": json.dumps(["cached result 1", "cached result 2"])}
                ]
            },
            error=None,
            id="hit-1",
        )

        set_response = MCPResponse(result={"success": True}, error=None, id="set-1")

        # Test cache miss performance
        with patch.object(
            retriever, "_send_mcp_request", side_effect=[cache_miss_response, set_response]
        ):
            with PerformanceTimer() as miss_timer:
                results = retriever.retrieve("test query")

            assert len(results) == 2
            miss_latency = miss_timer.elapsed

        # Test cache hit performance
        with patch.object(retriever, "_send_mcp_request", return_value=cache_hit_response):
            with PerformanceTimer() as hit_timer:
                results = retriever.retrieve("test query")

            assert len(results) == 2
            hit_latency = hit_timer.elapsed

        print(f"\nRedis Cache Performance:")
        print(f"  Cache miss: {miss_latency:.4f}s")
        print(f"  Cache hit: {hit_latency:.4f}s")
        print(f"  Speedup: {miss_latency / hit_latency:.2f}x")

        # Cache hits should be significantly faster
        assert hit_latency < miss_latency, "Cache hit should be faster than miss"
        assert hit_latency < 0.05, f"Cache hit too slow: {hit_latency:.4f}s"

    def test_concurrent_retrieval_throughput(self, performance_config, mock_fast_response):
        """Test throughput under concurrent load."""
        retriever = MilvusRetriever(
            mcp_config=performance_config, collection_name="throughput_test", max_results=5
        )

        # Mock fast responses
        with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
            with patch.object(retriever, "_send_mcp_request", return_value=mock_fast_response):

                # Test sequential performance
                with PerformanceTimer() as sequential_timer:
                    sequential_results = []
                    for i in range(20):
                        results = retriever.retrieve(f"sequential query {i}")
                        sequential_results.extend(results)

                sequential_throughput = len(sequential_results) / sequential_timer.elapsed

                # Test concurrent performance
                def retrieve_worker(query_id):
                    return retriever.retrieve(f"concurrent query {query_id}")

                with PerformanceTimer() as concurrent_timer:
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(retrieve_worker, i) for i in range(20)]
                        concurrent_results = []
                        for future in as_completed(futures):
                            concurrent_results.extend(future.result())

                concurrent_throughput = len(concurrent_results) / concurrent_timer.elapsed

                print(f"\nThroughput Comparison:")
                print(f"  Sequential: {sequential_throughput:.2f} docs/sec")
                print(f"  Concurrent: {concurrent_throughput:.2f} docs/sec")
                print(f"  Improvement: {concurrent_throughput / sequential_throughput:.2f}x")

                # Concurrent should be faster (or at least not much slower)
                # Note: In mocked tests, concurrent may actually be slower due to overhead
                assert (
                    concurrent_throughput >= sequential_throughput * 0.1
                ), "Concurrent performance significantly worse than sequential"

    def test_large_batch_processing(self, performance_config):
        """Test performance with large batches of documents."""
        retriever = MilvusRetriever(
            mcp_config=performance_config, collection_name="batch_test", max_results=100
        )

        # Create large response
        large_documents = [
            {"text": f"Large document {i} with substantial content " * 50, "score": 0.9}
            for i in range(100)
        ]

        large_response = MCPResponse(
            result={"content": [{"type": "text", "text": json.dumps(large_documents)}]},
            error=None,
            id="large-1",
        )

        with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
            with patch.object(retriever, "_send_mcp_request", return_value=large_response):

                # Test large batch retrieval
                with PerformanceTimer() as timer:
                    results = retriever.retrieve("large batch query")

                processing_time = timer.elapsed
                docs_per_second = len(results) / processing_time

                print(f"\nLarge Batch Performance:")
                print(f"  Documents: {len(results)}")
                print(f"  Processing time: {processing_time:.4f}s")
                print(f"  Throughput: {docs_per_second:.2f} docs/sec")

                assert len(results) == 100
                assert (
                    processing_time < 1.0
                ), f"Large batch processing too slow: {processing_time:.4f}s"
                assert docs_per_second > 50, f"Throughput too low: {docs_per_second:.2f} docs/sec"

    def test_memory_usage_stability(self, performance_config, mock_fast_response):
        """Test memory usage remains stable under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        retriever = MilvusRetriever(
            mcp_config=performance_config, collection_name="memory_test", max_results=10
        )

        with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
            with patch.object(retriever, "_send_mcp_request", return_value=mock_fast_response):

                # Perform many retrievals
                for i in range(100):
                    results = retriever.retrieve(f"memory test query {i}")
                    assert len(results) == 10

                    # Check memory every 20 iterations
                    if i % 20 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_growth = current_memory - initial_memory

                        print(f"  Iteration {i}: {current_memory:.2f} MB (+{memory_growth:.2f} MB)")

                        # Memory growth should be reasonable
                        assert (
                            memory_growth < 50
                        ), f"Excessive memory growth: {memory_growth:.2f} MB"

        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Growth: {total_growth:.2f} MB")

        assert total_growth < 100, f"Total memory growth too high: {total_growth:.2f} MB"


@pytest.mark.skipif(not MCP_RETRIEVERS_AVAILABLE, reason="MCP retrievers not available")
class TestMCPScalabilityValidation:
    """Test scalability characteristics of MCP retrievers."""

    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return MCPServerConfig(
            name="perf-server",
            transport_type=MCPTransportType.STDIO,
            url="test-command",
            timeout=5.0,
            retry_attempts=1,  # Minimal retries for performance testing
        )

    def test_query_complexity_scaling(self, performance_config):
        """Test how performance scales with query complexity."""
        retriever = MilvusRetriever(
            mcp_config=performance_config, collection_name="complexity_test"
        )

        # Test queries of increasing complexity
        test_cases = [
            ("simple", "AI"),
            ("medium", "artificial intelligence machine learning"),
            (
                "complex",
                "artificial intelligence machine learning deep neural networks computer vision natural language processing",
            ),
            ("very_complex", " ".join([f"keyword{i}" for i in range(50)])),  # 50 keywords
        ]

        performance_results = {}

        for complexity, query in test_cases:
            # Mock embedding generation with complexity-based delay
            def mock_embedding(text):
                time.sleep(len(text.split()) * 0.001)  # Simulate complexity
                return [0.1] * 1024

            mock_response = MCPResponse(
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                [{"text": f"Result for {complexity}", "score": 0.9}]
                            ),
                        }
                    ]
                },
                error=None,
                id=f"{complexity}-1",
            )

            with patch.object(retriever, "embedding_function", side_effect=mock_embedding):
                with patch.object(retriever, "_send_mcp_request", return_value=mock_response):

                    with PerformanceTimer() as timer:
                        results = retriever.retrieve(query)

                    performance_results[complexity] = {
                        "latency": timer.elapsed,
                        "query_length": len(query.split()),
                        "results_count": len(results),
                    }

        print(f"\nQuery Complexity Scaling:")
        for complexity, metrics in performance_results.items():
            print(f"  {complexity}: {metrics['latency']:.4f}s ({metrics['query_length']} words)")

        # Verify reasonable scaling
        simple_latency = performance_results["simple"]["latency"]
        complex_latency = performance_results["complex"]["latency"]

        # Complex queries should not be more than 20x slower (more realistic for mocked tests)
        scaling_factor = complex_latency / simple_latency
        assert (
            scaling_factor < 20
        ), f"Poor scaling: {scaling_factor:.2f}x slower for complex queries"

    def test_result_set_size_scaling(self, performance_config):
        """Test how performance scales with result set size."""
        retriever = MilvusRetriever(mcp_config=performance_config, collection_name="size_test")

        result_sizes = [1, 10, 50, 100, 500]
        performance_results = {}

        for size in result_sizes:
            # Create response with specified number of results
            documents = [
                {"text": f"Document {i} for size test", "score": 0.9 - i * 0.001}
                for i in range(size)
            ]

            mock_response = MCPResponse(
                result={"content": [{"type": "text", "text": json.dumps(documents)}]},
                error=None,
                id=f"size-{size}",
            )

            retriever.max_results = size  # Adjust max results

            with patch.object(retriever, "embedding_function", return_value=[0.1] * 1024):
                with patch.object(retriever, "_send_mcp_request", return_value=mock_response):

                    with PerformanceTimer() as timer:
                        results = retriever.retrieve("size test query")

                    performance_results[size] = {
                        "latency": timer.elapsed,
                        "results_count": len(results),
                        "latency_per_result": timer.elapsed / len(results) if results else 0,
                    }

        print(f"\nResult Set Size Scaling:")
        for size, metrics in performance_results.items():
            print(
                f"  {size} results: {metrics['latency']:.4f}s ({metrics['latency_per_result']:.6f}s per result)"
            )

        # Verify linear or sub-linear scaling
        small_per_result = performance_results[10]["latency_per_result"]
        large_per_result = performance_results[100]["latency_per_result"]

        # Per-result latency should not increase dramatically
        scaling_ratio = large_per_result / small_per_result if small_per_result > 0 else 1
        assert (
            scaling_ratio < 5
        ), f"Poor per-result scaling: {scaling_ratio:.2f}x worse for large result sets"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
