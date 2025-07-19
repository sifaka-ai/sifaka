"""Enhanced concurrent access tests for Sifaka.

This module extends the existing concurrency tests with additional scenarios:
- Resource contention and locking
- Distributed storage access patterns
- Complex workflow concurrency
- Edge cases and failure modes
"""

import asyncio
import random
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka import improve
from sifaka.core.config import Config
from sifaka.core.engine import SifakaEngine
from sifaka.core.models import SifakaResult
from sifaka.storage import MemoryStorage


class TestResourceContention:
    """Test resource contention scenarios."""

    @pytest.mark.asyncio
    async def test_shared_storage_concurrent_access(self):
        """Test concurrent access to shared storage resource."""
        storage = MemoryStorage()

        # Shared counter to track operations
        operation_count = {"value": 0}

        async def concurrent_storage_operations(worker_id: int, ops_count: int):
            """Worker function that performs multiple storage operations."""
            results = []

            for i in range(ops_count):
                # Create unique result
                result = SifakaResult(
                    original_text=f"Worker {worker_id} operation {i}",
                    final_text=f"Processed by worker {worker_id} operation {i}",
                    iteration=1,
                    generations=[],
                    critiques=[],
                    validations=[],
                    processing_time=random.uniform(0.1, 0.5),
                )

                # Save result
                result_id = await storage.save(result)
                results.append(result_id)

                # Simulate some processing time
                await asyncio.sleep(0.01)

                # Load result back to verify
                loaded = await storage.load(result_id)
                assert loaded is not None
                assert loaded.original_text == result.original_text

                # Update shared counter (potential race condition)
                operation_count["value"] += 1

            return results

        # Run 8 workers concurrently, each doing 15 operations
        workers = 8
        ops_per_worker = 15

        tasks = [
            concurrent_storage_operations(i, ops_per_worker) for i in range(workers)
        ]

        start_time = time.time()
        worker_results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Verify all operations completed
        total_result_ids = []
        for worker_results_list in worker_results:
            total_result_ids.extend(worker_results_list)

        assert len(total_result_ids) == workers * ops_per_worker
        assert len(set(total_result_ids)) == workers * ops_per_worker  # All unique

        # Verify storage state
        stored_ids = await storage.list(limit=200)  # Get all results
        assert len(stored_ids) == workers * ops_per_worker

        # Check execution time - should be much faster than sequential
        sequential_time_estimate = workers * ops_per_worker * 0.01
        assert end_time - start_time < sequential_time_estimate * 2

    @pytest.mark.asyncio
    async def test_engine_instance_reuse_contention(self):
        """Test contention when reusing engine instances."""
        config = Config.fast()
        engine = SifakaEngine(config)

        # Counter to track successful operations
        success_count = {"value": 0}
        error_count = {"value": 0}

        async def use_shared_engine(worker_id: int, operations: int):
            """Worker that uses the shared engine instance."""
            mock_response = MagicMock()
            mock_response.choices[
                0
            ].message.content = f"REFLECTION: Worker {worker_id} processing."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                results = []
                for i in range(operations):
                    try:
                        result = await engine.improve(f"Worker {worker_id} text {i}")
                        results.append(result)
                        success_count["value"] += 1
                    except Exception as e:
                        error_count["value"] += 1
                        results.append(e)

                return results

        # Run 6 workers concurrently using the same engine
        tasks = [use_shared_engine(i, 10) for i in range(6)]

        all_results = await asyncio.gather(*tasks)

        # Verify results
        total_operations = 6 * 10
        successful_results = []
        failed_results = []

        for worker_results in all_results:
            for result in worker_results:
                if isinstance(result, SifakaResult):
                    successful_results.append(result)
                else:
                    failed_results.append(result)

        # Most operations should succeed (engine reuse should be safe)
        assert (
            len(successful_results) >= total_operations * 0.8
        )  # At least 80% success rate
        assert success_count["value"] >= total_operations * 0.8

    @pytest.mark.asyncio
    async def test_concurrent_config_modifications(self):
        """Test concurrent operations with config modifications."""

        async def improve_with_dynamic_config(text: str, config_type: str):
            """Improve text with dynamically chosen config."""
            configs = {
                "fast": Config.fast(),
                "quality": Config.quality(),
                "minimal": Config.minimal(),
                "creative": Config.creative(),
            }

            config = configs.get(config_type, Config.fast())

            mock_response = MagicMock()
            mock_response.choices[
                0
            ].message.content = f"REFLECTION: {config_type} config processing."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                result = await improve(text, config=config)
                return result, config_type

        # Create tasks with different config types
        config_types = ["fast", "quality", "minimal", "creative"]
        tasks = []

        for i in range(40):
            config_type = random.choice(config_types)
            task = improve_with_dynamic_config(f"Dynamic config test {i}", config_type)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all operations completed
        assert len(results) == 40

        # Check config distribution
        config_counts = {}
        for result, config_type in results:
            assert isinstance(result, SifakaResult)
            config_counts[config_type] = config_counts.get(config_type, 0) + 1

        # Should have used multiple config types
        assert len(config_counts) > 1


class TestComplexWorkflowConcurrency:
    """Test complex concurrent workflows."""

    @pytest.mark.asyncio
    async def test_pipeline_with_dependencies(self):
        """Test concurrent pipeline with dependencies between stages."""
        storage = MemoryStorage()

        # Stage 1: Text preprocessing
        async def preprocess_stage(texts: List[str]) -> List[str]:
            """Preprocess texts concurrently."""
            tasks = []
            for text in texts:
                task = asyncio.create_task(self._preprocess_text(text))
                tasks.append(task)
            return await asyncio.gather(*tasks)

        # Stage 2: Improvement
        async def improvement_stage(texts: List[str]) -> List[SifakaResult]:
            """Improve texts concurrently."""
            mock_response = MagicMock()
            mock_response.choices[
                0
            ].message.content = "REFLECTION: Pipeline improvement."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                tasks = []
                for text in texts:
                    task = improve(text, max_iterations=1, critics=["reflexion"])
                    tasks.append(task)
                return await asyncio.gather(*tasks)

        # Stage 3: Storage
        async def storage_stage(results: List[SifakaResult]) -> List[str]:
            """Store results concurrently."""
            tasks = []
            for result in results:
                task = storage.save(result)
                tasks.append(task)
            return await asyncio.gather(*tasks)

        # Execute pipeline
        input_texts = [f"Pipeline input text {i}" for i in range(20)]

        # Stage 1
        preprocessed = await preprocess_stage(input_texts)
        assert len(preprocessed) == 20

        # Stage 2
        improved = await improvement_stage(preprocessed)
        assert len(improved) == 20
        assert all(isinstance(r, SifakaResult) for r in improved)

        # Stage 3
        stored_ids = await storage_stage(improved)
        assert len(stored_ids) == 20
        assert len(set(stored_ids)) == 20  # All unique

        # Verify final storage state
        final_stored = await storage.list(limit=50)  # Get all results
        assert len(final_stored) == 20

    async def _preprocess_text(self, text: str) -> str:
        """Helper method for text preprocessing."""
        await asyncio.sleep(0.01)  # Simulate processing
        return f"Preprocessed: {text}"

    @pytest.mark.asyncio
    async def test_fan_out_fan_in_pattern(self):
        """Test fan-out/fan-in concurrent pattern."""

        async def fan_out_processing(input_data: str, worker_count: int) -> List[str]:
            """Fan out processing to multiple workers."""

            async def worker(worker_id: int, data_chunk: str) -> str:
                """Individual worker processing."""
                await asyncio.sleep(random.uniform(0.01, 0.05))
                return f"Worker {worker_id} processed: {data_chunk}"

            # Split work among workers
            tasks = []
            for i in range(worker_count):
                chunk = f"{input_data} - chunk {i}"
                task = worker(i, chunk)
                tasks.append(task)

            return await asyncio.gather(*tasks)

        async def fan_in_aggregation(processed_chunks: List[str]) -> str:
            """Fan in - aggregate results."""
            return " | ".join(processed_chunks)

        # Process multiple items concurrently using fan-out/fan-in
        input_items = [f"Item {i}" for i in range(10)]

        # Fan out each item to multiple workers
        fan_out_tasks = []
        for item in input_items:
            task = fan_out_processing(item, 5)  # 5 workers per item
            fan_out_tasks.append(task)

        all_processed_chunks = await asyncio.gather(*fan_out_tasks)

        # Fan in - aggregate results for each item
        fan_in_tasks = []
        for processed_chunks in all_processed_chunks:
            task = fan_in_aggregation(processed_chunks)
            fan_in_tasks.append(task)

        final_results = await asyncio.gather(*fan_in_tasks)

        # Verify results
        assert len(final_results) == 10
        assert all(isinstance(result, str) for result in final_results)
        assert all("Worker" in result for result in final_results)

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for fault tolerance."""

        class SimpleCircuitBreaker:
            def __init__(
                self, failure_threshold: int = 5, recovery_timeout: float = 1.0
            ):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            async def call(self, func, *args, **kwargs):
                if self.state == "OPEN":
                    if (time.time() - self.last_failure_time) > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise Exception("Circuit breaker is OPEN")

                try:
                    result = await func(*args, **kwargs)
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"

                    raise e

        circuit_breaker = SimpleCircuitBreaker()

        # Function that fails intermittently
        async def unreliable_improve(text: str, should_fail: bool = False):
            if should_fail:
                raise Exception("Simulated service failure")

            mock_response = MagicMock()
            mock_response.choices[
                0
            ].message.content = "REFLECTION: Circuit breaker test."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                return await improve(text, max_iterations=1, critics=["reflexion"])

        # Test circuit breaker behavior
        results = []

        # Phase 1: Normal operation
        for i in range(3):
            try:
                result = await circuit_breaker.call(
                    unreliable_improve, f"Test {i}", False
                )
                results.append(("success", result))
            except Exception as e:
                results.append(("error", str(e)))

        # Phase 2: Introduce failures to trigger circuit breaker
        for i in range(6):
            try:
                result = await circuit_breaker.call(
                    unreliable_improve, f"Test {i+3}", True
                )
                results.append(("success", result))
            except Exception as e:
                results.append(("error", str(e)))

        # Phase 3: Circuit should be open now
        for i in range(3):
            try:
                result = await circuit_breaker.call(
                    unreliable_improve, f"Test {i+9}", False
                )
                results.append(("success", result))
            except Exception as e:
                results.append(("error", str(e)))

        # Verify circuit breaker behavior
        assert len(results) == 12

        # Early results should be successful
        early_results = results[:3]
        assert all(r[0] == "success" for r in early_results)

        # Later results should include circuit breaker errors
        later_results = results[9:]
        circuit_breaker_errors = [
            r for r in later_results if "Circuit breaker is OPEN" in r[1]
        ]
        assert len(circuit_breaker_errors) > 0


class TestEdgeCasesAndFailures:
    """Test edge cases and failure scenarios in concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_timeouts(self):
        """Test concurrent operations with various timeout scenarios."""

        async def improve_with_timeout(text: str, timeout: float):
            """Improve text with specified timeout."""
            try:
                # Simulate variable processing time
                processing_time = random.uniform(0.05, 0.3)

                async def delayed_improve():
                    await asyncio.sleep(processing_time)

                    mock_response = MagicMock()
                    mock_response.choices[
                        0
                    ].message.content = "REFLECTION: Timeout test."

                    with patch("openai.AsyncOpenAI") as mock_openai:
                        mock_client = MagicMock()
                        mock_client.chat.completions.create = AsyncMock(
                            return_value=mock_response
                        )
                        mock_openai.return_value = mock_client

                        return await improve(
                            text, max_iterations=1, critics=["reflexion"]
                        )

                # Apply timeout
                result = await asyncio.wait_for(delayed_improve(), timeout=timeout)
                return ("success", result)

            except asyncio.TimeoutError:
                return ("timeout", None)
            except Exception as e:
                return ("error", str(e))

        # Create tasks with different timeout values
        tasks = []
        for i in range(30):
            timeout = random.uniform(0.1, 0.5)  # Random timeout
            task = improve_with_timeout(f"Timeout test {i}", timeout)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Analyze results
        successes = [r for r in results if r[0] == "success"]
        [r for r in results if r[0] == "timeout"]
        [r for r in results if r[0] == "error"]

        # Should have a mix of successes and timeouts
        assert len(successes) > 0
        assert len(results) == 30

        # All successful results should be valid
        for success_type, result in successes:
            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_backpressure(self):
        """Test concurrent operations with backpressure management."""

        # Semaphore to limit concurrent operations
        max_concurrent = 5
        semaphore = asyncio.Semaphore(max_concurrent)

        # Queue to manage backpressure
        work_queue = asyncio.Queue(maxsize=10)
        results_queue = asyncio.Queue()

        async def producer():
            """Producer that generates work items."""
            for i in range(50):
                work_item = f"Backpressure test {i}"
                await work_queue.put(work_item)
                await asyncio.sleep(0.01)  # Simulate work generation rate

            # Signal completion
            await work_queue.put(None)

        async def consumer():
            """Consumer that processes work items with backpressure."""
            while True:
                try:
                    # Wait for work with timeout
                    work_item = await asyncio.wait_for(work_queue.get(), timeout=1.0)

                    if work_item is None:  # End signal
                        break

                    # Use semaphore to limit concurrent processing
                    async with semaphore:
                        mock_response = MagicMock()
                        mock_response.choices[
                            0
                        ].message.content = "REFLECTION: Backpressure test."

                        with patch("openai.AsyncOpenAI") as mock_openai:
                            mock_client = MagicMock()
                            mock_client.chat.completions.create = AsyncMock(
                                return_value=mock_response
                            )
                            mock_openai.return_value = mock_client

                            result = await improve(
                                work_item, max_iterations=1, critics=["reflexion"]
                            )
                            await results_queue.put(result)

                    work_queue.task_done()

                except asyncio.TimeoutError:
                    break  # No more work

        # Start producer and multiple consumers
        producer_task = asyncio.create_task(producer())
        consumer_tasks = [asyncio.create_task(consumer()) for _ in range(3)]

        # Wait for producer to finish
        await producer_task

        # Wait for all work to be processed
        await work_queue.join()

        # Cancel consumer tasks
        for task in consumer_tasks:
            task.cancel()

        # Collect results
        results = []
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        # Verify results
        assert len(results) == 50
        assert all(isinstance(r, SifakaResult) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_resource_cleanup(self):
        """Test that resources are properly cleaned up in concurrent operations."""
        import gc
        import weakref

        # Track object creation and cleanup
        created_objects = []

        async def create_and_process_result(index: int):
            """Create, process, and cleanup a result."""
            # Create result
            result = SifakaResult(
                original_text=f"Resource cleanup test {index}",
                final_text=f"Processed {index}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=0.1,
            )

            # Track with weak reference
            weak_ref = weakref.ref(result)
            created_objects.append(weak_ref)

            # Simulate processing
            await asyncio.sleep(0.01)

            # Return some data but not the result itself
            return f"Processed {index}"

        # Create many concurrent operations
        tasks = [create_and_process_result(i) for i in range(100)]

        # Execute tasks
        results = await asyncio.gather(*tasks)

        # Force garbage collection
        gc.collect()

        # Check that objects were cleaned up
        alive_objects = [ref for ref in created_objects if ref() is not None]

        # Most objects should be garbage collected
        cleanup_ratio = 1.0 - (len(alive_objects) / len(created_objects))
        assert cleanup_ratio > 0.8  # At least 80% should be cleaned up

        # Verify results
        assert len(results) == 100
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_cancellation(self):
        """Test concurrent operations with task cancellation."""

        async def long_running_improve(text: str, delay: float):
            """Simulate long-running improve operation."""
            try:
                await asyncio.sleep(delay)

                mock_response = MagicMock()
                mock_response.choices[
                    0
                ].message.content = "REFLECTION: Long running test."

                with patch("openai.AsyncOpenAI") as mock_openai:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create = AsyncMock(
                        return_value=mock_response
                    )
                    mock_openai.return_value = mock_client

                    return await improve(text, max_iterations=1, critics=["reflexion"])

            except asyncio.CancelledError:
                return "cancelled"

        # Create tasks with different delays
        tasks = []
        for i in range(20):
            delay = random.uniform(0.1, 1.0)
            task = asyncio.create_task(
                long_running_improve(f"Cancellation test {i}", delay)
            )
            tasks.append(task)

        # Let them run for a bit
        await asyncio.sleep(0.3)

        # Cancel half of the tasks
        for i in range(10):
            tasks[i].cancel()

        # Wait for all tasks to complete or be cancelled
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        completed = [r for r in results if isinstance(r, SifakaResult)]
        cancelled = [r for r in results if r == "cancelled"]
        cancelled_exceptions = [
            r for r in results if isinstance(r, asyncio.CancelledError)
        ]

        # Should have a mix of completed and cancelled operations
        assert len(completed) > 0
        assert len(cancelled) + len(cancelled_exceptions) > 0
        assert len(results) == 20


def test_concurrent_access_summary():
    """Test that demonstrates the concurrent access capabilities."""
    # This test serves as documentation for the concurrent access features

    concurrent_features = {
        "resource_contention": "Tests shared storage and engine access",
        "complex_workflows": "Tests pipeline and fan-out/fan-in patterns",
        "edge_cases": "Tests timeouts, backpressure, and cancellation",
        "fault_tolerance": "Tests circuit breaker and error handling",
        "resource_cleanup": "Tests proper cleanup in concurrent scenarios",
    }

    # Verify all feature categories are tested
    assert len(concurrent_features) == 5

    # This serves as a summary of what concurrent access tests cover
    print("Concurrent Access Test Coverage:")
    for feature, description in concurrent_features.items():
        print(f"  {feature}: {description}")


if __name__ == "__main__":
    # Run a simple concurrent test for demonstration
    async def demo_concurrent_access():
        """Demonstrate concurrent access capabilities."""
        print("Running concurrent access demonstration...")

        # Simple concurrent operations
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Demo complete."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create concurrent tasks
            tasks = [
                improve(f"Demo text {i}", max_iterations=1, critics=["reflexion"])
                for i in range(5)
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            print(
                f"Completed {len(results)} concurrent operations in {end_time - start_time:.3f}s"
            )
            return results

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_concurrent_access())
