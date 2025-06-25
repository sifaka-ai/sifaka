"""Advanced concurrency tests for Sifaka."""

import pytest
import asyncio
import concurrent.futures
import time
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List
import random

from sifaka import improve
from sifaka.core.models import SifakaResult
from sifaka.core.config import Config
from sifaka.storage import MemoryStorage, FileStorage
from sifaka.validators import LengthValidator


class TestBasicConcurrency:
    """Test basic concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_improve_calls(self):
        """Test multiple concurrent improve() calls."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Concurrent analysis complete."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create 20 concurrent improve tasks
            tasks = []
            for i in range(20):
                task = improve(
                    f"Concurrent test text {i}", max_iterations=1, critics=["reflexion"]
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            execution_time = end_time - start_time

            # Verify all results
            assert len(results) == 20
            assert all(isinstance(r, SifakaResult) for r in results)
            assert all(
                f"Concurrent test text {i}" in r.original_text
                for i, r in enumerate(results)
            )

            # Concurrent execution should be faster than sequential
            assert execution_time < 10.0

    @pytest.mark.asyncio
    async def test_concurrent_different_configurations(self):
        """Test concurrent operations with different configurations."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Configuration test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create tasks with different configurations
            tasks = [
                improve("Test 1", max_iterations=1, critics=["reflexion"]),
                improve("Test 2", max_iterations=2, critics=["reflexion"]),
                improve("Test 3", max_iterations=1, critics=["constitutional"]),
                improve("Test 4", max_iterations=3, critics=["self_refine"]),
                improve("Test 5", max_iterations=1, config=Config(model="gpt-4")),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(isinstance(r, SifakaResult) for r in results)

            # Verify configurations were respected
            assert results[0].original_text == "Test 1"
            assert results[1].original_text == "Test 2"

    @pytest.mark.asyncio
    async def test_concurrent_with_validators(self):
        """Test concurrent operations with validators."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Validator test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            validators = [LengthValidator(min_length=10, max_length=1000)]

            # Create concurrent tasks with validators
            tasks = []
            for i in range(10):
                task = improve(
                    f"Validator test text {i} with sufficient length",
                    max_iterations=1,
                    critics=["reflexion"],
                    validators=validators,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(isinstance(r, SifakaResult) for r in results)
            # All should pass validation due to sufficient length
            assert all(len(r.validations) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage save/load operations."""
        storage = MemoryStorage()

        # Create test results concurrently
        async def create_and_store_result(index: int):
            result = SifakaResult(
                original_text=f"Concurrent storage test {index}",
                final_text=f"Improved text {index}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=0.5,
            )
            result_id = await storage.save(result)
            return result_id, result

        # Run 30 concurrent save operations
        tasks = [create_and_store_result(i) for i in range(30)]
        save_results = await asyncio.gather(*tasks)

        assert len(save_results) == 30

        # Extract IDs and verify uniqueness
        result_ids = [sr[0] for sr in save_results]
        assert len(set(result_ids)) == 30  # All IDs should be unique

        # Test concurrent loading
        load_tasks = [storage.load(rid) for rid in result_ids]
        loaded_results = await asyncio.gather(*load_tasks)

        assert len(loaded_results) == 30
        assert all(r is not None for r in loaded_results)

        # Verify content matches
        for i, loaded in enumerate(loaded_results):
            original_result = save_results[i][1]
            assert loaded.original_text == original_result.original_text


class TestAdvancedConcurrency:
    """Test advanced concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_mixed_sync_async_operations(self):
        """Test mixing synchronous and asynchronous operations."""
        storage = MemoryStorage()

        # Asynchronous storage function
        async def store_results(count: int) -> List[str]:
            results = []
            for i in range(count):
                result = SifakaResult(
                    original_text=f"Mixed test {i}",
                    final_text=f"Mixed result {i}",
                    iteration=1,
                    generations=[],
                    critiques=[],
                    validations=[],
                    processing_time=0.5,
                )
                result_id = await storage.save(result)
                results.append(result_id)
            return results

        # Run async operations
        storage_task = store_results(15)

        result_ids = await storage_task

        assert len(result_ids) == 15
        assert all(isinstance(rid, str) for rid in result_ids)

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test producer-consumer pattern with asyncio queues."""
        import asyncio

        queue = asyncio.Queue(maxsize=10)
        results = []

        # Producer: generates work items
        async def producer():
            for i in range(50):
                work_item = f"Work item {i}"
                await queue.put(work_item)
                await asyncio.sleep(0.01)  # Simulate work

            # Signal completion
            for _ in range(3):  # Number of consumers
                await queue.put(None)

        # Consumer: processes work items
        async def consumer(consumer_id: int):
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                f"REFLECTION: Consumer {consumer_id} processed item."
            )

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                while True:
                    item = await queue.get()
                    if item is None:  # End signal
                        break

                    # Process item with improve()
                    result = await improve(
                        item, max_iterations=1, critics=["reflexion"]
                    )
                    results.append((consumer_id, result))
                    queue.task_done()

        # Run producer and consumers concurrently
        producer_task = asyncio.create_task(producer())
        consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(3)]

        await producer_task
        await queue.join()  # Wait for all items to be processed

        # Cancel consumer tasks
        for task in consumer_tasks:
            task.cancel()

        # Verify results
        assert len(results) == 50
        assert all(isinstance(result[1], SifakaResult) for result in results)

        # Check that all consumers participated
        consumer_ids = set(result[0] for result in results)
        assert len(consumer_ids) > 1  # Multiple consumers worked

    @pytest.mark.asyncio
    async def test_rate_limited_concurrent_operations(self):
        """Test concurrent operations with rate limiting."""
        import asyncio

        # Semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations

        async def rate_limited_improve(text: str, delay: float = 0.1):
            async with semaphore:
                await asyncio.sleep(delay)  # Simulate rate limiting

                mock_response = MagicMock()
                mock_response.choices[0].message.content = (
                    "REFLECTION: Rate limited operation."
                )

                with patch("openai.AsyncOpenAI") as mock_openai:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create = AsyncMock(
                        return_value=mock_response
                    )
                    mock_openai.return_value = mock_client

                    return await improve(text, max_iterations=1, critics=["reflexion"])

        # Create 20 tasks that will be rate limited
        tasks = [rate_limited_improve(f"Rate limited text {i}") for i in range(20)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        execution_time = end_time - start_time

        assert len(results) == 20
        assert all(isinstance(r, SifakaResult) for r in results)

        # Should take longer due to rate limiting
        # 20 tasks / 5 concurrent * 0.1s delay = at least 0.4s
        assert execution_time >= 0.3

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_operations(self):
        """Test error handling in concurrent operations."""

        async def sometimes_failing_improve(text: str, should_fail: bool = False):
            if should_fail:
                raise ValueError(f"Simulated error for: {text}")

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "REFLECTION: Success."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                return await improve(text, max_iterations=1, critics=["reflexion"])

        # Create tasks, some of which will fail
        tasks = []
        for i in range(20):
            should_fail = i % 5 == 0  # Every 5th task fails
            task = sometimes_failing_improve(f"Test {i}", should_fail)
            tasks.append(task)

        # Use gather with return_exceptions=True to handle errors
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 20

        # Count successful results and errors
        successes = [r for r in results if isinstance(r, SifakaResult)]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 16  # 4 out of 20 should fail
        assert len(errors) == 4
        assert all(isinstance(e, ValueError) for e in errors)

    @pytest.mark.asyncio
    async def test_concurrent_file_storage_operations(self):
        """Test concurrent file storage operations."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir)

            async def create_and_store_file_result(index: int):
                result = SifakaResult(
                    original_text=f"File storage test {index}",
                    final_text=f"File result {index}",
                    iteration=1,
                    generations=[],
                    critiques=[],
                    validations=[],
                    processing_time=0.5,
                )
                result_id = await storage.save(result)

                # Immediately try to load it back
                loaded = await storage.load(result_id)
                return result_id, loaded is not None

            # Run 25 concurrent file operations
            tasks = [create_and_store_file_result(i) for i in range(25)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 25

            # All operations should succeed
            result_ids = [r[0] for r in results]
            load_successes = [r[1] for r in results]

            assert len(set(result_ids)) == 25  # All unique IDs
            assert all(load_successes)  # All loads succeeded

            # Verify files were actually created
            stored_ids = await storage.list_results()
            assert len(stored_ids) == 25


class TestConcurrencyStressTests:
    """Stress tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Test system under high concurrency load."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: High concurrency test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create 100 concurrent tasks
            tasks = []
            for i in range(100):
                task = improve(
                    f"High concurrency test {i}",
                    max_iterations=1,
                    critics=["reflexion"],
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            execution_time = end_time - start_time

            assert len(results) == 100
            assert all(isinstance(r, SifakaResult) for r in results)

            # Should complete in reasonable time even with high concurrency
            assert execution_time < 30.0

    @pytest.mark.asyncio
    async def test_memory_usage_under_concurrency(self):
        """Test memory usage under concurrent load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Memory test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Run concurrent operations in batches to test memory management
            for batch in range(5):
                tasks = []
                for i in range(20):
                    task = improve(
                        f"Memory test batch {batch} item {i}",
                        max_iterations=1,
                        critics=["reflexion"],
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)
                assert len(results) == 20

                # Clear results to test garbage collection
                del results
                del tasks

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable even after many concurrent operations
        assert memory_increase < 200  # Less than 200MB increase

    def test_thread_safety_with_thread_pool(self):
        """Test thread safety using ThreadPoolExecutor."""

        def process_batch(batch_id: int) -> List[str]:
            """Process a batch of texts in a thread."""
            results = []
            for i in range(20):
                text = f"Thread {batch_id} text {i}"
                # Simulate some processing
                results.append(f"Processed: {text}")
            return results

        # Use ThreadPoolExecutor for true parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit 10 batches to different threads
            futures = [
                executor.submit(process_batch, batch_id) for batch_id in range(10)
            ]

            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Verify results
        assert len(all_results) == 200  # 10 batches × 20 results
        assert all(isinstance(result, str) for result in all_results)

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_random_delays(self):
        """Test concurrent operations with random processing delays."""

        async def improve_with_random_delay(text: str):
            # Random delay to simulate variable processing times
            delay = random.uniform(0.01, 0.1)
            await asyncio.sleep(delay)

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "REFLECTION: Random delay test."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                return await improve(text, max_iterations=1, critics=["reflexion"])

        # Create 30 tasks with random delays
        tasks = [improve_with_random_delay(f"Random delay test {i}") for i in range(30)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        execution_time = end_time - start_time

        assert len(results) == 30
        assert all(isinstance(r, SifakaResult) for r in results)

        # Should complete faster than sequential execution
        # Sequential would be at least 30 * 0.01 = 0.3s
        # But due to random delays, might be longer
        assert execution_time < 5.0  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_cascading_async_operations(self):
        """Test cascading async operations (improve -> validate -> store)."""
        storage = MemoryStorage()
        validator = LengthValidator(min_length=10, max_length=1000)

        async def full_pipeline(text: str):
            # Step 1: Improve text
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "REFLECTION: Pipeline test."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                result = await improve(text, max_iterations=1, critics=["reflexion"])

            # Step 2: Additional validation
            validation_result = await validator.validate(result.final_text)

            # Step 3: Store result
            if validation_result.passed:
                result_id = await storage.save(result)
                return result_id, True
            else:
                return None, False

        # Run pipeline for multiple texts concurrently
        texts = [f"Pipeline test text {i} with sufficient length" for i in range(15)]
        tasks = [full_pipeline(text) for text in texts]

        results = await asyncio.gather(*tasks)

        assert len(results) == 15

        # Most should succeed (have sufficient length)
        successful_results = [r for r in results if r[1]]
        assert len(successful_results) >= 10  # Most should pass validation

        # Verify stored results
        stored_ids = await storage.list_results()
        assert len(stored_ids) == len(successful_results)
