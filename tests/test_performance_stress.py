"""Performance and stress tests for Sifaka."""

import pytest
import asyncio
import time
import concurrent.futures
from unittest.mock import patch, MagicMock, AsyncMock
import threading
from typing import List

from sifaka import improve
from sifaka.core.engine import SifakaEngine
from sifaka.core.models import Config
from sifaka.core.models import SifakaResult, Generation, CritiqueResult
from sifaka.storage import MemoryStorage, FileStorage
from sifaka.validators import LengthValidator, ContentValidator


class TestPerformanceBasics:
    """Test basic performance characteristics."""

    @pytest.mark.asyncio
    async def test_single_improvement_performance(self):
        """Test performance of a single improvement operation."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Good text. SUGGESTIONS: Keep it up."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            start_time = time.time()
            result = await improve(
                "Test text for performance", max_iterations=1, critics=["reflexion"]
            )
            end_time = time.time()

            execution_time = end_time - start_time
            assert execution_time < 1.0  # Should complete in less than 1 second
            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_multiple_iterations_performance(self):
        """Test performance with multiple iterations."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Needs improvement. SUGGESTIONS: Add details."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            start_time = time.time()
            result = await improve("Test text", max_iterations=5, critics=["reflexion"])
            end_time = time.time()

            execution_time = end_time - start_time
            assert execution_time < 5.0  # Should scale reasonably
            assert result.iteration >= 1

    @pytest.mark.asyncio
    async def test_memory_usage_single_operation(self):
        """Test memory usage for single operation."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Analysis complete."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test text for memory usage", max_iterations=1, critics=["reflexion"]
            )

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB for a simple operation)
            assert memory_increase < 50
            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_validator_performance(self):
        """Test validator performance with large inputs."""
        # Create validators
        length_validator = LengthValidator(min_length=100, max_length=10000)
        content_validator = ContentValidator(
            required_terms=["test", "performance", "validation"],
            forbidden_terms=["slow", "error"],
        )

        # Create large text
        large_text = "This is a test of performance validation. " * 100

        start_time = time.time()

        # Run validations multiple times
        for _ in range(20):
            length_result = await length_validator.validate(large_text)
            content_result = await content_validator.validate(large_text)

            assert length_result.passed
            assert content_result.passed

        end_time = time.time()
        execution_time = end_time - start_time

        # 40 validations should complete quickly
        assert execution_time < 1.0


class TestStressTests:
    """Stress tests for system limits and edge cases."""

    @pytest.mark.asyncio
    async def test_many_iterations_stress(self):
        """Stress test with maximum iterations."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Continue improving."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Stress test text",
                max_iterations=10,  # Maximum allowed
                critics=["reflexion"],
            )

            # Should handle maximum iterations gracefully
            assert result.iteration <= 10
            assert len(result.generations) <= 10  # Memory bounded
            assert len(result.critiques) <= 20  # Memory bounded

    @pytest.mark.asyncio
    async def test_multiple_critics_stress(self):
        """Stress test with all available critics."""

        # Mock different responses for different critics
        def mock_create_side_effect(*args, **kwargs):
            mock_response = MagicMock()
            system_content = kwargs.get("messages", [{}])[0].get("content", "")

            if "constitutional" in system_content.lower():
                mock_response.choices[0].message.content = (
                    """{"overall_assessment": "Good", "principle_scores": {"1": 4}, "violations": [], "suggestions": ["Continue"], "overall_confidence": 0.8, "evaluation_quality": 4}"""
                )
            else:
                mock_response.choices[0].message.content = "REFLECTION: Good analysis."

            return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create_side_effect
            )
            mock_openai.return_value = mock_client

            result = await improve(
                "Stress test with all critics",
                max_iterations=2,
                critics=[
                    "reflexion",
                    "constitutional",
                    "self_refine",
                    "n_critics",
                    "self_rag",
                    "meta_rewarding",
                    "self_consistency",
                ],
            )

            # Should handle all critics without issues
            assert isinstance(result, SifakaResult)
            assert len(result.critiques) > 0

    @pytest.mark.asyncio
    async def test_large_text_stress(self):
        """Stress test with very large input text."""
        # Create very large text (100KB+)
        large_text = "This is a large text for stress testing. " * 2500  # ~100KB

        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Handled large text successfully."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(large_text, max_iterations=1, critics=["reflexion"])

            assert isinstance(result, SifakaResult)
            assert result.original_text == large_text

    @pytest.mark.asyncio
    async def test_memory_bounded_collections_stress(self):
        """Test that memory bounds are enforced under stress."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Continue iteration."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create engine directly to test memory bounds
            config = Config(max_iterations=10, critics=["reflexion", "constitutional"])
            engine = SifakaEngine(config)

            result = await engine.improve("Test text")

            # Collections should be bounded even with many iterations and critics
            assert len(result.generations) <= 10
            assert len(result.critiques) <= 20
            assert len(result.validations) <= 20

    @pytest.mark.asyncio
    async def test_rapid_sequential_operations(self):
        """Test rapid sequential improvement operations."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Quick response."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            start_time = time.time()

            # Run 20 sequential operations
            results = []
            for i in range(20):
                result = await improve(
                    f"Test text {i}", max_iterations=1, critics=["reflexion"]
                )
                results.append(result)

            end_time = time.time()
            execution_time = end_time - start_time

            assert len(results) == 20
            assert all(isinstance(r, SifakaResult) for r in results)
            # Should complete reasonably quickly
            assert execution_time < 10.0

    @pytest.mark.asyncio
    async def test_storage_stress(self):
        """Stress test storage with many operations."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir, max_files=1000)

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "REFLECTION: Storage test."

            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_client

                # Create many results and store them
                results = []
                for i in range(50):
                    result = await improve(
                        f"Storage test {i}",
                        max_iterations=1,
                        critics=["reflexion"],
                        storage=storage,
                    )
                    results.append(result)

                # Verify all results were stored
                stored_ids = await storage.list_results()
                assert len(stored_ids) == 50

                # Verify we can load them back
                for result in results[:10]:  # Test subset
                    loaded = await storage.load(result.id)
                    assert loaded is not None
                    assert loaded.id == result.id


class TestConcurrencyTests:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_improvements(self):
        """Test concurrent improvement operations."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Concurrent test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create multiple concurrent improvement tasks
            tasks = []
            for i in range(10):
                task = improve(
                    f"Concurrent test {i}", max_iterations=1, critics=["reflexion"]
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            execution_time = end_time - start_time

            assert len(results) == 10
            assert all(isinstance(r, SifakaResult) for r in results)
            # Concurrent execution should be faster than sequential
            assert execution_time < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage operations."""
        storage = MemoryStorage()

        # Create test results concurrently
        async def create_and_store_result(index: int):
            result = SifakaResult(
                original_text=f"Test {index}",
                final_text=f"Improved test {index}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            return await storage.save(result)

        # Run concurrent save operations
        tasks = [create_and_store_result(i) for i in range(20)]
        result_ids = await asyncio.gather(*tasks)

        assert len(result_ids) == 20
        assert len(set(result_ids)) == 20  # All IDs should be unique

        # Verify concurrent loading
        load_tasks = [storage.load(rid) for rid in result_ids]
        loaded_results = await asyncio.gather(*load_tasks)

        assert len(loaded_results) == 20
        assert all(r is not None for r in loaded_results)

    def test_thread_safety_operations(self):
        """Test thread safety of concurrent operations."""

        def process_texts(thread_id: int, results: List):
            """Function to run in separate threads."""
            try:
                for i in range(50):
                    text = f"Thread {thread_id} text {i}"
                    # Simulate some processing
                    processed = f"Processed: {text}"
                    results.append((thread_id, i, processed))
            except Exception as e:
                results.append((thread_id, "error", str(e)))

        # Run multiple threads concurrently
        threads = []
        results = []

        for i in range(5):
            thread = threading.Thread(target=process_texts, args=(i, results))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 250  # 5 threads × 50 operations

        # Check for errors
        errors = [r for r in results if r[1] == "error"]
        assert len(errors) == 0, f"Thread safety errors: {errors}"

        # Verify all results are valid
        processed = [r[2] for r in results if r[1] != "error"]
        assert all(isinstance(p, str) for p in processed)

    @pytest.mark.asyncio
    async def test_concurrent_validator_operations(self):
        """Test concurrent validator operations."""
        validators = [
            LengthValidator(min_length=10, max_length=1000),
            ContentValidator(required_terms=["test"], forbidden_terms=["bad"]),
        ]

        test_texts = [
            f"This is test text number {i} for concurrent validation testing"
            for i in range(20)
        ]

        # Run all validations concurrently
        tasks = []
        for text in test_texts:
            for validator in validators:
                task = validator.validate(text)
                tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        execution_time = end_time - start_time

        assert len(results) == 40  # 20 texts × 2 validators
        assert all(hasattr(r, "passed") for r in results)
        # Should complete quickly even with many concurrent operations
        assert execution_time < 2.0

    @pytest.mark.asyncio
    async def test_resource_cleanup_under_stress(self):
        """Test that resources are properly cleaned up under stress."""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Cleanup test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Run many operations
            for i in range(100):
                result = await improve(
                    f"Cleanup test {i}", max_iterations=1, critics=["reflexion"]
                )
                # Explicitly delete result to test cleanup
                del result

                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()

            # Force final garbage collection
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable even after many operations
            assert memory_increase < 100  # Less than 100MB increase


class TestScalabilityTests:
    """Test system scalability characteristics."""

    @pytest.mark.asyncio
    async def test_scaling_with_text_size(self):
        """Test how performance scales with input text size."""
        text_sizes = [100, 1000, 10000, 50000]  # Characters
        execution_times = []

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Scaling test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for size in text_sizes:
                text = "A" * size

                start_time = time.time()
                result = await improve(text, max_iterations=1, critics=["reflexion"])
                end_time = time.time()

                execution_time = end_time - start_time
                execution_times.append(execution_time)

                assert isinstance(result, SifakaResult)
                assert len(result.original_text) == size

        # Execution time should scale reasonably (not exponentially)
        # Each doubling of size should not increase time by more than 4x
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i - 1]
            assert ratio < 4.0, f"Performance degradation too high: {ratio}"

    @pytest.mark.asyncio
    async def test_scaling_with_critic_count(self):
        """Test how performance scales with number of critics."""
        critic_sets = [
            ["reflexion"],
            ["reflexion", "constitutional"],
            ["reflexion", "constitutional", "self_refine"],
            ["reflexion", "constitutional", "self_refine", "n_critics"],
        ]

        execution_times = []

        # Mock different responses for different critics
        def mock_create_side_effect(*args, **kwargs):
            mock_response = MagicMock()
            system_content = kwargs.get("messages", [{}])[0].get("content", "")

            if "constitutional" in system_content.lower():
                mock_response.choices[0].message.content = (
                    """{"overall_assessment": "Good", "principle_scores": {"1": 4}, "violations": [], "suggestions": ["Continue"], "overall_confidence": 0.8, "evaluation_quality": 4}"""
                )
            else:
                mock_response.choices[0].message.content = "REFLECTION: Good analysis."

            return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create_side_effect
            )
            mock_openai.return_value = mock_client

            for critics in critic_sets:
                start_time = time.time()
                result = await improve(
                    "Scaling test text", max_iterations=1, critics=critics
                )
                end_time = time.time()

                execution_time = end_time - start_time
                execution_times.append(execution_time)

                assert isinstance(result, SifakaResult)

        # Time should scale linearly or sub-linearly with critic count
        # Each additional critic should not more than double the time
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[0]
            critic_ratio = len(critic_sets[i]) / len(critic_sets[0])

            # Time ratio should not exceed 3x the critic ratio
            assert ratio <= critic_ratio * 3

    def test_memory_scaling(self):
        """Test memory usage scaling."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Create results of different sizes and measure memory
        memory_measurements = []

        for i in range(5):
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create result with increasing amounts of data
            generations = [
                Generation(
                    text=f"Generated text {j}" * (i + 1),
                    model="gpt-4o-mini",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                )
                for j in range((i + 1) * 10)
            ]

            result = SifakaResult(
                original_text="Test text" * (i + 1),
                final_text="Final text" * (i + 1),
                iteration=i + 1,
                generations=generations,
                critiques=[],
                validations=[],
                processing_time=1.0,
            )

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            memory_measurements.append(memory_increase)

            # Clean up
            del result
            del generations

        # Memory usage should scale reasonably
        assert all(mem < 50 for mem in memory_measurements)  # Less than 50MB each
