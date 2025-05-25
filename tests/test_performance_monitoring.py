#!/usr/bin/env python3
"""
Tests for performance monitoring functionality in Sifaka.

This module tests the performance monitoring capabilities added to the Chain class,
ensuring that timing data is collected correctly and performance analysis works as expected.
"""

import pytest
from sifaka.core.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.retrievers import MockRetriever
from sifaka.utils.performance import PerformanceMonitor


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear performance data before each test
        monitor = PerformanceMonitor.get_instance()
        monitor.clear()

    def test_chain_performance_tracking(self):
        """Test that chain execution is tracked for performance."""
        # Create a basic chain
        model = create_model("mock:test")
        retriever = MockRetriever()
        validator = LengthValidator(min_length=10, max_length=500)

        chain = Chain(
            model=model,
            prompt="Test prompt for performance monitoring",
            model_retrievers=[retriever],
        )
        chain.validate_with(validator)

        # Clear performance data
        chain.clear_performance_data()

        # Run the chain
        thought = chain.run()

        # Get performance summary
        summary = chain.get_performance_summary()

        # Verify that performance data was collected
        assert summary["total_operations"] > 0
        assert summary["total_time"] > 0
        assert "operations" in summary

        # Verify specific operations were tracked
        operations = summary["operations"]
        assert any("chain_execution_" in op for op in operations.keys())
        assert "text_generation" in operations
        assert "validation" in operations

        # Verify operation details
        for op_name, metrics in operations.items():
            assert "avg_time" in metrics
            assert "total_time" in metrics
            assert "call_count" in metrics
            assert metrics["call_count"] >= 1
            assert metrics["total_time"] >= 0

    def test_performance_bottleneck_detection(self):
        """Test that performance bottlenecks are correctly identified."""
        # Create a chain
        model = create_model("mock:test")
        chain = Chain(model=model, prompt="Test prompt")

        # Clear performance data
        chain.clear_performance_data()

        # Run the chain
        chain.run()

        # Get bottlenecks
        bottlenecks = chain.get_performance_bottlenecks()

        # For mock operations, bottlenecks should be empty (all operations < 100ms)
        assert isinstance(bottlenecks, list)
        # Mock operations are very fast, so no bottlenecks expected
        assert len(bottlenecks) == 0

    def test_performance_data_clearing(self):
        """Test that performance data can be cleared."""
        # Create and run a chain
        model = create_model("mock:test")
        chain = Chain(model=model, prompt="Test prompt")

        # Run the chain to generate performance data
        chain.run()

        # Verify data exists
        summary = chain.get_performance_summary()
        assert summary["total_operations"] > 0

        # Clear the data
        chain.clear_performance_data()

        # Verify data is cleared
        summary = chain.get_performance_summary()
        assert summary["total_operations"] == 0

    def test_performance_with_critics(self):
        """Test performance monitoring with critics enabled."""
        # Create a chain with critics
        model = create_model("mock:test")
        retriever = MockRetriever()
        validator = LengthValidator(min_length=10, max_length=500)
        critic = ReflexionCritic(model_name="mock:critic")

        chain = Chain(
            model=model,
            prompt="Test prompt that will fail validation",
            model_retrievers=[retriever],
        )
        chain.validate_with(validator)
        chain.improve_with(critic)

        # Enable critics to always run
        chain.with_options(always_apply_critics=True)

        # Clear performance data
        chain.clear_performance_data()

        # Run the chain
        thought = chain.run()

        # Get performance summary
        summary = chain.get_performance_summary()

        # Verify critic operations were tracked
        operations = summary["operations"]
        assert "critic_feedback" in operations
        assert any("critic_" in op for op in operations.keys())

    def test_performance_with_multiple_validators(self):
        """Test performance monitoring with multiple validators."""
        # Create a chain with multiple validators
        model = create_model("mock:test")
        validator1 = LengthValidator(min_length=10, max_length=500)
        validator2 = LengthValidator(min_length=5, max_length=1000)

        chain = Chain(model=model, prompt="Test prompt")
        chain.validate_with(validator1)
        chain.validate_with(validator2)

        # Clear performance data
        chain.clear_performance_data()

        # Run the chain
        thought = chain.run()

        # Get performance summary
        summary = chain.get_performance_summary()

        # Verify individual validator operations were tracked
        operations = summary["operations"]
        assert "validation_LengthValidator" in operations

        # Since both validators have the same class name, they share the same performance key
        # But the call count should reflect multiple calls
        length_validator_metrics = operations["validation_LengthValidator"]
        assert length_validator_metrics["call_count"] == 2  # Two validators of same type

    def test_performance_summary_structure(self):
        """Test that performance summary has the expected structure."""
        # Create and run a chain
        model = create_model("mock:test")
        chain = Chain(model=model, prompt="Test prompt")

        chain.clear_performance_data()
        chain.run()

        summary = chain.get_performance_summary()

        # Verify top-level structure
        assert "total_operations" in summary
        assert "total_time" in summary
        assert "avg_time_per_operation" in summary
        assert "slowest_operations" in summary
        assert "most_frequent_operations" in summary
        assert "unique_operations" in summary
        assert "operations" in summary

        # Verify operations structure
        operations = summary["operations"]
        for op_name, metrics in operations.items():
            assert "name" in metrics
            assert "total_time" in metrics
            assert "call_count" in metrics
            assert "avg_time" in metrics
            assert "min_time" in metrics
            assert "max_time" in metrics
            assert "recent_avg_time" in metrics

    def test_performance_monitor_singleton(self):
        """Test that PerformanceMonitor is a proper singleton."""
        monitor1 = PerformanceMonitor.get_instance()
        monitor2 = PerformanceMonitor.get_instance()

        # Should be the same instance
        assert monitor1 is monitor2

        # Should maintain state across instances
        monitor1.record_timing("test_operation", 0.1)
        stats = monitor2.get_stats()
        assert "test_operation" in stats

    def test_performance_thread_safety(self):
        """Test that performance monitoring is thread-safe."""
        import threading
        import time

        monitor = PerformanceMonitor.get_instance()
        monitor.clear()

        def record_timings():
            for i in range(10):
                monitor.record_timing("thread_test", 0.001)
                time.sleep(0.001)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=record_timings)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all recordings were captured
        stats = monitor.get_stats()
        assert "thread_test" in stats
        assert stats["thread_test"]["call_count"] == 30  # 3 threads * 10 recordings each


if __name__ == "__main__":
    pytest.main([__file__])
