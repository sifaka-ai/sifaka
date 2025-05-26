#!/usr/bin/env python3
"""Performance benchmarks and regression tests for Sifaka.

This test suite measures performance characteristics of Sifaka components
and detects performance regressions. It includes benchmarks for execution
time, memory usage, and throughput.
"""

import pytest
import time
import psutil
import os
import asyncio
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.content import ContentValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.utils.logging import get_logger

from tests.utils import (
    create_test_chain,
    assert_thought_valid,
    assert_performance_within_bounds,
    MockModelFactory,
    create_performance_scenario
)

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    operation_count: int
    throughput: float  # operations per second


class PerformanceMeasurer:
    """Utility class for measuring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
    def measure_operation(self, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of a single operation."""
        # Record initial state
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()
        
        # Execute operation
        result = operation_func(*args, **kwargs)
        
        # Record final state
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - self.initial_memory
        peak_memory = max(start_memory, end_memory)
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            cpu_percent=(start_cpu + end_cpu) / 2,
            operation_count=1,
            throughput=1.0 / execution_time if execution_time > 0 else 0
        )
    
    def measure_batch_operations(self, operation_func, batch_size: int, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of batch operations."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        results = []
        for i in range(batch_size):
            result = operation_func(*args, **kwargs)
            results.append(result)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_usage = end_memory - self.initial_memory
        throughput = batch_size / execution_time if execution_time > 0 else 0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=end_memory,
            cpu_percent=self.process.cpu_percent(),
            operation_count=batch_size,
            throughput=throughput
        )


class TestBasicPerformance:
    """Test basic performance characteristics."""
    
    def setup_method(self):
        """Set up performance measurement."""
        self.measurer = PerformanceMeasurer()
    
    def test_simple_chain_performance(self):
        """Test performance of simple chain execution."""
        def run_simple_chain():
            model = MockModel(model_name="perf-test")
            chain = Chain(model=model, prompt="Write a simple sentence.")
            return chain.run()
        
        metrics = self.measurer.measure_operation(run_simple_chain)
        
        # Performance expectations for simple chain
        assert metrics.execution_time < 1.0, f"Simple chain too slow: {metrics.execution_time:.3f}s"
        assert metrics.memory_usage_mb < 50, f"Simple chain uses too much memory: {metrics.memory_usage_mb:.1f}MB"
        
        logger.info(f"Simple chain performance: {metrics.execution_time:.3f}s, {metrics.memory_usage_mb:.1f}MB")
    
    def test_chain_with_validators_performance(self):
        """Test performance of chain with multiple validators."""
        def run_validated_chain():
            model = MockModel(model_name="perf-test")
            chain = Chain(model=model, prompt="Write about performance testing.")
            
            # Add multiple validators
            chain.validate_with(LengthValidator(min_length=10, max_length=500))
            chain.validate_with(RegexValidator(required_patterns=[r"performance"]))
            chain.validate_with(ContentValidator(prohibited=["slow"], name="Speed Filter"))
            
            return chain.run()
        
        metrics = self.measurer.measure_operation(run_validated_chain)
        
        # Performance expectations with validation
        assert metrics.execution_time < 2.0, f"Validated chain too slow: {metrics.execution_time:.3f}s"
        assert metrics.memory_usage_mb < 75, f"Validated chain uses too much memory: {metrics.memory_usage_mb:.1f}MB"
        
        logger.info(f"Validated chain performance: {metrics.execution_time:.3f}s, {metrics.memory_usage_mb:.1f}MB")
    
    def test_chain_with_critics_performance(self):
        """Test performance of chain with critics."""
        def run_criticized_chain():
            model = MockModel(model_name="perf-test")
            critic_model = MockModel(model_name="critic-perf-test")
            
            chain = Chain(
                model=model,
                prompt="Write about AI performance optimization.",
                always_apply_critics=True
            )
            
            # Add critics
            chain.improve_with(ReflexionCritic(model=critic_model))
            chain.improve_with(SelfRefineCritic(model=critic_model))
            
            return chain.run()
        
        metrics = self.measurer.measure_operation(run_criticized_chain)
        
        # Performance expectations with critics
        assert metrics.execution_time < 5.0, f"Criticized chain too slow: {metrics.execution_time:.3f}s"
        assert metrics.memory_usage_mb < 100, f"Criticized chain uses too much memory: {metrics.memory_usage_mb:.1f}MB"
        
        logger.info(f"Criticized chain performance: {metrics.execution_time:.3f}s, {metrics.memory_usage_mb:.1f}MB")


class TestThroughputPerformance:
    """Test throughput and batch processing performance."""
    
    def setup_method(self):
        """Set up performance measurement."""
        self.measurer = PerformanceMeasurer()
    
    def test_sequential_chain_throughput(self):
        """Test throughput of sequential chain execution."""
        def run_single_chain():
            model = MockModel(model_name="throughput-test")
            chain = Chain(model=model, prompt="Write a brief summary.")
            chain.validate_with(LengthValidator(min_length=10, max_length=100))
            return chain.run()
        
        batch_size = 10
        metrics = self.measurer.measure_batch_operations(run_single_chain, batch_size)
        
        # Throughput expectations
        assert metrics.throughput > 2.0, f"Sequential throughput too low: {metrics.throughput:.2f} ops/sec"
        assert metrics.execution_time < 10.0, f"Batch execution too slow: {metrics.execution_time:.3f}s"
        
        logger.info(f"Sequential throughput: {metrics.throughput:.2f} ops/sec ({batch_size} operations in {metrics.execution_time:.3f}s)")
    
    def test_concurrent_chain_performance(self):
        """Test performance benefits of concurrent execution."""
        def run_concurrent_chains(num_chains: int = 5):
            async def run_single_chain_async():
                model = MockModel(model_name="concurrent-test")
                chain = Chain(model=model, prompt="Write about concurrency.")
                chain.validate_with(LengthValidator(min_length=10, max_length=100))
                return await chain._run_async()
            
            async def run_all_chains():
                tasks = [run_single_chain_async() for _ in range(num_chains)]
                return await asyncio.gather(*tasks)
            
            return asyncio.run(run_all_chains())
        
        start_time = time.time()
        results = run_concurrent_chains(5)
        concurrent_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 5
        for result in results:
            assert_thought_valid(result)
        
        # Performance should be better than sequential
        assert concurrent_time < 5.0, f"Concurrent execution too slow: {concurrent_time:.3f}s"
        
        concurrent_throughput = len(results) / concurrent_time
        logger.info(f"Concurrent throughput: {concurrent_throughput:.2f} ops/sec ({len(results)} operations in {concurrent_time:.3f}s)")
    
    def test_memory_efficiency_batch_processing(self):
        """Test memory efficiency during batch processing."""
        initial_memory = self.measurer.process.memory_info().rss / 1024 / 1024
        
        # Process multiple chains
        for i in range(20):
            model = MockModel(model_name=f"memory-test-{i}")
            chain = Chain(model=model, prompt=f"Write about memory test {i}.")
            result = chain.run()
            assert_thought_valid(result)
        
        final_memory = self.measurer.process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB for 20 chains"
        
        logger.info(f"Memory efficiency: {memory_increase:.1f}MB increase for 20 chains")


class TestScalabilityPerformance:
    """Test performance scalability with increasing load."""
    
    def setup_method(self):
        """Set up performance measurement."""
        self.measurer = PerformanceMeasurer()
    
    def test_validator_scalability(self):
        """Test performance scaling with number of validators."""
        validator_counts = [1, 3, 5, 10]
        execution_times = []
        
        for count in validator_counts:
            def run_chain_with_validators():
                model = MockModel(model_name="scalability-test")
                chain = Chain(model=model, prompt="Write about scalability testing.")
                
                # Add specified number of validators
                for i in range(count):
                    chain.validate_with(LengthValidator(min_length=10, max_length=500))
                
                return chain.run()
            
            metrics = self.measurer.measure_operation(run_chain_with_validators)
            execution_times.append(metrics.execution_time)
            
            # Each individual test should complete reasonably quickly
            assert metrics.execution_time < 5.0, f"Chain with {count} validators too slow: {metrics.execution_time:.3f}s"
        
        # Performance should scale reasonably (not exponentially)
        # Allow for some increase but not dramatic
        max_time = max(execution_times)
        min_time = min(execution_times)
        scaling_factor = max_time / min_time if min_time > 0 else 1
        
        assert scaling_factor < 5.0, f"Validator scaling too poor: {scaling_factor:.2f}x increase from {validator_counts[0]} to {validator_counts[-1]} validators"
        
        logger.info(f"Validator scalability: {execution_times} (scaling factor: {scaling_factor:.2f}x)")
    
    def test_critic_scalability(self):
        """Test performance scaling with number of critics."""
        critic_counts = [1, 2, 3]
        execution_times = []
        
        for count in critic_counts:
            def run_chain_with_critics():
                model = MockModel(model_name="critic-scalability-test")
                critic_model = MockModel(model_name="critic-model")
                
                chain = Chain(
                    model=model,
                    prompt="Write about critic scalability.",
                    always_apply_critics=True
                )
                
                # Add specified number of critics
                for i in range(count):
                    if i % 2 == 0:
                        chain.improve_with(ReflexionCritic(model=critic_model))
                    else:
                        chain.improve_with(SelfRefineCritic(model=critic_model))
                
                return chain.run()
            
            metrics = self.measurer.measure_operation(run_chain_with_critics)
            execution_times.append(metrics.execution_time)
            
            # Each test should complete reasonably quickly
            assert metrics.execution_time < 10.0, f"Chain with {count} critics too slow: {metrics.execution_time:.3f}s"
        
        logger.info(f"Critic scalability: {execution_times}")
    
    def test_storage_scalability(self):
        """Test storage performance with increasing data volume."""
        storage = MemoryStorage()
        save_times = []
        load_times = []
        
        data_sizes = [10, 50, 100, 200]
        
        for size in data_sizes:
            thoughts = []
            
            # Create thoughts
            for i in range(size):
                thought = Thought(
                    prompt=f"Scalability test {i}",
                    text=f"Test data for scalability measurement {i}" * 10,  # Make text longer
                    iteration=i
                )
                thoughts.append(thought)
            
            # Measure save performance
            start_time = time.time()
            for thought in thoughts:
                storage.save(thought.id, thought)
            save_time = time.time() - start_time
            save_times.append(save_time)
            
            # Measure load performance
            start_time = time.time()
            for thought in thoughts:
                loaded = storage.load(thought.id)
                assert loaded.id == thought.id
            load_time = time.time() - start_time
            load_times.append(load_time)
            
            # Individual performance should be reasonable
            assert save_time < 5.0, f"Save time too slow for {size} thoughts: {save_time:.3f}s"
            assert load_time < 5.0, f"Load time too slow for {size} thoughts: {load_time:.3f}s"
        
        logger.info(f"Storage scalability - Save times: {save_times}, Load times: {load_times}")


class TestRegressionBenchmarks:
    """Benchmark tests to detect performance regressions."""
    
    def setup_method(self):
        """Set up performance measurement."""
        self.measurer = PerformanceMeasurer()
    
    def test_baseline_chain_benchmark(self):
        """Baseline benchmark for simple chain execution."""
        def baseline_operation():
            model = MockModel(model_name="baseline-test")
            chain = Chain(model=model, prompt="Write a baseline test sentence.")
            return chain.run()
        
        # Run multiple times for average
        times = []
        for _ in range(5):
            metrics = self.measurer.measure_operation(baseline_operation)
            times.append(metrics.execution_time)
        
        avg_time = sum(times) / len(times)
        
        # Baseline performance expectation
        assert avg_time < 0.5, f"Baseline performance regression: {avg_time:.3f}s average"
        
        logger.info(f"Baseline benchmark: {avg_time:.3f}s average ({min(times):.3f}s - {max(times):.3f}s)")
    
    def test_complex_chain_benchmark(self):
        """Benchmark for complex chain with all components."""
        def complex_operation():
            model = MockModel(model_name="complex-benchmark")
            critic_model = MockModel(model_name="complex-critic")
            storage = MemoryStorage()
            
            chain = Chain(
                model=model,
                prompt="Write a comprehensive analysis of AI performance optimization strategies.",
                storage=storage,
                always_apply_critics=True
            )
            
            # Add multiple validators
            chain.validate_with(LengthValidator(min_length=50, max_length=1000))
            chain.validate_with(RegexValidator(required_patterns=[r"AI", r"performance"]))
            chain.validate_with(ContentValidator(prohibited=["slow", "bad"], name="Quality Filter"))
            
            # Add multiple critics
            chain.improve_with(ReflexionCritic(model=critic_model))
            chain.improve_with(SelfRefineCritic(model=critic_model))
            
            return chain.run()
        
        metrics = self.measurer.measure_operation(complex_operation)
        
        # Complex operation performance expectation
        assert metrics.execution_time < 8.0, f"Complex chain performance regression: {metrics.execution_time:.3f}s"
        assert metrics.memory_usage_mb < 150, f"Complex chain memory regression: {metrics.memory_usage_mb:.1f}MB"
        
        logger.info(f"Complex benchmark: {metrics.execution_time:.3f}s, {metrics.memory_usage_mb:.1f}MB")
    
    def test_concurrent_benchmark(self):
        """Benchmark for concurrent operations."""
        async def concurrent_operation():
            tasks = []
            for i in range(10):
                model = MockModel(model_name=f"concurrent-benchmark-{i}")
                chain = Chain(model=model, prompt=f"Write about concurrent test {i}.")
                chain.validate_with(LengthValidator(min_length=10, max_length=200))
                tasks.append(chain._run_async())
            
            return await asyncio.gather(*tasks)
        
        start_time = time.time()
        results = asyncio.run(concurrent_operation())
        execution_time = time.time() - start_time
        
        # Concurrent performance expectation
        assert execution_time < 5.0, f"Concurrent performance regression: {execution_time:.3f}s for 10 operations"
        assert len(results) == 10, "All concurrent operations should complete"
        
        for result in results:
            assert_thought_valid(result)
        
        throughput = len(results) / execution_time
        logger.info(f"Concurrent benchmark: {execution_time:.3f}s, {throughput:.2f} ops/sec")
