"""
Performance benchmarks for Sifaka.

This module provides tests for:
1. Response time across different models and adapters
2. Memory consumption with large inputs
3. Throughput under high request volume
"""

import pytest
import time
import gc
import asyncio
import statistics
import os
import random
import string
import psutil
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from sifaka.models.base import ModelProvider, ModelConfig
from sifaka.rules.base import Rule, RuleResult
from sifaka.critics.base import CriticOutput, CriticResult, CriticMetadata
from sifaka.adapters.rules.base import BaseAdapter
from sifaka.adapters.rules.classifier import ClassifierAdapter, create_classifier_rule
from sifaka.classifiers.base import ClassificationResult


# ---------------------------------------------------------------------------
# Utility functions for benchmarking
# ---------------------------------------------------------------------------

def measure_time(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Measure execution time of a function.

    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dictionary with result and timing information
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time

    return {
        "result": result,
        "elapsed": elapsed
    }


def measure_memory(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Measure memory usage of a function.

    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dictionary with result, timing, and memory information
    """
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    return {
        "result": result,
        "elapsed": elapsed,
        "memory_diff_mb": mem_after - mem_before
    }


def generate_text(length: int) -> str:
    """Generate random text of specified length.

    Args:
        length: Length of text to generate

    Returns:
        Randomly generated text
    """
    chars = string.ascii_letters + string.digits + ' ' * 10 + '.,!?;:' + '\n'
    return ''.join(random.choice(chars) for _ in range(length))


async def async_measure_time(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Measure execution time of an async function.

    Args:
        func: Async function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Dictionary with result and timing information
    """
    start_time = time.time()
    result = await func(*args, **kwargs)
    elapsed = time.time() - start_time

    return {
        "result": result,
        "elapsed": elapsed
    }


# ---------------------------------------------------------------------------
# Mock classes for testing
# ---------------------------------------------------------------------------

class MockModelProvider(ModelProvider):
    """Mock model provider with configurable response time."""

    def __init__(self, latency: float = 0.01, **kwargs):
        """Initialize mock provider with simulated latency.

        Args:
            latency: Simulated response time in seconds
            **kwargs: Additional configuration parameters
        """
        config = kwargs.get("config", {
            "name": "mock_provider",
            "description": "Mock provider for benchmarking",
            "params": {
                "latency": latency
            }
        })
        super().__init__(config)
        self.latency = latency
        self._token_rate = kwargs.get("token_rate", 20)  # tokens per character

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response with simulated latency.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response dictionary
        """
        # Simulate processing time based on input length and configured latency
        time.sleep(self.latency * (1 + len(prompt) / 1000))

        # Generate mock response proportional to prompt length
        response_length = min(len(prompt) * 2, 2000)
        response_text = generate_text(response_length)

        return {
            "text": response_text,
            "model": "mock-model",
            "provider": "mock-provider",
            "usage": {
                "prompt_tokens": len(prompt) // self._token_rate,
                "completion_tokens": len(response_text) // self._token_rate,
                "total_tokens": (len(prompt) + len(response_text)) // self._token_rate
            }
        }


class MockClassifier:
    """Mock classifier with configurable response time."""

    def __init__(self, latency: float = 0.005):
        """Initialize mock classifier with simulated latency.

        Args:
            latency: Simulated classification time in seconds
        """
        self.latency = latency
        self._labels = ["positive", "negative", "neutral"]

    @property
    def name(self) -> str:
        return "mock_classifier"

    @property
    def description(self) -> str:
        return "Mock classifier for benchmarking"

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "labels": self._labels,
            "latency": self.latency
        }

    def classify(self, text: str) -> ClassificationResult:
        """Classify text with simulated latency.

        Args:
            text: Text to classify

        Returns:
            Classification result
        """
        # Simulate processing time based on text length and configured latency
        time.sleep(self.latency * (1 + len(text) / 10000))

        # Return mock classification
        label = random.choice(self._labels)
        confidence = random.uniform(0.7, 0.95)

        return ClassificationResult(
            label=label,
            confidence=confidence,
            metadata={"text_length": len(text)}
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Batch classify texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]


class MockRule(Rule):
    """Mock rule with configurable validation time."""

    def __init__(self, name: str, description: str, latency: float = 0.001):
        """Initialize mock rule with simulated latency.

        Args:
            name: Rule name
            description: Rule description
            latency: Simulated validation time in seconds
        """
        super().__init__(name=name, description=description)
        self.latency = latency

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text with simulated latency.

        Args:
            text: Text to validate
            **kwargs: Additional validation parameters

        Returns:
            Validation result
        """
        # Simulate processing time based on text length and configured latency
        time.sleep(self.latency * (1 + len(text) / 20000))

        # Return mock validation result
        passed = random.random() > 0.2  # 80% pass rate

        return RuleResult(
            passed=passed,
            message="Mock validation " + ("passed" if passed else "failed"),
            metadata={"text_length": len(text)}
        )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_models():
    """Create a set of mock model providers with different latencies."""
    return {
        "fast": MockModelProvider(latency=0.01),
        "medium": MockModelProvider(latency=0.05),
        "slow": MockModelProvider(latency=0.1)
    }


@pytest.fixture
def mock_classifiers():
    """Create a set of mock classifiers with different latencies."""
    return {
        "fast": MockClassifier(latency=0.002),
        "medium": MockClassifier(latency=0.005),
        "slow": MockClassifier(latency=0.01)
    }


@pytest.fixture
def mock_rules():
    """Create a set of mock rules with different latencies."""
    return {
        "fast": MockRule(name="fast_rule", description="Fast rule", latency=0.0005),
        "medium": MockRule(name="medium_rule", description="Medium rule", latency=0.001),
        "slow": MockRule(name="slow_rule", description="Slow rule", latency=0.002)
    }


@pytest.fixture
def mock_adapters(mock_classifiers):
    """Create a set of adapters wrapping mock classifiers."""
    adapters = {}

    for speed, classifier in mock_classifiers.items():
        adapters[speed] = ClassifierAdapter(
            adaptee=classifier,
            valid_labels=["positive", "neutral"],
            threshold=0.7
        )

    return adapters


@pytest.fixture
def text_samples():
    """Generate test samples of different sizes."""
    return {
        "small": [generate_text(100) for _ in range(20)],
        "medium": [generate_text(1000) for _ in range(10)],
        "large": [generate_text(10000) for _ in range(5)],
        "very_large": [generate_text(100000) for _ in range(2)]
    }


# ---------------------------------------------------------------------------
# Response time benchmark tests
# ---------------------------------------------------------------------------

class TestResponseTime:
    """Benchmarks for response time across different components."""

    def test_model_response_time(self, mock_models, text_samples):
        """Benchmark response time for different model providers."""
        results = {}

        # Test each model with each text size
        for model_speed, model in mock_models.items():
            model_results = {}

            for size, samples in text_samples.items():
                # Skip very large samples for slow models to avoid long test times
                if size == "very_large" and model_speed == "slow":
                    continue

                latencies = []

                for text in samples:
                    stats = measure_time(model.generate, text)
                    latencies.append(stats["elapsed"])

                model_results[size] = {
                    "mean_latency": statistics.mean(latencies),
                    "p95_latency": np.percentile(latencies, 95),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }

            results[model_speed] = model_results

        # Print results for visibility in test output
        print("\nModel Response Time Benchmark Results:")
        for model_speed, model_results in results.items():
            print(f"\n  {model_speed.upper()} Model:")
            for size, metrics in model_results.items():
                print(f"    {size} texts: {metrics['mean_latency']:.4f}s mean, "
                      f"{metrics['p95_latency']:.4f}s p95")

        # Simple assertions to validate benchmarks ran correctly
        assert "fast" in results
        assert "medium" in results

        # Verify that faster models have lower response times
        for size in ["small", "medium", "large"]:
            assert results["fast"][size]["mean_latency"] < results["slow"][size]["mean_latency"]

    def test_classifier_response_time(self, mock_classifiers, text_samples):
        """Benchmark response time for different classifiers."""
        results = {}

        # Test each classifier with each text size
        for classifier_speed, classifier in mock_classifiers.items():
            classifier_results = {}

            for size, samples in text_samples.items():
                # Skip very large samples for slow classifiers
                if size == "very_large" and classifier_speed == "slow":
                    continue

                latencies = []

                for text in samples:
                    stats = measure_time(classifier.classify, text)
                    latencies.append(stats["elapsed"])

                classifier_results[size] = {
                    "mean_latency": statistics.mean(latencies),
                    "p95_latency": np.percentile(latencies, 95),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }

            results[classifier_speed] = classifier_results

        # Print results for visibility in test output
        print("\nClassifier Response Time Benchmark Results:")
        for classifier_speed, classifier_results in results.items():
            print(f"\n  {classifier_speed.upper()} Classifier:")
            for size, metrics in classifier_results.items():
                print(f"    {size} texts: {metrics['mean_latency']:.4f}s mean, "
                      f"{metrics['p95_latency']:.4f}s p95")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify that faster classifiers have lower response times
        for size in ["small", "medium", "large"]:
            assert results["fast"][size]["mean_latency"] < results["medium"][size]["mean_latency"]

    def test_adapter_response_time(self, mock_adapters, text_samples):
        """Benchmark response time for different adapters."""
        results = {}

        # Test each adapter with each text size
        for adapter_speed, adapter in mock_adapters.items():
            adapter_results = {}

            for size, samples in text_samples.items():
                # Skip very large samples for slow adapters
                if size == "very_large" and adapter_speed == "slow":
                    continue

                latencies = []

                for text in samples:
                    stats = measure_time(adapter.validate, text)
                    latencies.append(stats["elapsed"])

                adapter_results[size] = {
                    "mean_latency": statistics.mean(latencies),
                    "p95_latency": np.percentile(latencies, 95),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }

            results[adapter_speed] = adapter_results

        # Print results for visibility in test output
        print("\nAdapter Response Time Benchmark Results:")
        for adapter_speed, adapter_results in results.items():
            print(f"\n  {adapter_speed.upper()} Adapter:")
            for size, metrics in adapter_results.items():
                print(f"    {size} texts: {metrics['mean_latency']:.4f}s mean, "
                      f"{metrics['p95_latency']:.4f}s p95")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify that faster adapters have lower response times
        for size in ["small", "medium", "large"]:
            assert results["fast"][size]["mean_latency"] < results["slow"][size]["mean_latency"]


# ---------------------------------------------------------------------------
# Memory consumption benchmark tests
# ---------------------------------------------------------------------------

class TestMemoryConsumption:
    """Benchmarks for memory consumption with large inputs."""

    def test_model_memory_usage(self, mock_models):
        """Benchmark memory usage for model providers with increasing input sizes."""
        results = {}

        # Test with increasing text sizes
        sizes = [100, 1000, 10000, 100000]

        for model_speed, model in mock_models.items():
            size_results = {}

            for size in sizes:
                # Generate text of specific size
                text = generate_text(size)

                # Measure memory usage during model generation
                stats = measure_memory(model.generate, text)

                size_results[size] = {
                    "memory_diff_mb": stats["memory_diff_mb"],
                    "elapsed": stats["elapsed"]
                }

            results[model_speed] = size_results

        # Print results for visibility in test output
        print("\nModel Memory Usage Benchmark Results:")
        for model_speed, size_results in results.items():
            print(f"\n  {model_speed.upper()} Model:")
            for size, metrics in size_results.items():
                print(f"    {size} chars: {metrics['memory_diff_mb']:.2f}MB")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify that memory usage increases with input size
        for model_speed in results:
            assert results[model_speed][100]["memory_diff_mb"] < results[model_speed][10000]["memory_diff_mb"]

    def test_classifier_memory_usage(self, mock_classifiers):
        """Benchmark memory usage for classifiers with increasing input sizes."""
        results = {}

        # Test with increasing text sizes
        sizes = [100, 1000, 10000, 100000]

        for classifier_speed, classifier in mock_classifiers.items():
            size_results = {}

            for size in sizes:
                # Generate text of specific size
                text = generate_text(size)

                # Measure memory usage during classification
                stats = measure_memory(classifier.classify, text)

                size_results[size] = {
                    "memory_diff_mb": stats["memory_diff_mb"],
                    "elapsed": stats["elapsed"]
                }

            results[classifier_speed] = size_results

        # Print results for visibility in test output
        print("\nClassifier Memory Usage Benchmark Results:")
        for classifier_speed, size_results in results.items():
            print(f"\n  {classifier_speed.upper()} Classifier:")
            for size, metrics in size_results.items():
                print(f"    {size} chars: {metrics['memory_diff_mb']:.2f}MB")

        # Assertions
        assert "fast" in results
        assert "medium" in results

    def test_adapter_memory_usage(self, mock_adapters):
        """Benchmark memory usage for adapters with increasing input sizes."""
        results = {}

        # Test with increasing text sizes
        sizes = [100, 1000, 10000, 100000]

        for adapter_speed, adapter in mock_adapters.items():
            size_results = {}

            for size in sizes:
                # Generate text of specific size
                text = generate_text(size)

                # Measure memory usage during validation
                stats = measure_memory(adapter.validate, text)

                size_results[size] = {
                    "memory_diff_mb": stats["memory_diff_mb"],
                    "elapsed": stats["elapsed"]
                }

            results[adapter_speed] = size_results

        # Print results for visibility in test output
        print("\nAdapter Memory Usage Benchmark Results:")
        for adapter_speed, size_results in results.items():
            print(f"\n  {adapter_speed.upper()} Adapter:")
            for size, metrics in size_results.items():
                print(f"    {size} chars: {metrics['memory_diff_mb']:.2f}MB")

        # Assertions
        assert "fast" in results
        assert "medium" in results

    def test_batched_operation_memory_usage(self, mock_classifiers):
        """Compare memory usage between batched and individual operations."""
        results = {}

        # Test with batch sizes from small to large
        batch_sizes = [1, 10, 50, 100]
        text_length = 1000  # Fixed text length

        for classifier_speed, classifier in mock_classifiers.items():
            batch_results = {}

            for batch_size in batch_sizes:
                # Generate batch of texts
                texts = [generate_text(text_length) for _ in range(batch_size)]

                # Measure individual operations
                start_time = time.time()
                individual_memory = measure_memory(
                    lambda: [classifier.classify(text) for text in texts]
                )
                individual_time = time.time() - start_time

                # Measure batch operation
                batch_memory = measure_memory(
                    classifier.batch_classify, texts
                )

                batch_results[batch_size] = {
                    "individual_memory_mb": individual_memory["memory_diff_mb"],
                    "batch_memory_mb": batch_memory["memory_diff_mb"],
                    "memory_ratio": batch_memory["memory_diff_mb"] / individual_memory["memory_diff_mb"] if individual_memory["memory_diff_mb"] > 0 else 1.0,
                    "individual_time": individual_time,
                    "batch_time": batch_memory["elapsed"],
                    "time_ratio": batch_memory["elapsed"] / individual_time if individual_time > 0 else 1.0
                }

            results[classifier_speed] = batch_results

        # Print results for visibility in test output
        print("\nBatched vs. Individual Operation Memory Usage:")
        for classifier_speed, batch_results in results.items():
            print(f"\n  {classifier_speed.upper()} Classifier:")
            for batch_size, metrics in batch_results.items():
                print(f"    Batch size {batch_size}: "
                      f"Memory ratio: {metrics['memory_ratio']:.2f}x, "
                      f"Time ratio: {metrics['time_ratio']:.2f}x")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify that batch operations are generally more memory efficient
        for classifier_speed in results:
            assert results[classifier_speed][100]["time_ratio"] < 1.0  # Batch should be faster


# ---------------------------------------------------------------------------
# Throughput benchmark tests
# ---------------------------------------------------------------------------

class TestThroughput:
    """Benchmarks for throughput under high request volume."""

    def test_model_throughput(self, mock_models):
        """Benchmark throughput for model providers under concurrent load."""
        results = {}

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16]
        requests_per_level = 50
        text_length = 500

        for model_speed, model in mock_models.items():
            concurrency_results = {}

            for concurrency in concurrency_levels:
                # Generate texts for requests
                texts = [generate_text(text_length) for _ in range(requests_per_level)]

                # Function to process a single request
                def process_request(text):
                    return model.generate(text)

                # Measure time with ThreadPoolExecutor
                start_time = time.time()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    results_list = list(executor.map(process_request, texts))

                elapsed = time.time() - start_time

                # Calculate throughput metrics
                throughput = len(texts) / elapsed
                latency_per_request = elapsed / len(texts)

                concurrency_results[concurrency] = {
                    "throughput": throughput,
                    "latency_per_request": latency_per_request,
                    "elapsed": elapsed
                }

            results[model_speed] = concurrency_results

        # Print results for visibility in test output
        print("\nModel Throughput Benchmark Results:")
        for model_speed, concurrency_results in results.items():
            print(f"\n  {model_speed.upper()} Model:")
            for concurrency, metrics in concurrency_results.items():
                print(f"    Concurrency {concurrency}: "
                      f"{metrics['throughput']:.2f} requests/second, "
                      f"{metrics['latency_per_request']*1000:.2f}ms per request")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify that throughput generally increases with concurrency
        for model_speed in results:
            assert results[model_speed][1]["throughput"] < results[model_speed][8]["throughput"]

    def test_classifier_throughput(self, mock_classifiers):
        """Benchmark throughput for classifiers under concurrent load."""
        results = {}

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16, 32]
        requests_per_level = 100
        text_length = 500

        for classifier_speed, classifier in mock_classifiers.items():
            concurrency_results = {}

            for concurrency in concurrency_levels:
                # Generate texts for requests
                texts = [generate_text(text_length) for _ in range(requests_per_level)]

                # Function to process a single request
                def process_request(text):
                    return classifier.classify(text)

                # Measure time with ThreadPoolExecutor
                start_time = time.time()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    results_list = list(executor.map(process_request, texts))

                elapsed = time.time() - start_time

                # Calculate throughput metrics
                throughput = len(texts) / elapsed
                latency_per_request = elapsed / len(texts)

                concurrency_results[concurrency] = {
                    "throughput": throughput,
                    "latency_per_request": latency_per_request,
                    "elapsed": elapsed
                }

            results[classifier_speed] = concurrency_results

        # Print results for visibility in test output
        print("\nClassifier Throughput Benchmark Results:")
        for classifier_speed, concurrency_results in results.items():
            print(f"\n  {classifier_speed.upper()} Classifier:")
            for concurrency, metrics in concurrency_results.items():
                print(f"    Concurrency {concurrency}: "
                      f"{metrics['throughput']:.2f} requests/second, "
                      f"{metrics['latency_per_request']*1000:.2f}ms per request")

        # Assertions
        assert "fast" in results
        assert "medium" in results

        # Verify throughput scaling with concurrency
        for classifier_speed in results:
            assert results[classifier_speed][1]["throughput"] < results[classifier_speed][16]["throughput"]

    def test_adapter_throughput(self, mock_adapters):
        """Benchmark throughput for adapters under concurrent load."""
        results = {}

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16, 32]
        requests_per_level = 100
        text_length = 500

        for adapter_speed, adapter in mock_adapters.items():
            concurrency_results = {}

            for concurrency in concurrency_levels:
                # Generate texts for requests
                texts = [generate_text(text_length) for _ in range(requests_per_level)]

                # Function to process a single request
                def process_request(text):
                    return adapter.validate(text)

                # Measure time with ThreadPoolExecutor
                start_time = time.time()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    results_list = list(executor.map(process_request, texts))

                elapsed = time.time() - start_time

                # Calculate throughput metrics
                throughput = len(texts) / elapsed
                latency_per_request = elapsed / len(texts)

                concurrency_results[concurrency] = {
                    "throughput": throughput,
                    "latency_per_request": latency_per_request,
                    "elapsed": elapsed
                }

            results[adapter_speed] = concurrency_results

        # Print results for visibility in test output
        print("\nAdapter Throughput Benchmark Results:")
        for adapter_speed, concurrency_results in results.items():
            print(f"\n  {adapter_speed.upper()} Adapter:")
            for concurrency, metrics in concurrency_results.items():
                print(f"    Concurrency {concurrency}: "
                      f"{metrics['throughput']:.2f} requests/second, "
                      f"{metrics['latency_per_request']*1000:.2f}ms per request")

        # Assertions
        assert "fast" in results
        assert "medium" in results

    def test_end_to_end_pipeline_throughput(self, mock_models, mock_adapters):
        """Benchmark throughput for an end-to-end pipeline (model + adapter)."""
        results = {}

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        requests_per_level = 20
        text_length = 200

        # Create fast and slow pipelines
        pipelines = {
            "fast": (mock_models["fast"], mock_adapters["fast"]),
            "slow": (mock_models["slow"], mock_adapters["slow"])
        }

        for pipeline_speed, (model, adapter) in pipelines.items():
            concurrency_results = {}

            for concurrency in concurrency_levels:
                # Generate prompts for requests
                prompts = [generate_text(text_length) for _ in range(requests_per_level)]

                # Function to process a single request through the pipeline
                def process_pipeline(prompt):
                    # Generate text with model
                    response = model.generate(prompt)

                    # Validate with adapter
                    validation = adapter.validate(response["text"])

                    return {
                        "response": response,
                        "validation": validation
                    }

                # Measure time with ThreadPoolExecutor
                start_time = time.time()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    results_list = list(executor.map(process_pipeline, prompts))

                elapsed = time.time() - start_time

                # Calculate throughput metrics
                throughput = len(prompts) / elapsed
                latency_per_request = elapsed / len(prompts)

                concurrency_results[concurrency] = {
                    "throughput": throughput,
                    "latency_per_request": latency_per_request,
                    "elapsed": elapsed
                }

            results[pipeline_speed] = concurrency_results

        # Print results for visibility in test output
        print("\nEnd-to-End Pipeline Throughput Benchmark Results:")
        for pipeline_speed, concurrency_results in results.items():
            print(f"\n  {pipeline_speed.upper()} Pipeline:")
            for concurrency, metrics in concurrency_results.items():
                print(f"    Concurrency {concurrency}: "
                      f"{metrics['throughput']:.2f} pipelines/second, "
                      f"{metrics['latency_per_request']*1000:.2f}ms per pipeline")

        # Assertions
        assert "fast" in results
        assert "slow" in results

        # Verify that the fast pipeline has higher throughput
        for concurrency in concurrency_levels:
            assert results["fast"][concurrency]["throughput"] > results["slow"][concurrency]["throughput"]