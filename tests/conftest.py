#!/usr/bin/env python3
"""Pytest configuration and shared fixtures for Sifaka tests.

This file contains pytest configuration and fixtures that are shared across
all test modules. It provides common test utilities, mock objects, and
setup/teardown functionality.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock

import pytest

from sifaka.core.thought import Thought
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Basic fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_model() -> MockModel:
    """Create a mock model for testing."""
    return MockModel(
        model_name="test-model", response_text="This is a test response from the mock model."
    )


@pytest.fixture
def memory_storage() -> MemoryStorage:
    """Create a memory storage instance for testing."""
    return MemoryStorage()


@pytest.fixture
def sample_thought() -> Thought:
    """Create a sample thought for testing."""
    return Thought(
        prompt="Test prompt for sample thought",
        text="This is sample text content for testing purposes.",
        model_prompt="Test prompt for sample thought",
    )


@pytest.fixture
def sample_thoughts() -> list[Thought]:
    """Create multiple sample thoughts for testing."""
    return [
        Thought(
            prompt=f"Test prompt {i}",
            text=f"This is sample text content {i} for testing purposes.",
            model_prompt=f"Test prompt {i}",
        )
        for i in range(1, 6)
    ]


# Mock fixtures
@pytest.fixture
def mock_api_response() -> dict[str, Any]:
    """Create a mock API response for testing."""
    return {
        "choices": [{"message": {"content": "Mock API response content"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_http_client() -> Mock:
    """Create a mock HTTP client for testing."""
    mock_client = Mock()
    mock_client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "Mock response"}}]
    }
    mock_client.post.return_value.status_code = 200
    return mock_client


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Create a performance monitor for testing."""
    from sifaka.utils.performance import PerformanceMonitor

    monitor = PerformanceMonitor()
    monitor.clear()  # Start with clean state
    yield monitor
    monitor.clear()  # Clean up after test


# Test data fixtures
@pytest.fixture
def sample_text_data() -> dict[str, str]:
    """Provide sample text data for various test scenarios."""
    return {
        "short": "Short text.",
        "medium": "This is a medium length text that contains multiple sentences. It should be suitable for most validation and processing tests.",
        "long": "This is a very long text that contains many sentences and should be suitable for testing scenarios that require longer content. "
        * 10,
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "Text with special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "unicode": "Text with unicode: 你好世界 🌍 café naïve résumé",
        "json": '{"name": "test", "value": 123, "active": true}',
        "xml": "<root><item>test</item><value>123</value></root>",
        "markdown": "# Title\n\nThis is **bold** and *italic* text.\n\n- List item 1\n- List item 2",
        "code": "def hello_world():\n    print('Hello, world!')\n    return True",
    }


@pytest.fixture
def sample_prompts() -> dict[str, str]:
    """Provide sample prompts for testing."""
    return {
        "simple": "Write a short story.",
        "detailed": "Write a detailed technical explanation of how machine learning works.",
        "creative": "Write a creative story about a robot learning to paint.",
        "analytical": "Analyze the pros and cons of renewable energy sources.",
        "instructional": "Explain how to bake a chocolate cake step by step.",
        "conversational": "Have a conversation about the weather.",
        "system_prompt": "You are a helpful assistant. Please respond professionally.",
    }


# Error testing fixtures
@pytest.fixture
def failing_mock_model() -> MockModel:
    """Create a mock model that always fails for error testing."""
    model = MockModel(model_name="failing-model")
    model.generate = Mock(side_effect=Exception("Mock model failure"))
    return model


@pytest.fixture
def slow_mock_model() -> MockModel:
    """Create a mock model with artificial delay for performance testing."""
    return MockModel(model_name="slow-model", response_text="Slow response", delay_seconds=0.1)


# Validation fixtures
@pytest.fixture
def validation_test_cases() -> dict[str, dict[str, Any]]:
    """Provide test cases for validation testing."""
    return {
        "valid_length": {
            "text": "This text has appropriate length for testing.",
            "should_pass": True,
            "min_length": 10,
            "max_length": 100,
        },
        "too_short": {"text": "Short", "should_pass": False, "min_length": 10, "max_length": 100},
        "too_long": {
            "text": "This text is way too long for the validation requirements. " * 10,
            "should_pass": False,
            "min_length": 10,
            "max_length": 100,
        },
        "empty": {"text": "", "should_pass": False, "min_length": 1, "max_length": 100},
    }


# Classification fixtures
@pytest.fixture
def classification_test_cases() -> dict[str, dict[str, Any]]:
    """Provide test cases for classification testing."""
    return {
        "positive_sentiment": {
            "text": "I love this product! It's amazing and wonderful!",
            "expected_labels": ["positive", "good", "happy"],
        },
        "negative_sentiment": {
            "text": "This is terrible and awful. I hate it completely.",
            "expected_labels": ["negative", "bad", "angry"],
        },
        "neutral_sentiment": {
            "text": "The weather is cloudy today. Meeting at 3 PM.",
            "expected_labels": ["neutral", "factual", "objective"],
        },
        "clean_content": {
            "text": "This is clean and appropriate content for all audiences.",
            "expected_labels": ["clean", "safe", "appropriate"],
        },
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_performance_monitor():
    """Automatically clean up performance monitor after each test."""
    yield
    try:
        from sifaka.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor.get_instance()
        monitor.clear()
    except ImportError:
        pass  # Performance module might not be available in all tests


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to avoid state leakage."""
    yield
    # Reset any singleton instances here
    try:
        from sifaka.utils.performance import PerformanceMonitor

        PerformanceMonitor.reset()
    except ImportError:
        pass
