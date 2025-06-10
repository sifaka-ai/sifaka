"""Test environment setup for Sifaka tests.

This module provides environment configuration and mocking
for tests that require external API keys or services.
"""

import os
from unittest.mock import Mock, patch

import pytest

# Mock API keys for testing
TEST_API_KEYS = {
    "GEMINI_API_KEY": "test-gemini-key-12345",
    "OPENAI_API_KEY": "test-openai-key-12345",
    "ANTHROPIC_API_KEY": "test-anthropic-key-12345",
}


@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Automatically mock environment variables for all tests."""
    with patch.dict(os.environ, TEST_API_KEYS):
        yield


@pytest.fixture
def mock_pydantic_agent():
    """Mock PydanticAI agent for testing."""
    mock_agent = Mock()
    mock_agent.model = "test:mock-model"

    # Mock run_async method
    async def mock_run_async(*args, **kwargs):
        result = Mock()
        result.data = "Mock generated text"
        result.all_messages = [
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Mock generated text"},
        ]
        result.cost = 0.001
        result.usage = {"tokens": 100}
        return result

    mock_agent.run_async = mock_run_async
    return mock_agent


@pytest.fixture
def mock_dependencies():
    """Create mock SifakaDependencies for testing."""
    from unittest.mock import Mock

    from sifaka.graph.dependencies import SifakaDependencies

    mock_deps = Mock(spec=SifakaDependencies)
    mock_deps.generator_agent = Mock()
    mock_deps.generator_agent.model = "test:mock-model"
    mock_deps.validators = []
    mock_deps.critics = {}
    mock_deps.retrievers = {}

    return mock_deps
