"""Pytest configuration for Sifaka tests."""

import asyncio
import os
from unittest.mock import patch

import pytest

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_api_keys():
    """Mock API keys for all tests."""
    test_keys = {
        "GEMINI_API_KEY": "test-gemini-key",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }
    with patch.dict(os.environ, test_keys):
        yield


@pytest.fixture
def mock_sync_validator():
    """Create a validator that works in sync context."""
    from sifaka.core.thought import SifakaThought
    from sifaka.validators.base import BaseValidator, ValidationResult

    class SyncTestValidator(BaseValidator):
        def __init__(self, name="sync_test"):
            super().__init__(name, "Sync test validator")

        async def validate_async(self, thought: SifakaThought) -> ValidationResult:
            return ValidationResult(
                passed=True,
                message="Sync test passed",
                validator_name=self.name,
                processing_time_ms=1.0,
            )

    return SyncTestValidator()
