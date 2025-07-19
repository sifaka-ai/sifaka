"""Root test configuration."""

import os
import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Configure pytest."""
    # Add the project root to the Python path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # Set up test environment
    setup_test_environment()


def setup_test_environment():
    """Set up environment for testing."""
    # Check if we're in CI or running unit tests
    is_ci = os.getenv("CI", "false").lower() == "true"
    is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

    # For unit tests, always use mock keys unless real ones are provided
    if is_ci or is_github_actions:
        # In CI, set mock API keys if not already set
        os.environ.setdefault("OPENAI_API_KEY", "test-key-for-ci")
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-ci")
        os.environ.setdefault("GEMINI_API_KEY", "test-key-for-ci")
        os.environ.setdefault("XAI_API_KEY", "test-key-for-ci")

        # Set CI flag for tests to detect
        os.environ["SIFAKA_TEST_MODE"] = "true"
        os.environ["USE_MOCK_LLM"] = "true"


@pytest.fixture(autouse=True)
def mock_api_clients(request, monkeypatch):
    """Automatically mock API clients for unit tests."""
    # Only apply to unit tests, not integration tests
    if "unit" in str(request.fspath) or os.getenv("CI", "false").lower() == "true":
        # Mock openai client creation
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_openai.OpenAI.return_value = mock_client

        # Mock the actual API call
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked response"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_client.chat.completions.create.return_value = mock_response

        monkeypatch.setattr("openai.AsyncOpenAI", mock_openai.AsyncOpenAI)
        monkeypatch.setattr("openai.OpenAI", mock_openai.OpenAI)


# Mock pkg_resources for CI environments where it might not be available
try:
    import pkg_resources  # noqa: F401
except ImportError:
    # Create a mock pkg_resources module
    mock_pkg_resources = MagicMock()
    mock_entry_point = MagicMock()
    mock_entry_point.load.return_value = MagicMock
    mock_pkg_resources.iter_entry_points.return_value = []
    sys.modules["pkg_resources"] = mock_pkg_resources
