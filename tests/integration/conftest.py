"""Configuration for integration tests."""

import os
from typing import Optional

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires API keys)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly requested."""
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(
            reason="Integration tests require --integration flag"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
    parser.addoption(
        "--llm-provider",
        action="store",
        default="openai",
        help="LLM provider to use for integration tests (openai, anthropic, google)",
    )


@pytest.fixture
def api_key(request) -> Optional[str]:
    """Get API key for the selected provider."""
    provider = request.config.getoption("--llm-provider")

    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = key_mapping.get(provider)
    if not env_var:
        pytest.skip(f"Unknown provider: {provider}")

    api_key = os.getenv(env_var)
    if not api_key:
        pytest.skip(
            f"No API key found for {provider}. Set {env_var} environment variable."
        )

    return api_key


@pytest.fixture
def llm_provider(request) -> str:
    """Get the LLM provider name."""
    return request.config.getoption("--llm-provider")


@pytest.fixture
def integration_timeout() -> float:
    """Get timeout for integration tests."""
    return float(os.getenv("INTEGRATION_TEST_TIMEOUT", "30.0"))


@pytest.fixture
def use_mocks() -> bool:
    """Check if mocks should be used (for CI environments)."""
    return (
        os.getenv("CI", "false").lower() == "true"
        or os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    )


@pytest.fixture
def mock_llm_provider(use_mocks, llm_provider):
    """Provide mock LLM for CI testing."""
    if not use_mocks:
        return None

    from .mock_responses import create_mock_llm

    return create_mock_llm(llm_provider)
