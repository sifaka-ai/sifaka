"""Helper functions for mocking LLM calls in integration tests."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from .mock_responses import create_mock_improvement_result, create_mock_llm


@contextmanager
def mock_llm_calls(provider: str = "openai") -> Generator[MagicMock, None, None]:
    """Context manager to mock LLM calls for integration tests."""
    if not should_use_mocks():
        yield None
        return

    mock_llm = create_mock_llm(provider)

    # Mock different LLM provider imports
    patches = []

    if provider == "openai":
        patches.extend(
            [
                patch("sifaka.llms.openai.OpenAI", return_value=mock_llm),
                patch("sifaka.llms.openai.AsyncOpenAI", return_value=mock_llm),
            ]
        )
    elif provider == "anthropic":
        patches.extend(
            [
                patch("sifaka.llms.anthropic.Anthropic", return_value=mock_llm),
                patch("sifaka.llms.anthropic.AsyncAnthropic", return_value=mock_llm),
            ]
        )
    elif provider == "google":
        patches.extend(
            [
                patch(
                    "sifaka.llms.google.genai.GenerativeModel", return_value=mock_llm
                ),
            ]
        )

    # Apply all patches
    for p in patches:
        p.start()

    try:
        yield mock_llm
    finally:
        # Stop all patches
        for p in patches:
            p.stop()


@contextmanager
def mock_improve_function():
    """Mock the main improve functions for faster CI testing."""
    if not should_use_mocks():
        yield None
        return

    def mock_improve_sync(text: str, **kwargs) -> Any:
        """Mock synchronous improve function."""
        iterations = kwargs.get("max_iterations", 1)
        critic = (
            kwargs.get("critics", ["reflexion"])[0]
            if kwargs.get("critics")
            else "reflexion"
        )

        result = create_mock_improvement_result(text, iterations, critic)

        # Create a mock result object
        mock_result = MagicMock()
        mock_result.final_text = result["final_text"]
        mock_result.original_text = result["original_text"]
        mock_result.iterations = result["iterations"]
        mock_result.total_tokens = result["total_tokens"]
        mock_result.improvement_history = result["improvement_history"]
        mock_result.metadata = result["metadata"]

        return mock_result

    async def mock_improve_async(text: str, **kwargs) -> Any:
        """Mock asynchronous improve function."""
        return mock_improve_sync(text, **kwargs)

    with (
        patch("sifaka.improve_sync", side_effect=mock_improve_sync),
        patch("sifaka.improve_async", side_effect=mock_improve_async),
    ):
        yield


def should_use_mocks() -> bool:
    """Check if mocks should be used."""
    return (
        os.getenv("CI", "false").lower() == "true"
        or os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    )


def setup_ci_environment():
    """Set up environment for CI testing with mocks."""
    if should_use_mocks():
        # Set dummy API keys for CI
        os.environ.setdefault("OPENAI_API_KEY", "mock-key")
        os.environ.setdefault("ANTHROPIC_API_KEY", "mock-key")
        os.environ.setdefault("GEMINI_API_KEY", "mock-key")

        # Reduce timeouts for faster CI
        os.environ.setdefault("INTEGRATION_TEST_TIMEOUT", "5.0")
