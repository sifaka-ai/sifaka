"""Configuration for unit tests."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure we're using mock API keys for unit tests
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("XAI_API_KEY", "test-key")
os.environ["USE_MOCK_LLM"] = "true"


@pytest.fixture(autouse=True)
def mock_llm_clients():
    """Mock all LLM clients for unit tests."""
    with (
        patch("openai.AsyncOpenAI") as mock_async_openai,
        patch("openai.OpenAI") as mock_openai,
        patch("anthropic.AsyncAnthropic") as mock_anthropic,
        patch("google.generativeai.GenerativeModel") as mock_google,
    ):
        # Mock OpenAI responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mocked LLM response"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        # Set up async client
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_openai.return_value = async_client

        # Set up sync client
        sync_client = MagicMock()
        sync_client.chat.completions.create = MagicMock(return_value=mock_response)
        mock_openai.return_value = sync_client

        # Mock Anthropic
        anthropic_response = MagicMock()
        anthropic_response.content = [MagicMock(text="Mocked Anthropic response")]
        anthropic_client = MagicMock()
        anthropic_client.messages.create = AsyncMock(return_value=anthropic_response)
        mock_anthropic.return_value = anthropic_client

        # Mock Google
        google_response = MagicMock()
        google_response.text = "Mocked Google response"
        google_model = MagicMock()
        google_model.generate_content_async = AsyncMock(return_value=google_response)
        mock_google.return_value = google_model

        yield {
            "openai_async": mock_async_openai,
            "openai_sync": mock_openai,
            "anthropic": mock_anthropic,
            "google": mock_google,
        }


@pytest.fixture
def mock_sifaka_result():
    """Create a mock SifakaResult for testing."""
    from sifaka.core.models import (
        CritiqueResult,
        Generation,
        SifakaResult,
        ValidationResult,
    )

    return SifakaResult(
        original_text="Test text",
        final_text="Improved test text",
        iteration=1,
        generations=[
            Generation(
                iteration=1,
                text="Improved test text",
                model="gpt-4o-mini",
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
        ],
        critiques=[
            CritiqueResult(
                critic="test_critic",
                feedback="Test feedback",
                suggestions=["Test suggestion"],
                needs_improvement=False,
                confidence=0.8,
            )
        ],
        validations=[
            ValidationResult(
                validator="test_validator",
                passed=True,
                score=1.0,
                details="Test validation passed",
            )
        ],
        processing_time=1.0,
    )


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from sifaka.core.config import Config, CriticConfig, EngineConfig, LLMConfig
    from sifaka.core.types import CriticType

    return Config(
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
        ),
        critic=CriticConfig(
            critics=[CriticType.SELF_REFINE],
            confidence_threshold=0.6,
        ),
        engine=EngineConfig(
            max_iterations=3,
            parallel_critics=False,
        ),
    )


# Mock pkg_resources if not available
try:
    import pkg_resources  # noqa: F401
except ImportError:
    mock_pkg = MagicMock()
    mock_pkg.iter_entry_points.return_value = []
    sys.modules["pkg_resources"] = mock_pkg
