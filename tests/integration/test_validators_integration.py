"""Integration tests for validators with real LLM calls."""

import pytest

from sifaka import improve
from sifaka.core.config import Config
from sifaka.validators import (
    ComposableValidator,
    ContentValidator,
    FormatValidator,
    LengthValidator,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_length_validator_integration(api_key, llm_provider, integration_timeout):
    """Test length validator with real improvements."""
    original_text = "This is a very short text that needs to be expanded significantly to meet the minimum length requirements."

    # Require longer text
    validator = LengthValidator(min_length=500, max_length=1000)

    result = await improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        config=Config(timeout_seconds=integration_timeout),
    )

    # Check that final text meets length requirements
    assert 500 <= len(result.final_text) <= 1000
    assert result.final_text != original_text

    # Verify validation was applied
    assert len(result.validations) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_format_validator_integration(api_key, llm_provider, integration_timeout):
    """Test format validator with structured output requirements."""
    original_text = "Write a summary with three bullet points about AI safety."

    # Require paragraph format (3-5 paragraphs)
    validator = FormatValidator(min_paragraphs=3, max_paragraphs=5)

    result = await improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        config=Config(timeout_seconds=integration_timeout),
    )

    # Check format - should have 3-5 paragraphs
    paragraphs = [p.strip() for p in result.final_text.split("\n\n") if p.strip()]
    assert 3 <= len(paragraphs) <= 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_content_validator_integration(
    api_key, llm_provider, integration_timeout
):
    """Test content validator for specific requirements."""
    original_text = "Explain machine learning."

    # Require specific content elements
    validator = ContentValidator(
        required_terms=["supervised learning", "unsupervised learning", "examples"],
        forbidden_terms=["jargon", "complex mathematics"],
    )

    result = await improve(
        original_text,
        validators=[validator],
        max_iterations=3,
        config=Config(timeout_seconds=integration_timeout),
    )

    # Check content requirements
    lower_text = result.final_text.lower()
    assert "supervised learning" in lower_text
    assert "unsupervised learning" in lower_text
    assert "example" in lower_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_composite_validator_integration(
    api_key, llm_provider, integration_timeout
):
    """Test multiple validators working together."""
    original_text = "Write about Python programming."

    # Combine multiple validators
    validators = [
        LengthValidator(min_length=200, max_length=500),
        FormatValidator(min_paragraphs=2),
        ContentValidator(required_terms=["functions", "classes", "modules"]),
    ]

    composite = ComposableValidator(validators=validators)

    result = await improve(
        original_text,
        validators=[composite],
        max_iterations=4,
        config=Config(timeout_seconds=integration_timeout),
    )

    # Check all requirements are met
    assert 200 <= len(result.final_text) <= 500
    paragraphs = [p for p in result.final_text.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2

    lower_text = result.final_text.lower()
    assert all(term in lower_text for term in ["function", "class", "module"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validator_rejection_handling(api_key, llm_provider, integration_timeout):
    """Test handling of validator rejections."""
    original_text = "Short."

    # Very strict validator that's hard to satisfy
    validator = LengthValidator(min_length=1000, max_length=1001)

    result = await improve(
        original_text,
        validators=[validator],
        max_iterations=5,  # Give it several attempts
        config=Config(timeout_seconds=integration_timeout),
    )

    # Should eventually satisfy the validator or use best attempt
    assert len(result.final_text) > len(original_text)

    # Check that validation attempts were made
    assert len(result.validations) > 0
