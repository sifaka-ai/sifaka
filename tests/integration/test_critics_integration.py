"""Integration tests for all critics with real LLM calls."""

import pytest

from sifaka import improve
from sifaka.critics import (
    ConstitutionalCritic,
    MetaRewardingCritic,
    NCriticsCritic,
    PromptCritic,
    ReflexionCritic,
    SelfConsistencyCritic,
    SelfRAGCritic,
    SelfRefineCritic,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "critic_class,critic_kwargs",
    [
        (ReflexionCritic, {}),
        (
            ConstitutionalCritic,
            {"principles": ["Be helpful", "Be harmless", "Be honest"]},
        ),
        (SelfRefineCritic, {}),
        (NCriticsCritic, {}),
        (SelfRAGCritic, {}),
        (MetaRewardingCritic, {}),
        (SelfConsistencyCritic, {}),
        (PromptCritic, {"prompt": "Improve clarity and conciseness"}),
    ],
)
def test_critic_integration(
    critic_class, critic_kwargs, api_key, llm_provider, integration_timeout
):
    """Test each critic with real LLM calls."""
    # Sample text that needs improvement
    original_text = """
    The company's new product will revolutionize the industry.
    It has many features that customers want.
    Sales will definitely increase significantly.
    """

    # Initialize critic
    critic = critic_class(**critic_kwargs)

    # Run improvement
    result = improve(
        original_text,
        critic=critic,
        max_iterations=2,  # Limit iterations for testing
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )

    # Assertions
    assert result.improved_text != original_text
    assert result.iterations > 0
    assert result.total_tokens > 0
    assert result.improvement_history
    assert len(result.improvement_history) == result.iterations

    # Check that each iteration has required fields
    for iteration in result.improvement_history:
        assert iteration.critique
        assert iteration.improved_text
        assert iteration.tokens_used > 0
        assert iteration.latency_ms > 0


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_iterations(api_key, llm_provider, integration_timeout):
    """Test multiple iterations with timeout handling."""
    original_text = "This is bad text that needs lots of improvement."

    result = improve(
        original_text,
        max_iterations=5,
        timeout=integration_timeout,
        llm_provider=llm_provider,
        api_key=api_key,
    )

    assert result.improved_text != original_text
    assert 1 <= result.iterations <= 5
    assert result.total_tokens > 0

    # Check that improvements are actually different
    texts = [original_text] + [h.improved_text for h in result.improvement_history]
    unique_texts = set(texts)
    assert len(unique_texts) > 1


@pytest.mark.integration
def test_timeout_handling(api_key, llm_provider):
    """Test that timeout is properly handled."""
    original_text = "Short text."

    # Use very short timeout to trigger timeout
    with pytest.raises(TimeoutError):
        improve(
            original_text,
            max_iterations=10,
            timeout=0.001,  # 1ms timeout should trigger
            llm_provider=llm_provider,
            api_key=api_key,
        )


@pytest.mark.integration
def test_error_recovery(api_key, llm_provider, integration_timeout):
    """Test error recovery with retry logic."""
    # Use text that might cause issues
    problematic_text = ""

    with pytest.raises(ValueError):
        improve(
            problematic_text,
            llm_provider=llm_provider,
            api_key=api_key,
        )


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_provider_switching(provider, integration_timeout):
    """Test switching between different LLM providers."""
    # Skip if API key not available
    import os

    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    api_key = os.getenv(key_mapping[provider])
    if not api_key:
        pytest.skip(f"No API key for {provider}")

    original_text = "Test text for provider switching."

    result = improve(
        original_text,
        max_iterations=1,
        timeout=integration_timeout,
        llm_provider=provider,
        api_key=api_key,
    )

    assert result.improved_text != original_text
    assert result.iterations == 1
    assert result.metadata.get("llm_provider") == provider
