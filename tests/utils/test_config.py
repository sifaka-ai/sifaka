"""
Tests for the configuration utilities.
"""

import pytest
from pydantic import ValidationError
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.config.chain import ChainConfig
from sifaka.utils.config.rules import RuleConfig
from sifaka.utils.config.critics import CriticConfig
from sifaka.utils.config.retrieval import RetrieverConfig


def test_model_config():
    """Test that ModelConfig works correctly."""
    # Test with valid parameters
    config = ModelConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
    )

    assert config.model == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 100

    # Test with invalid temperature
    with pytest.raises(ValidationError):
        ModelConfig(
            model="gpt-4",
            temperature=2.0,  # Temperature should be between 0 and 1
        )

    # Test with invalid max_tokens
    with pytest.raises(ValidationError):
        ModelConfig(
            model="gpt-4",
            max_tokens=-1,  # Max tokens should be positive
        )


def test_chain_config():
    """Test that ChainConfig works correctly."""
    # Test with valid parameters
    config = ChainConfig(
        max_attempts=3,
        timeout_seconds=30,
        retry_delay=1.0,
    )

    assert config.max_attempts == 3
    assert config.timeout_seconds == 30
    assert config.retry_delay == 1.0

    # Test with timeout_seconds parameter
    config2 = ChainConfig(
        max_attempts=3,
        timeout_seconds=45,
        retry_delay=1.0,
    )

    assert config2.timeout_seconds == 45

    # Test with invalid max_attempts
    with pytest.raises(ValidationError):
        ChainConfig(
            max_attempts=0,  # Max attempts should be positive
        )

    # Test with invalid timeout_seconds
    with pytest.raises(ValidationError):
        ChainConfig(
            timeout_seconds=-1,  # Timeout should be positive
        )


def test_rule_config():
    """Test that RuleConfig works correctly."""
    # Test with valid parameters
    config = RuleConfig(
        params={
            "min_chars": 10,
            "max_chars": 100,
        },
        priority="HIGH",
    )

    assert config.params["min_chars"] == 10
    assert config.params["max_chars"] == 100
    assert config.priority == "HIGH"

    # Test with another valid priority
    config = RuleConfig(
        priority="LOW",
    )
    assert config.priority == "LOW"

    # Test with default priority
    config = RuleConfig()
    assert config.priority == "MEDIUM"  # Default value


def test_critic_config():
    """Test that CriticConfig works correctly."""
    # Test with valid parameters
    config = CriticConfig(
        system_prompt="You are a helpful critic.",
        temperature=0.7,
        max_tokens=100,
    )

    assert config.system_prompt == "You are a helpful critic."
    assert config.temperature == 0.7
    assert config.max_tokens == 100

    # Test with invalid temperature
    with pytest.raises(ValidationError):
        CriticConfig(
            system_prompt="You are a helpful critic.",
            temperature=2.0,  # Temperature should be between 0 and 1
        )


def test_retriever_config():
    """Test that RetrieverConfig works correctly."""
    # Test with valid parameters
    config = RetrieverConfig(
        max_results=5,
        min_score=0.1,
    )

    assert config.max_results == 5
    assert config.min_score == 0.1

    # Test with invalid max_results
    with pytest.raises(ValidationError):
        RetrieverConfig(
            max_results=0,  # Max results should be positive
        )

    # Test with invalid min_score
    with pytest.raises(ValidationError):
        RetrieverConfig(
            min_score=-0.1,  # Min score should be non-negative
        )
