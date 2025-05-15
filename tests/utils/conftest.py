"""
Pytest fixtures for utility tests.
"""

import pytest
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.config.chain import ChainConfig
from sifaka.utils.config.rules import RuleConfig, RulePriority
from sifaka.utils.config.critics import CriticConfig
from sifaka.utils.config.retrieval import RetrieverConfig


@pytest.fixture
def model_config():
    """Fixture for a model configuration."""
    config = ModelConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
    )
    return config


@pytest.fixture
def chain_config():
    """Fixture for a chain configuration."""
    config = ChainConfig(
        max_attempts=3,
        timeout_seconds=30,
        retry_delay=1.0,
    )
    return config


@pytest.fixture
def rule_config():
    """Fixture for a rule configuration."""
    config = RuleConfig(
        params={
            "min_chars": 10,
            "max_chars": 100,
        },
        priority=RulePriority.HIGH,
    )
    return config


@pytest.fixture
def critic_config():
    """Fixture for a critic configuration."""
    config = CriticConfig(
        system_prompt="You are a helpful critic.",
        temperature=0.7,
        max_tokens=100,
    )
    return config


@pytest.fixture
def retriever_config():
    """Fixture for a retriever configuration."""
    config = RetrieverConfig(
        max_results=5,
        min_score=0.1,
    )
    return config
