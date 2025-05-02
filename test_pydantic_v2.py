"""
Test script to verify Pydantic v2 updates.
"""

import sys
import pytest
from typing import Dict, Any, List

from sifaka.critics.models import CriticConfig
from sifaka.classifiers.base import ClassificationResult
from sifaka.rules.factual.accuracy import AccuracyConfig, create_accuracy_validator


def test_critic_config():
    """Test CriticConfig with Pydantic v2 features."""
    print("Testing CriticConfig...")

    # Test valid config
    config = CriticConfig(
        name="test_critic",
        description="A test critic",
        min_confidence=0.8,
        max_attempts=5,
        cache_size=200,
        priority=2,
        cost=2.0,
    )
    print(f"Valid config created: {config}")
    assert config.name == "test_critic"
    assert config.description == "A test critic"
    assert config.min_confidence == 0.8
    assert config.max_attempts == 5
    assert config.cache_size == 200
    assert config.priority == 2
    assert config.cost == 2.0

    # Test validation constraints
    with pytest.raises(ValueError):
        CriticConfig(
            name="", description="test"  # Should fail min_length validation
        )

    with pytest.raises(ValueError):
        CriticConfig(
            name="test", description="test", min_confidence=1.5  # Should fail le=1.0 validation
        )


def test_classification_result():
    """Test ClassificationResult with Pydantic v2 features."""
    print("Testing ClassificationResult...")

    # Test valid result
    result = ClassificationResult(
        label="positive",
        confidence=0.95,
        metadata={"source": "test"},
    )
    print(f"Valid result created: {result}")
    assert result.label == "positive"
    assert result.confidence == 0.95
    assert result.metadata == {"source": "test"}

    # Test validation constraints
    with pytest.raises(ValueError):
        ClassificationResult(
            label="positive", confidence=1.5  # Should fail le=1.0 validation
        )

    with pytest.raises(ValueError):
        ClassificationResult(
            label="positive", confidence=-0.1  # Should fail ge=0.0 validation
        )


def test_accuracy_config():
    """Test AccuracyConfig with Pydantic v2 features."""
    print("Testing AccuracyConfig...")

    # Test valid config
    config = AccuracyConfig(
        knowledge_base=["Fact 1", "Fact 2"],
        threshold=0.7,
        cache_size=50
    )
    print(f"Valid config created: {config}")
    assert config.knowledge_base == ["Fact 1", "Fact 2"]
    assert config.threshold == 0.7
    assert config.cache_size == 50

    # Test validation constraints
    with pytest.raises(ValueError):
        AccuracyConfig(
            knowledge_base=[],  # Should fail min_length validation
            threshold=0.7,
            cache_size=50
        )

    with pytest.raises(ValueError):
        AccuracyConfig(
            knowledge_base=["Fact 1"],
            threshold=1.5,  # Should fail le=1.0 validation
            cache_size=50
        )


def main():
    """Run all tests."""
    test_critic_config()
    test_classification_result()
    test_accuracy_config()
    print("All tests passed!")


if __name__ == "__main__":
    main()
