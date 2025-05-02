"""
Test script to verify Pydantic v2 updates.
"""

import sys
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

    # Test validation constraints
    try:
        invalid_config = CriticConfig(
            name="", description="test"  # Should fail min_length validation
        )
        print("ERROR: Empty name validation failed")
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")

    try:
        invalid_config = CriticConfig(
            name="test", description="test", min_confidence=1.5  # Should fail le=1.0 validation
        )
        print("ERROR: min_confidence validation failed")
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")

    return True


def test_classification_result():
    """Test ClassificationResult with Pydantic v2 features."""
    print("\nTesting ClassificationResult...")

    # Since ClassificationResult is frozen, we need to provide all fields at initialization
    result = ClassificationResult(label="positive", confidence=0.95, metadata={"source": "test"})
    print(f"Valid result created: {result}")

    # Test with_metadata method (which creates a new instance)
    new_result = result.with_metadata(extra="info")
    print(f"Result with added metadata: {new_result}")

    # Test validation constraints
    try:
        invalid_result = ClassificationResult(
            label="negative", confidence=1.5  # Should fail le=1.0 validation
        )
        print("ERROR: confidence validation failed")
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")

    return True


def test_accuracy_config():
    """Test AccuracyConfig with Pydantic v2 features."""
    print("\nTesting AccuracyConfig...")

    # Test valid config
    config = AccuracyConfig(knowledge_base=["Fact 1", "Fact 2"], threshold=0.7, cache_size=50)
    print(f"Valid config created: {config}")

    # Test validation constraints
    try:
        invalid_config = AccuracyConfig(
            knowledge_base=[], threshold=0.5  # Should fail min_length validation
        )
        print("ERROR: empty knowledge_base validation failed")
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")

    # Test validator creation
    validator = create_accuracy_validator(knowledge_base=["The Earth is round"], threshold=0.6)
    print(f"Created validator with config: {validator._config}")

    return True


def main():
    """Run all tests."""
    print("Testing Pydantic v2 updates...\n")

    all_passed = True
    all_passed &= test_critic_config()
    all_passed &= test_classification_result()
    all_passed &= test_accuracy_config()

    if all_passed:
        print("\nAll tests passed! Pydantic v2 updates are working correctly.")
        return 0
    else:
        print("\nSome tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
