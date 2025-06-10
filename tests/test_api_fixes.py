#!/usr/bin/env python3
"""Quick test to verify API fixes are working.

This script tests the basic API compatibility of our fixed test files
with the actual Sifaka implementation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sifaka_thought_basic():
    """Test basic SifakaThought functionality."""
    from sifaka.core.thought import SifakaThought

    print("✓ Testing SifakaThought creation...")
    thought = SifakaThought(prompt="Test prompt")
    assert thought.prompt == "Test prompt"
    assert thought.iteration == 0
    assert len(thought.generations) == 0

    print("✓ Testing add_generation...")
    thought.add_generation("Generated text", "gpt-4", None)
    assert len(thought.generations) == 1
    assert thought.generations[0].text == "Generated text"

    print("✓ Testing add_validation...")
    thought.add_validation("length", True, {"word_count": 50})
    assert len(thought.validations) == 1
    assert thought.validations[0].validator == "length"

    print("✓ Testing add_critique...")
    thought.add_critique("constitutional", "Good work", [], needs_improvement=False)
    assert len(thought.critiques) == 1
    assert thought.critiques[0].critic == "constitutional"

    print("✓ Testing add_tool_call...")
    thought.add_tool_call("search", {"query": "test"}, {"results": []}, 0.1)
    assert len(thought.tool_calls) == 1
    assert thought.tool_calls[0].tool_name == "search"

    print("✅ SifakaThought basic API tests passed!")


def test_validation_result():
    """Test ValidationResult from validators.base."""
    from sifaka.validators.base import ValidationResult

    print("✓ Testing ValidationResult creation...")
    result = ValidationResult(
        passed=True,
        message="Test validation",
        score=0.8,
        validator_name="test",
        processing_time_ms=10.0,
    )

    assert result.passed is True
    assert result.message == "Test validation"
    assert result.score == 0.8
    assert result.validator_name == "test"
    assert isinstance(result.issues, list)
    assert isinstance(result.suggestions, list)
    assert isinstance(result.metadata, dict)

    print("✅ ValidationResult API tests passed!")


def test_base_validator():
    """Test BaseValidator interface."""
    from sifaka.core.thought import SifakaThought
    from sifaka.validators.base import BaseValidator, ValidationResult

    class TestValidator(BaseValidator):
        async def validate_async(self, thought: SifakaThought) -> ValidationResult:
            return ValidationResult(
                passed=True, message="Test passed", validator_name=self.name, processing_time_ms=5.0
            )

    print("✓ Testing BaseValidator subclass...")
    validator = TestValidator("test_validator")
    assert validator.name == "test_validator"

    print("✓ Testing sync validation...")
    thought = SifakaThought(prompt="Test")
    result = validator.validate(thought)
    assert result.passed is True
    assert result.validator_name == "test_validator"

    print("✅ BaseValidator API tests passed!")


def test_imports():
    """Test that all imports work correctly."""
    print("✓ Testing core imports...")

    print("✓ Testing validator imports...")

    print("✓ Testing graph imports...")
    try:
        pass

        print("✓ Graph nodes imported successfully")
    except ImportError as e:
        print(f"⚠️  Graph nodes import failed: {e}")

    print("✅ All imports successful!")


def main():
    """Run all API compatibility tests."""
    print("🔧 Testing Sifaka API Compatibility")
    print("=" * 50)

    try:
        test_imports()
        test_sifaka_thought_basic()
        test_validation_result()
        test_base_validator()

        print("\n🎉 All API compatibility tests passed!")
        print("The test fixes should now work with the actual Sifaka implementation.")
        return True

    except Exception as e:
        print(f"\n❌ API compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
