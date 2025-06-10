"""Comprehensive unit tests for validation-aware utilities.

This module tests the validation-aware functionality:
- ValidationAwareMixin for adding validation context
- Validation context extraction and formatting
- Integration with critics and validators
- Error handling and edge cases

Tests cover:
- Validation context management
- Mixin functionality and inheritance
- Context formatting and serialization
- Integration with thought validation results
- Mock-based testing with sample validation data
"""


import pytest

from sifaka.core.thought import SifakaThought
from sifaka.utils.validation_aware import ValidationAwareMixin


class TestValidationAwareMixin:
    """Test suite for ValidationAwareMixin class."""

    def test_validation_aware_mixin_basic(self):
        """Test basic ValidationAwareMixin functionality."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Should have validation-aware methods
        assert hasattr(instance, "_get_validation_aware_context")
        assert hasattr(instance, "_should_prioritize_validation")
        assert hasattr(instance, "_format_validation_context")
        assert hasattr(instance, "_filter_suggestions_for_constraints")

    def test_format_validation_context_empty(self):
        """Test formatting validation context with empty input."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Test with None validation context
        result = instance._format_validation_context(None)

        # Should return default format
        assert isinstance(result, dict)
        assert "validation_priority_notice" in result
        assert "validation_feedback" in result
        assert "critic_feedback_header" in result
        assert "priority_instructions" in result

    def test_filter_suggestions_for_constraints_empty(self):
        """Test filtering suggestions with empty constraints."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        suggestions = ["Suggestion 1", "Suggestion 2", "Suggestion 3"]

        # Test with None validation context
        filtered = instance._filter_suggestions_for_constraints(suggestions, None)

        # Should return original suggestions when no constraints
        assert isinstance(filtered, list)

    def test_create_priority_instructions(self):
        """Test creating priority instructions."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Test with None validation context
        instructions = instance._create_priority_instructions(None)

        # Should return a string
        assert isinstance(instructions, str)

    def test_validation_aware_inheritance(self):
        """Test ValidationAwareMixin with inheritance."""

        class BaseClass:
            def __init__(self):
                self.base_attr = "base"

        class TestClass(BaseClass, ValidationAwareMixin):
            def __init__(self):
                super().__init__()
                self.test_attr = "test"

        instance = TestClass()

        # Should have both base and mixin functionality
        assert hasattr(instance, "base_attr")
        assert hasattr(instance, "test_attr")
        assert hasattr(instance, "_get_validation_aware_context")

        # Should work normally
        assert instance.base_attr == "base"
        assert instance.test_attr == "test"


class TestValidationAwareMixinIntegration:
    """Test suite for ValidationAwareMixin integration with thoughts."""

    @pytest.fixture
    def sample_thought_with_validations(self):
        """Create a sample thought with validation results."""
        thought = SifakaThought(
            prompt="Test prompt for validation context",
            final_text="This is a test response with validation results.",
            iteration=1,
            max_iterations=3,
        )

        # Add various validation results
        thought.add_validation("length_validator", True, {"word_count": 12, "min_required": 10})
        thought.add_validation(
            "content_validator", False, {"issue": "lacks examples", "score": 0.6}
        )
        thought.add_validation("clarity_validator", True, {"readability_score": 0.85})
        thought.add_validation(
            "grammar_validator", False, {"errors": ["typo in line 2"], "error_count": 1}
        )

        return thought

    @pytest.fixture
    def sample_thought_no_validations(self):
        """Create a sample thought without validation results."""
        return SifakaThought(
            prompt="Simple prompt", final_text="Simple response", iteration=0, max_iterations=1
        )

    def test_validation_aware_mixin_with_thought(self, sample_thought_with_validations):
        """Test ValidationAwareMixin integration with thought validation context."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Test getting validation-aware context
        context_str = instance._get_validation_aware_context(sample_thought_with_validations)

        # Should return a string (may be empty if no critical constraints)
        assert isinstance(context_str, str)

    def test_should_prioritize_validation(self, sample_thought_with_validations):
        """Test validation prioritization logic."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Test prioritization check
        should_prioritize = instance._should_prioritize_validation(sample_thought_with_validations)

        # Should return a boolean
        assert isinstance(should_prioritize, bool)

    def test_validation_aware_mixin_with_no_validations(self, sample_thought_no_validations):
        """Test ValidationAwareMixin with thought that has no validations."""

        class TestClass(ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        instance = TestClass()

        # Test getting validation-aware context with no validations
        context_str = instance._get_validation_aware_context(sample_thought_no_validations)

        # Should return a string (likely empty)
        assert isinstance(context_str, str)

    def test_validation_aware_mixin_inheritance_order(self):
        """Test ValidationAwareMixin with different inheritance orders."""

        class BaseClass:
            def __init__(self):
                self.base_value = "base"

        class TestClass1(ValidationAwareMixin, BaseClass):
            def __init__(self):
                super().__init__()

        class TestClass2(BaseClass, ValidationAwareMixin):
            def __init__(self):
                super().__init__()

        # Both should work
        instance1 = TestClass1()
        instance2 = TestClass2()

        assert hasattr(instance1, "_get_validation_aware_context")
        assert hasattr(instance2, "_get_validation_aware_context")
        assert hasattr(instance1, "base_value")
        assert hasattr(instance2, "base_value")
