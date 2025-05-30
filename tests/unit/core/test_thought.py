#!/usr/bin/env python3
"""Unit tests for Thought core functionality.

This module contains comprehensive unit tests for the Thought class,
including initialization, validation results, critic feedback, and serialization.
"""

import json
from datetime import datetime

from sifaka.core.thought import CriticFeedback, Thought
from sifaka.validators.shared import ValidationResult


class TestThoughtInitialization:
    """Test Thought initialization and basic properties."""

    def test_basic_initialization(self):
        """Test basic Thought initialization."""
        thought = Thought(prompt="Test prompt", text="Test response text")

        assert thought.prompt == "Test prompt"
        assert thought.text == "Test response text"
        assert thought.model_prompt is None  # Not set by default
        assert isinstance(thought.timestamp, datetime)
        assert thought.id is not None
        assert len(thought.id) > 0

    def test_initialization_with_all_fields(self):
        """Test Thought initialization with all fields."""
        thought = Thought(
            prompt="User prompt",
            text="Generated text",
            model_prompt="System: You are helpful.\nUser: User prompt",
            system_prompt="You are helpful.",
            metadata={"key": "value"},
        )

        assert thought.prompt == "User prompt"
        assert thought.text == "Generated text"
        assert thought.model_prompt == "System: You are helpful.\nUser: User prompt"
        assert thought.system_prompt == "You are helpful."
        assert thought.metadata == {"key": "value"}

    def test_unique_ids(self):
        """Test that each Thought gets a unique ID."""
        thought1 = Thought(prompt="Test 1", text="Text 1")
        thought2 = Thought(prompt="Test 2", text="Text 2")

        assert thought1.id != thought2.id

    def test_timestamp_generation(self):
        """Test that timestamp is automatically generated."""
        before = datetime.now()
        thought = Thought(prompt="Test", text="Text")
        after = datetime.now()

        assert before <= thought.timestamp <= after

    def test_default_values(self):
        """Test default values for optional fields."""
        thought = Thought(prompt="Test", text="Text")

        assert thought.system_prompt is None
        assert thought.metadata == {}
        assert thought.validation_results is None
        assert thought.critic_feedback is None


class TestThoughtValidationResults:
    """Test Thought validation results functionality."""

    def test_add_validation_result(self):
        """Test adding a single validation result."""
        thought = Thought(prompt="Test", text="Text")
        result = ValidationResult(
            passed=True, message="Validation passed", validator_name="test_validator"
        )

        # Thought doesn't have add_validation_result method
        # Instead, validation_results is set directly by the chain
        thought.validation_results = {"test_validator": result}

        assert thought.validation_results is not None
        assert len(thought.validation_results) == 1
        assert thought.validation_results["test_validator"] == result

    def test_add_multiple_validation_results(self):
        """Test adding multiple validation results."""
        thought = Thought(prompt="Test", text="Text")

        result1 = ValidationResult(
            passed=True, message="First validation passed", validator_name="validator1"
        )
        result2 = ValidationResult(
            passed=False, message="Second validation failed", validator_name="validator2"
        )

        # Set validation results as dict
        thought.validation_results = {"validator1": result1, "validator2": result2}

        assert len(thought.validation_results) == 2
        assert thought.validation_results["validator1"] == result1
        assert thought.validation_results["validator2"] == result2

    def test_validation_passed_logic(self):
        """Test logic for determining if all validations passed."""

        def validation_passed(thought):
            """Helper function to check if all validations passed."""
            if not thought.validation_results:
                return True
            return all(result.passed for result in thought.validation_results.values())

        thought = Thought(prompt="Test", text="Text")

        # No validation results - should return True
        assert validation_passed(thought) is True

        # All validations pass
        result1 = ValidationResult(passed=True, message="Pass 1")
        result2 = ValidationResult(passed=True, message="Pass 2")
        thought.validation_results = {"v1": result1, "v2": result2}
        assert validation_passed(thought) is True

        # One validation fails
        result3 = ValidationResult(passed=False, message="Fail")
        thought.validation_results = {"v1": result1, "v2": result2, "v3": result3}
        assert validation_passed(thought) is False


class TestThoughtCriticFeedback:
    """Test Thought critic feedback functionality."""

    def test_add_critic_feedback(self):
        """Test adding critic feedback."""
        thought = Thought(prompt="Test", text="Text")
        feedback = CriticFeedback(
            critic_name="test_critic",
            feedback="This could be improved",
            needs_improvement=True,
            confidence=0.8,
        )

        # add_critic_feedback returns a new Thought
        new_thought = thought.add_critic_feedback(feedback)

        assert new_thought.critic_feedback is not None
        assert len(new_thought.critic_feedback) == 1
        assert new_thought.critic_feedback[0] == feedback

    def test_add_multiple_critic_feedback(self):
        """Test adding multiple critic feedback."""
        thought = Thought(prompt="Test", text="Text")

        feedback1 = CriticFeedback(
            critic_name="critic1", feedback="First feedback", needs_improvement=True, confidence=0.7
        )
        feedback2 = CriticFeedback(
            critic_name="critic2",
            feedback="Second feedback",
            needs_improvement=False,
            confidence=0.9,
        )

        # Chain the add_critic_feedback calls since they return new Thoughts
        thought_with_feedback = thought.add_critic_feedback(feedback1).add_critic_feedback(
            feedback2
        )

        assert len(thought_with_feedback.critic_feedback) == 2
        assert thought_with_feedback.critic_feedback[0] == feedback1
        assert thought_with_feedback.critic_feedback[1] == feedback2

    def test_needs_improvement_logic(self):
        """Test logic for determining if improvement is needed."""

        def needs_improvement(thought):
            """Helper function to check if any critic feedback indicates improvement is needed."""
            if not thought.critic_feedback:
                return False
            return any(feedback.needs_improvement for feedback in thought.critic_feedback)

        thought = Thought(prompt="Test", text="Text")

        # No critic feedback - should return False
        assert needs_improvement(thought) is False

        # No improvement needed
        feedback1 = CriticFeedback(
            critic_name="critic1", feedback="Good", needs_improvement=False, confidence=0.9
        )
        thought_with_feedback1 = thought.add_critic_feedback(feedback1)
        assert needs_improvement(thought_with_feedback1) is False

        # Improvement needed
        feedback2 = CriticFeedback(
            critic_name="critic2", feedback="Needs work", needs_improvement=True, confidence=0.8
        )
        thought_with_feedback2 = thought_with_feedback1.add_critic_feedback(feedback2)
        assert needs_improvement(thought_with_feedback2) is True


class TestThoughtSerialization:
    """Test Thought serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict functionality."""
        thought = Thought(prompt="Test prompt", text="Test text")

        data = thought.to_dict()

        assert isinstance(data, dict)
        assert data["prompt"] == "Test prompt"
        assert data["text"] == "Test text"
        assert "id" in data
        assert "timestamp" in data

    def test_to_dict_with_validation_results(self):
        """Test to_dict with validation results."""
        thought = Thought(prompt="Test", text="Text")
        result = ValidationResult(passed=True, message="Validation passed")
        # Use add_validation_result method which takes name and result
        thought_with_validation = thought.add_validation_result("test_validator", result)

        data = thought_with_validation.to_dict()

        assert "validation_results" in data
        assert len(data["validation_results"]) == 1
        assert data["validation_results"]["test_validator"]["passed"] is True

    def test_to_dict_with_critic_feedback(self):
        """Test to_dict with critic feedback."""
        thought = Thought(prompt="Test", text="Text")
        feedback = CriticFeedback(
            critic_name="test_critic", feedback="Good work", needs_improvement=False, confidence=0.9
        )
        thought_with_feedback = thought.add_critic_feedback(feedback)

        data = thought_with_feedback.to_dict()

        assert "critic_feedback" in data
        assert len(data["critic_feedback"]) == 1
        assert data["critic_feedback"][0]["critic_name"] == "test_critic"

    def test_from_dict_basic(self):
        """Test basic from_dict functionality."""
        data = {
            "id": "test-id",
            "prompt": "Test prompt",
            "text": "Test text",
            "model_prompt": "Test prompt",
            "timestamp": "2023-01-01T12:00:00",
            "model_name": "test-model",
            "metadata": {"key": "value"},
        }

        thought = Thought.from_dict(data)

        assert thought.id == "test-id"
        assert thought.prompt == "Test prompt"
        assert thought.text == "Test text"
        assert thought.model_name == "test-model"
        assert thought.metadata == {"key": "value"}

    def test_from_dict_with_validation_results(self):
        """Test from_dict with validation results."""
        data = {
            "id": "test-id",
            "prompt": "Test",
            "text": "Text",
            "model_prompt": "Test",
            "timestamp": "2023-01-01T12:00:00",
            "validation_results": [
                {"passed": True, "message": "Validation passed", "validator_name": "test_validator"}
            ],
        }

        thought = Thought.from_dict(data)

        assert thought.validation_results is not None
        assert len(thought.validation_results) == 1
        # validation_results is a dict, get the first validator result
        validator_result = list(thought.validation_results.values())[0]
        assert validator_result.passed is True

    def test_serialization_roundtrip(self):
        """Test that serialization roundtrip preserves data."""
        original = Thought(
            prompt="Test prompt",
            text="Test text",
            model_name="test-model",
            metadata={"key": "value"},
        )

        # Add validation result
        result = ValidationResult(
            passed=True, message="Validation passed", validator_name="test_validator"
        )
        original = original.add_validation_result(result)

        # Add critic feedback
        feedback = CriticFeedback(
            critic_name="test_critic", feedback="Good work", needs_improvement=False, confidence=0.9
        )
        original = original.add_critic_feedback(feedback)

        # Serialize and deserialize
        data = original.to_dict()
        restored = Thought.from_dict(data)

        # Check that all data is preserved
        assert restored.id == original.id
        assert restored.prompt == original.prompt
        assert restored.text == original.text
        assert restored.model_name == original.model_name
        assert restored.metadata == original.metadata
        assert len(restored.validation_results or {}) == len(original.validation_results or {})
        assert len(restored.critic_feedback or []) == len(original.critic_feedback or [])

    def test_json_serialization(self):
        """Test JSON serialization compatibility."""
        thought = Thought(prompt="Test prompt", text="Test text", model_name="test-model")

        # Should be JSON serializable
        data = thought.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Should be able to restore from JSON data
        restored_thought = Thought.from_dict(restored_data)
        assert restored_thought.prompt == thought.prompt
        assert restored_thought.text == thought.text


class TestThoughtComparison:
    """Test Thought comparison and equality."""

    def test_equality_same_content(self):
        """Test equality with same content."""
        thought1 = Thought(prompt="Test", text="Text")
        thought2 = Thought(prompt="Test", text="Text")

        # Different instances with same content should not be equal (different IDs)
        assert thought1 != thought2
        assert thought1.id != thought2.id

    def test_equality_same_id(self):
        """Test equality with same ID."""
        thought1 = Thought(prompt="Test", text="Text")

        # Create thought with same ID
        data = thought1.to_dict()
        thought2 = Thought.from_dict(data)

        # Should be equal if IDs match
        assert thought1.id == thought2.id

    def test_string_representation(self):
        """Test string representation."""
        thought = Thought(prompt="Test prompt", text="Test text")

        str_repr = str(thought)
        assert "Test prompt" in str_repr or "Test text" in str_repr

        repr_str = repr(thought)
        assert "Thought" in repr_str
        assert thought.id in repr_str


class TestThoughtEdgeCases:
    """Test Thought edge cases and error handling."""

    def test_empty_text(self):
        """Test Thought with empty text."""
        thought = Thought(prompt="Test", text="")

        assert thought.text == ""
        assert thought.prompt == "Test"

    def test_none_values(self):
        """Test Thought with None values where allowed."""
        thought = Thought(prompt="Test", text="Text")

        # These should be allowed to be None
        thought.system_prompt = None
        thought.model_name = None

        assert thought.system_prompt is None
        assert thought.model_name is None

    def test_large_text(self):
        """Test Thought with very large text."""
        large_text = "A" * 10000
        thought = Thought(prompt="Test", text=large_text)

        assert len(thought.text) == 10000
        assert thought.text == large_text

    def test_unicode_content(self):
        """Test Thought with unicode content."""
        thought = Thought(
            prompt="Test with unicode: 你好世界 🌍",
            text="Response with unicode: café naïve résumé 🎉",
        )

        assert "你好世界" in thought.prompt
        assert "café" in thought.text
        assert "🎉" in thought.text
