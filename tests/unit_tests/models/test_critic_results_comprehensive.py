"""Comprehensive unit tests for critic result models.

This module provides extensive testing for critic result data structures:
- CriticResult: Main result container for critic feedback
- CritiqueFeedback: Individual feedback items
- Model validation and constraints
- Serialization and deserialization
- Edge cases and error handling
- Integration with Pydantic validation
"""


import pytest
from pydantic import ValidationError

from sifaka.models.critic_results import CriticResult, CritiqueFeedback


class TestCritiqueFeedback:
    """Test CritiqueFeedback data model."""

    def test_critique_feedback_creation(self):
        """Test creating a valid CritiqueFeedback."""
        feedback = CritiqueFeedback(
            feedback="This text needs improvement",
            suggestions=["Add more examples", "Improve clarity"],
            confidence=0.85,
            category="clarity",
        )

        assert feedback.feedback == "This text needs improvement"
        assert feedback.suggestions == ["Add more examples", "Improve clarity"]
        assert feedback.confidence == 0.85
        assert feedback.category == "clarity"

    def test_critique_feedback_minimal(self):
        """Test CritiqueFeedback with minimal required fields."""
        feedback = CritiqueFeedback(feedback="Basic feedback")

        assert feedback.feedback == "Basic feedback"
        assert feedback.suggestions == []
        assert feedback.confidence is None
        assert feedback.category is None

    def test_critique_feedback_empty_suggestions(self):
        """Test CritiqueFeedback with empty suggestions list."""
        feedback = CritiqueFeedback(feedback="No suggestions", suggestions=[])

        assert feedback.feedback == "No suggestions"
        assert feedback.suggestions == []

    def test_critique_feedback_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        CritiqueFeedback(feedback="test", confidence=0.0)
        CritiqueFeedback(feedback="test", confidence=0.5)
        CritiqueFeedback(feedback="test", confidence=1.0)

        # Invalid confidence scores should raise ValidationError
        with pytest.raises(ValidationError):
            CritiqueFeedback(feedback="test", confidence=-0.1)

        with pytest.raises(ValidationError):
            CritiqueFeedback(feedback="test", confidence=1.1)

    def test_critique_feedback_string_validation(self):
        """Test string field validation."""
        # Empty feedback should be allowed
        feedback = CritiqueFeedback(feedback="")
        assert feedback.feedback == ""

        # None feedback should raise ValidationError
        with pytest.raises(ValidationError):
            CritiqueFeedback(feedback=None)

    def test_critique_feedback_suggestions_validation(self):
        """Test suggestions list validation."""
        # Valid suggestions
        feedback = CritiqueFeedback(feedback="test", suggestions=["suggestion 1", "suggestion 2"])
        assert len(feedback.suggestions) == 2

        # Empty strings in suggestions should be allowed
        feedback = CritiqueFeedback(feedback="test", suggestions=["", "valid suggestion"])
        assert feedback.suggestions[0] == ""
        assert feedback.suggestions[1] == "valid suggestion"

    def test_critique_feedback_serialization(self):
        """Test serialization to dict."""
        feedback = CritiqueFeedback(
            feedback="Test feedback",
            suggestions=["suggestion 1", "suggestion 2"],
            confidence=0.9,
            category="quality",
        )

        data = feedback.model_dump()

        assert data["feedback"] == "Test feedback"
        assert data["suggestions"] == ["suggestion 1", "suggestion 2"]
        assert data["confidence"] == 0.9
        assert data["category"] == "quality"

    def test_critique_feedback_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "feedback": "Deserialized feedback",
            "suggestions": ["suggestion A", "suggestion B"],
            "confidence": 0.75,
            "category": "structure",
        }

        feedback = CritiqueFeedback(**data)

        assert feedback.feedback == "Deserialized feedback"
        assert feedback.suggestions == ["suggestion A", "suggestion B"]
        assert feedback.confidence == 0.75
        assert feedback.category == "structure"


class TestCriticResult:
    """Test CriticResult data model."""

    def test_critic_result_creation(self):
        """Test creating a valid CriticResult."""
        feedback_items = [
            CritiqueFeedback(feedback="First issue", confidence=0.8),
            CritiqueFeedback(feedback="Second issue", confidence=0.9),
        ]

        result = CriticResult(
            critic_name="test_critic",
            overall_feedback="Overall assessment",
            feedback_items=feedback_items,
            needs_improvement=True,
            confidence=0.85,
            metadata={"model": "gpt-4", "temperature": 0.7},
        )

        assert result.critic_name == "test_critic"
        assert result.overall_feedback == "Overall assessment"
        assert len(result.feedback_items) == 2
        assert result.needs_improvement is True
        assert result.confidence == 0.85
        assert result.metadata["model"] == "gpt-4"

    def test_critic_result_minimal(self):
        """Test CriticResult with minimal required fields."""
        result = CriticResult(critic_name="minimal_critic", overall_feedback="Basic feedback")

        assert result.critic_name == "minimal_critic"
        assert result.overall_feedback == "Basic feedback"
        assert result.feedback_items == []
        assert result.needs_improvement is False
        assert result.confidence is None
        assert result.metadata == {}

    def test_critic_result_empty_feedback_items(self):
        """Test CriticResult with empty feedback items."""
        result = CriticResult(
            critic_name="empty_feedback", overall_feedback="No specific feedback", feedback_items=[]
        )

        assert result.critic_name == "empty_feedback"
        assert len(result.feedback_items) == 0

    def test_critic_result_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        CriticResult(critic_name="test", overall_feedback="test", confidence=0.0)
        CriticResult(critic_name="test", overall_feedback="test", confidence=0.5)
        CriticResult(critic_name="test", overall_feedback="test", confidence=1.0)

        # Invalid confidence scores
        with pytest.raises(ValidationError):
            CriticResult(critic_name="test", overall_feedback="test", confidence=-0.1)

        with pytest.raises(ValidationError):
            CriticResult(critic_name="test", overall_feedback="test", confidence=1.1)

    def test_critic_result_string_validation(self):
        """Test string field validation."""
        # Empty strings should be allowed
        result = CriticResult(critic_name="", overall_feedback="")
        assert result.critic_name == ""
        assert result.overall_feedback == ""

        # None values should raise ValidationError
        with pytest.raises(ValidationError):
            CriticResult(critic_name=None, overall_feedback="test")

        with pytest.raises(ValidationError):
            CriticResult(critic_name="test", overall_feedback=None)

    def test_critic_result_needs_improvement_default(self):
        """Test needs_improvement default value."""
        result = CriticResult(critic_name="default_test", overall_feedback="Test feedback")

        assert result.needs_improvement is False

    def test_critic_result_metadata_handling(self):
        """Test metadata field handling."""
        # Complex metadata
        metadata = {
            "model_info": {"name": "gpt-4", "version": "2024-01"},
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
            "timing": {"start_time": "2024-01-01T00:00:00", "duration_ms": 1500},
            "nested": {"deep": {"value": 42}},
        }

        result = CriticResult(
            critic_name="metadata_test", overall_feedback="Test", metadata=metadata
        )

        assert result.metadata["model_info"]["name"] == "gpt-4"
        assert result.metadata["parameters"]["temperature"] == 0.7
        assert result.metadata["nested"]["deep"]["value"] == 42

    def test_critic_result_serialization(self):
        """Test serialization to dict."""
        feedback_items = [
            CritiqueFeedback(feedback="Issue 1", confidence=0.8),
            CritiqueFeedback(feedback="Issue 2", suggestions=["Fix this"]),
        ]

        result = CriticResult(
            critic_name="serialization_test",
            overall_feedback="Overall assessment",
            feedback_items=feedback_items,
            needs_improvement=True,
            confidence=0.9,
            metadata={"test": "value"},
        )

        data = result.model_dump()

        assert data["critic_name"] == "serialization_test"
        assert data["overall_feedback"] == "Overall assessment"
        assert len(data["feedback_items"]) == 2
        assert data["needs_improvement"] is True
        assert data["confidence"] == 0.9
        assert data["metadata"]["test"] == "value"

    def test_critic_result_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "critic_name": "deserialization_test",
            "overall_feedback": "Deserialized feedback",
            "feedback_items": [
                {"feedback": "Item 1", "confidence": 0.7},
                {"feedback": "Item 2", "suggestions": ["Suggestion A"]},
            ],
            "needs_improvement": True,
            "confidence": 0.85,
            "metadata": {"source": "test"},
        }

        result = CriticResult(**data)

        assert result.critic_name == "deserialization_test"
        assert result.overall_feedback == "Deserialized feedback"
        assert len(result.feedback_items) == 2
        assert result.feedback_items[0].feedback == "Item 1"
        assert result.feedback_items[1].suggestions == ["Suggestion A"]
        assert result.needs_improvement is True
        assert result.confidence == 0.85
        assert result.metadata["source"] == "test"

    def test_critic_result_complex_feedback_items(self):
        """Test CriticResult with complex feedback items."""
        feedback_items = [
            CritiqueFeedback(
                feedback="Complex feedback 1",
                suggestions=["Suggestion 1", "Suggestion 2", "Suggestion 3"],
                confidence=0.95,
                category="structure",
            ),
            CritiqueFeedback(
                feedback="Complex feedback 2", suggestions=[], confidence=0.6, category="content"
            ),
            CritiqueFeedback(
                feedback="Complex feedback 3", suggestions=["Single suggestion"], category="style"
            ),
        ]

        result = CriticResult(
            critic_name="complex_test",
            overall_feedback="Complex overall feedback",
            feedback_items=feedback_items,
            needs_improvement=True,
            confidence=0.8,
        )

        assert len(result.feedback_items) == 3
        assert result.feedback_items[0].category == "structure"
        assert len(result.feedback_items[0].suggestions) == 3
        assert result.feedback_items[1].category == "content"
        assert len(result.feedback_items[1].suggestions) == 0
        assert result.feedback_items[2].confidence is None

    def test_critic_result_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very long strings
        long_feedback = "x" * 10000
        result = CriticResult(critic_name="edge_test", overall_feedback=long_feedback)
        assert len(result.overall_feedback) == 10000

        # Many feedback items
        many_items = [CritiqueFeedback(feedback=f"Feedback {i}") for i in range(100)]
        result = CriticResult(
            critic_name="many_items", overall_feedback="Many items test", feedback_items=many_items
        )
        assert len(result.feedback_items) == 100

        # Extreme confidence values
        result = CriticResult(
            critic_name="extreme_confidence", overall_feedback="Test", confidence=0.0
        )
        assert result.confidence == 0.0

        result = CriticResult(
            critic_name="extreme_confidence", overall_feedback="Test", confidence=1.0
        )
        assert result.confidence == 1.0
