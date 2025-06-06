"""Comprehensive unit tests for SifakaThought and related models.

This module tests the core state container and all related Pydantic models:
- SifakaThought: Central state container with audit trail
- Generation: Text generation records
- ValidationResult: Validation outcome tracking
- CritiqueResult: Critique feedback tracking
- ToolCall: Tool execution records

Tests cover:
- Model creation and validation
- Field constraints and validation
- Immutability and state management
- Serialization and deserialization
- Audit trail functionality
- Conversation connectivity
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any
from pydantic import ValidationError

from sifaka.core.thought import (
    SifakaThought,
    Generation,
    ValidationResult,
    CritiqueResult,
    ToolCall,
)


class TestSifakaThought:
    """Test the SifakaThought model for thought state management."""

    def test_thought_creation_minimal(self):
        """Test creating a SifakaThought with minimal required fields."""
        thought = SifakaThought(prompt="Test prompt")

        assert thought.prompt == "Test prompt"
        assert thought.current_text is None
        assert thought.final_text is None
        assert thought.iteration == 0
        assert thought.max_iterations == 3  # Default value
        assert thought.generations == []
        assert thought.validations == []
        assert thought.critiques == []
        assert thought.tool_calls == []
        assert thought.techniques_applied == []
        assert thought.parent_thought_id is None
        assert thought.child_thought_ids == []
        assert thought.conversation_id is None
        assert isinstance(thought.id, str)
        assert isinstance(thought.created_at, datetime)
        assert isinstance(thought.updated_at, datetime)

    def test_thought_creation_full(self):
        """Test creating a SifakaThought with all fields."""
        thought_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        created_at = datetime.now()

        thought = SifakaThought(
            id=thought_id,
            prompt="Full test prompt",
            current_text="Current text content",
            final_text="Final text content",
            iteration=2,
            max_iterations=5,
            parent_thought_id=parent_id,
            conversation_id=conversation_id,
            created_at=created_at,
        )

        assert thought.id == thought_id
        assert thought.prompt == "Full test prompt"
        assert thought.current_text == "Current text content"
        assert thought.final_text == "Final text content"
        assert thought.iteration == 2
        assert thought.max_iterations == 5
        assert thought.parent_thought_id == parent_id
        assert thought.conversation_id == conversation_id
        assert thought.created_at == created_at

    def test_thought_add_generation(self):
        """Test adding generation results to a thought."""
        thought = SifakaThought(prompt="Test prompt")

        # Test with None result (error case)
        thought.add_generation("Generated text", "gpt-4", None)

        assert len(thought.generations) == 1
        assert thought.current_text == "Generated text"
        assert thought.generations[0].text == "Generated text"
        assert thought.generations[0].model == "gpt-4"
        assert thought.generations[0].iteration == 0
        assert thought.generations[0].cost is None
        assert thought.generations[0].usage is None
        assert thought.generations[0].conversation_history == []

    def test_thought_add_validation(self):
        """Test adding validation results to a thought."""
        thought = SifakaThought(prompt="Test prompt")

        thought.add_validation("length-validator", True, {"word_count": 150})

        assert len(thought.validations) == 1
        assert thought.validations[0].validator == "length-validator"
        assert thought.validations[0].passed is True
        assert thought.validations[0].details["word_count"] == 150
        assert thought.validations[0].iteration == 0

    def test_thought_add_critique(self):
        """Test adding critique results to a thought."""
        thought = SifakaThought(prompt="Test prompt")

        thought.add_critique(
            "constitutional-critic",
            "Needs improvement",
            ["Add examples", "Include citations"],
            confidence=0.8,
            needs_improvement=True,
        )

        assert len(thought.critiques) == 1
        assert thought.critiques[0].critic == "constitutional-critic"
        assert thought.critiques[0].feedback == "Needs improvement"
        assert len(thought.critiques[0].suggestions) == 2
        assert thought.critiques[0].confidence == 0.8
        assert thought.critiques[0].needs_improvement is True
        assert thought.critiques[0].iteration == 0
        assert "constitutional-critic" in thought.techniques_applied

    def test_thought_add_tool_call(self):
        """Test adding tool call records to a thought."""
        thought = SifakaThought(prompt="Test prompt")

        thought.add_tool_call("search-tool", {"query": "test"}, {"results": ["result1"]}, 0.5)

        assert len(thought.tool_calls) == 1
        assert thought.tool_calls[0].tool_name == "search-tool"
        assert thought.tool_calls[0].args["query"] == "test"
        assert thought.tool_calls[0].result["results"] == ["result1"]
        assert thought.tool_calls[0].execution_time == 0.5
        assert thought.tool_calls[0].iteration == 0

    def test_thought_should_continue(self):
        """Test the should_continue logic."""
        thought = SifakaThought(prompt="Test prompt", max_iterations=2)

        # Should continue initially
        assert thought.should_continue() is True

        # Should continue if validation fails
        thought.add_validation("test-validator", False, {})
        assert thought.should_continue() is True

        # Should continue if validation passes but has critique suggestions
        thought.validations.clear()
        thought.add_validation("test-validator", True, {})
        thought.add_critique("test-critic", "feedback", ["suggestion"], needs_improvement=True)
        assert thought.should_continue() is True

        # Should stop if validation passes and no critique suggestions
        thought.critiques.clear()
        thought.add_critique("test-critic", "feedback", [], needs_improvement=False)
        assert thought.should_continue() is False

        # Should stop if max iterations reached
        thought.iteration = 2
        assert thought.should_continue() is False

    def test_thought_connect_to(self):
        """Test connecting thoughts for conversation continuity."""
        parent = SifakaThought(prompt="Parent prompt")
        child = SifakaThought(prompt="Child prompt")

        child.connect_to(parent)

        assert child.parent_thought_id == parent.id
        assert child.conversation_id == parent.id  # Parent has no conversation_id
        assert parent.child_thought_ids == [child.id]

        # Test with existing conversation_id
        grandchild = SifakaThought(prompt="Grandchild prompt")
        grandchild.connect_to(child)

        assert grandchild.parent_thought_id == child.id
        assert grandchild.conversation_id == parent.id  # Inherits from parent
        assert child.child_thought_ids == [grandchild.id]

    def test_thought_validation_passed(self):
        """Test the validation_passed method."""
        thought = SifakaThought(prompt="Test prompt")

        # No validations - should return False
        assert thought.validation_passed() is False

        # All validations pass
        thought.add_validation("validator1", True, {})
        thought.add_validation("validator2", True, {})
        assert thought.validation_passed() is True

        # Some validations fail
        thought.add_validation("validator3", False, {})
        assert thought.validation_passed() is False

    def test_thought_finalize(self):
        """Test finalizing a thought."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "Final content"

        thought.finalize()

        assert thought.final_text == "Final content"

    def test_thought_get_summary(self):
        """Test getting a thought summary."""
        thought = SifakaThought(prompt="Test prompt")
        thought.current_text = "Some text"
        thought.add_generation("Generated", "gpt-4", None)
        thought.add_validation("validator", True, {})

        summary = thought.get_summary()

        assert summary["id"] == thought.id
        assert summary["iteration"] == 0
        assert summary["has_text"] is True
        assert summary["text_length"] == 9
        assert summary["generations_count"] == 1
        assert summary["validations_count"] == 1
        assert summary["is_finalized"] is False

    def test_thought_serialization(self):
        """Test thought serialization and deserialization."""
        thought = SifakaThought(prompt="Test serialization")
        thought.add_generation("Generated text", "gpt-4", None)
        thought.add_validation("validator", True, {"score": 0.9})

        # Serialize
        data = thought.model_dump()
        assert data["prompt"] == "Test serialization"
        assert len(data["generations"]) == 1
        assert len(data["validations"]) == 1

        # Deserialize
        restored = SifakaThought.model_validate(data)
        assert restored.prompt == thought.prompt
        assert len(restored.generations) == 1
        assert len(restored.validations) == 1
        assert restored.generations[0].text == "Generated text"


class TestGeneration:
    """Test the Generation model for text generation tracking."""

    def test_generation_creation_minimal(self):
        """Test creating a Generation with minimal required fields."""
        generation = Generation(
            iteration=0,
            text="Generated text content",
            model="openai:gpt-4",
            timestamp=datetime.now(),
        )

        assert generation.iteration == 0
        assert generation.text == "Generated text content"
        assert generation.model == "openai:gpt-4"
        assert generation.conversation_history == []
        assert generation.cost is None
        assert generation.usage is None
        assert isinstance(generation.timestamp, datetime)

    def test_generation_creation_full(self):
        """Test creating a Generation with all fields."""
        timestamp = datetime.now()
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        generation = Generation(
            iteration=1,
            text="Generated text content",
            model="openai:gpt-4",
            timestamp=timestamp,
            conversation_history=conversation_history,
            cost=0.05,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        assert generation.iteration == 1
        assert generation.text == "Generated text content"
        assert generation.model == "openai:gpt-4"
        assert generation.conversation_history == conversation_history
        assert generation.cost == 0.05
        assert generation.usage["total_tokens"] == 30
        assert generation.timestamp == timestamp


class TestValidationResult:
    """Test the ValidationResult model for validation tracking."""

    def test_validation_result_creation_minimal(self):
        """Test creating a ValidationResult with minimal required fields."""
        result = ValidationResult(
            iteration=0,
            validator="test-validator",
            passed=True,
            details={},
            timestamp=datetime.now(),
        )

        assert result.iteration == 0
        assert result.validator == "test-validator"
        assert result.passed is True
        assert result.details == {}
        assert isinstance(result.timestamp, datetime)

    def test_validation_result_creation_full(self):
        """Test creating a ValidationResult with all fields."""
        timestamp = datetime.now()

        result = ValidationResult(
            iteration=1,
            validator="content-validator",
            passed=False,
            details={"length": 150, "min_required": 200},
            timestamp=timestamp,
        )

        assert result.iteration == 1
        assert result.validator == "content-validator"
        assert result.passed is False
        assert result.details["length"] == 150
        assert result.details["min_required"] == 200
        assert result.timestamp == timestamp


class TestCritiqueResult:
    """Test the CritiqueResult model for critique tracking."""

    def test_critique_result_creation_minimal(self):
        """Test creating a CritiqueResult with minimal required fields."""
        result = CritiqueResult(
            iteration=0,
            critic="test-critic",
            feedback="Test feedback",
            suggestions=[],
            timestamp=datetime.now(),
        )

        assert result.iteration == 0
        assert result.critic == "test-critic"
        assert result.feedback == "Test feedback"
        assert result.suggestions == []
        assert result.needs_improvement is True  # Default value
        assert result.confidence is None
        assert isinstance(result.timestamp, datetime)

    def test_critique_result_creation_full(self):
        """Test creating a CritiqueResult with all fields."""
        timestamp = datetime.now()

        result = CritiqueResult(
            iteration=1,
            critic="constitutional-critic",
            feedback="The text violates constitutional principles",
            suggestions=["Add more balanced perspective", "Include counterarguments"],
            timestamp=timestamp,
            confidence=0.85,
            reasoning="Based on constitutional analysis",
            needs_improvement=True,
            critic_metadata={"version": "1.0"},
            processing_time_ms=200.5,
            model_name="gpt-3.5-turbo",
            paper_reference="Constitutional AI paper",
            methodology="Constitutional critique",
            tools_used=["search"],
            retrieval_context={"sources": ["constitution"]},
        )

        assert result.iteration == 1
        assert result.critic == "constitutional-critic"
        assert result.feedback == "The text violates constitutional principles"
        assert len(result.suggestions) == 2
        assert "Add more balanced perspective" in result.suggestions
        assert "Include counterarguments" in result.suggestions
        assert result.confidence == 0.85
        assert result.reasoning == "Based on constitutional analysis"
        assert result.needs_improvement is True
        assert result.critic_metadata["version"] == "1.0"
        assert result.processing_time_ms == 200.5
        assert result.model_name == "gpt-3.5-turbo"
        assert result.paper_reference == "Constitutional AI paper"
        assert result.methodology == "Constitutional critique"
        assert result.tools_used == ["search"]
        assert result.retrieval_context["sources"] == ["constitution"]
        assert result.timestamp == timestamp


class TestToolCall:
    """Test the ToolCall model for tool execution tracking."""

    def test_tool_call_creation_minimal(self):
        """Test creating a ToolCall with minimal required fields."""
        tool_call = ToolCall(
            iteration=0,
            tool_name="test-tool",
            args={},
            result=None,
            execution_time=0.5,
            timestamp=datetime.now(),
        )

        assert tool_call.iteration == 0
        assert tool_call.tool_name == "test-tool"
        assert tool_call.args == {}
        assert tool_call.result is None
        assert tool_call.execution_time == 0.5
        assert isinstance(tool_call.timestamp, datetime)

    def test_tool_call_creation_full(self):
        """Test creating a ToolCall with all fields."""
        timestamp = datetime.now()

        tool_call = ToolCall(
            iteration=1,
            tool_name="search-tool",
            args={"query": "renewable energy", "limit": 10},
            result={"results": ["result1", "result2"], "count": 2},
            execution_time=1.25,
            timestamp=timestamp,
        )

        assert tool_call.iteration == 1
        assert tool_call.tool_name == "search-tool"
        assert tool_call.args["query"] == "renewable energy"
        assert tool_call.args["limit"] == 10
        assert tool_call.result["count"] == 2
        assert len(tool_call.result["results"]) == 2
        assert tool_call.execution_time == 1.25
        assert tool_call.timestamp == timestamp


class TestModelIntegration:
    """Test integration between different thought-related models."""

    def test_complete_thought_workflow(self):
        """Test a complete thought with all audit trail components."""
        # Create a thought
        thought = SifakaThought(
            prompt="Test complete workflow",
            max_iterations=2,
        )

        # Add generation
        thought.add_generation("Initial generated text", "gpt-4", None)

        # Add validation results
        thought.add_validation("length-validator", True, {"word_count": 150})
        thought.add_validation("content-validator", False, {"issue": "lacks examples"})

        # Add critique result
        thought.add_critique(
            "constitutional-critic",
            "Add more balanced perspective",
            ["Include counterarguments", "Add citations"],
            confidence=0.8,
            needs_improvement=True,
        )

        # Add tool call
        thought.add_tool_call(
            "search-tool", {"query": "examples"}, {"results": ["example1", "example2"]}, 0.5
        )

        # Update iteration and add improved generation
        thought.iteration = 1
        thought.add_generation("Improved text with examples and citations", "gpt-4", None)
        thought.finalize()

        # Verify the complete audit trail
        assert thought.prompt == "Test complete workflow"
        assert thought.iteration == 1
        assert thought.current_text == "Improved text with examples and citations"
        assert thought.final_text == "Improved text with examples and citations"
        assert len(thought.generations) == 2
        assert len(thought.validations) == 2
        assert len(thought.critiques) == 1
        assert len(thought.tool_calls) == 1

        # Verify generation progression
        assert thought.generations[0].text == "Initial generated text"
        assert thought.generations[1].text == "Improved text with examples and citations"

        # Verify validation results
        assert thought.validations[0].passed is True
        assert thought.validations[1].passed is False

        # Verify critique feedback
        assert thought.critiques[0].needs_improvement is True
        assert len(thought.critiques[0].suggestions) == 2

        # Verify tool usage
        assert thought.tool_calls[0].tool_name == "search-tool"
        assert thought.tool_calls[0].result["results"] == ["example1", "example2"]
