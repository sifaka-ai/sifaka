"""Simple tests for models module to improve coverage."""

from datetime import datetime

from sifaka.core.models import (
    CritiqueResult,
    Generation,
    SifakaResult,
    ToolUsage,
    ValidationResult,
)


class TestModels:
    """Test model classes."""

    def test_generation(self):
        """Test Generation model."""
        gen = Generation(
            text="Generated text",
            model="gpt-4",
            iteration=1,
            prompt="Improve this",
            tokens_used=100,
            processing_time=1.5,
        )

        assert gen.text == "Generated text"
        assert gen.model == "gpt-4"
        assert gen.iteration == 1
        assert gen.tokens_used == 100
        assert isinstance(gen.timestamp, datetime)

    def test_tool_usage(self):
        """Test ToolUsage model."""
        tool = ToolUsage(
            tool_name="search",
            status="success",
            input_data="query text",
            result_count=5,
            processing_time=0.5,
        )

        assert tool.tool_name == "search"
        assert tool.status == "success"
        assert tool.result_count == 5

        # Test failure case
        tool = ToolUsage(
            tool_name="api", status="failure", error_message="Connection timeout"
        )
        assert tool.status == "failure"
        assert tool.error_message == "Connection timeout"

    def test_validation_result(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            validator="length", passed=True, score=0.9, details="Good length"
        )

        assert result.validator == "length"
        assert result.passed is True
        assert result.score == 0.9
        assert isinstance(result.timestamp, datetime)

    def test_critique_result(self):
        """Test CritiqueResult model."""
        critique = CritiqueResult(
            critic="style",
            feedback="Needs improvement",
            suggestions=["Use active voice", "Add examples"],
            needs_improvement=True,
            confidence=0.8,
            model_used="gpt-4",
            temperature_used=0.7,
            tokens_used=200,
            processing_time=2.0,
        )

        assert critique.critic == "style"
        assert len(critique.suggestions) == 2
        assert critique.confidence == 0.8
        assert critique.model_used == "gpt-4"

        # Test with tools
        critique.tools_used.append(
            ToolUsage(tool_name="grammar_check", status="success")
        )
        assert len(critique.tools_used) == 1

    def test_sifaka_result(self):
        """Test SifakaResult model."""
        result = SifakaResult(
            original_text="Original",
            final_text="Improved",
            iteration=2,
            processing_time=5.0,
        )

        assert result.original_text == "Original"
        assert result.final_text == "Improved"
        assert result.iteration == 2
        assert len(result.id) > 0  # Auto-generated

        # Test current_text property
        assert result.current_text == "Original"  # No generations yet

        # Add generation
        result.add_generation("Version 1", "gpt-4", tokens=50)
        assert result.current_text == "Version 1"
        assert len(result.generations) == 1

        # Test needs_improvement property
        assert result.needs_improvement is True  # No critiques

        # Add critique
        result.add_critique(
            critic="test", feedback="Good", suggestions=[], needs_improvement=False
        )
        assert result.needs_improvement is False

        # Test all_passed property
        assert result.all_passed is False  # No validations

        result.add_validation("test", True, 1.0, "Perfect")
        assert result.all_passed is True

        # Test increment_iteration
        result.increment_iteration()
        assert result.iteration == 3

        # Test set_final_text
        result.set_final_text("Final version")
        assert result.final_text == "Final version"

        # Test get_quality_progression
        progression = result.get_quality_progression()
        assert "text_length_progression" in progression
        assert "word_count_progression" in progression

    def test_sifaka_result_deque_limits(self):
        """Test memory bounds on collections."""
        result = SifakaResult(original_text="Test", final_text="Test", iteration=1)

        # Add more than maxlen
        for i in range(15):
            result.generations.append(
                Generation(text=f"Gen {i}", model="gpt-4", iteration=i)
            )

        # Should only keep last 10
        assert len(result.generations) == 10

        # Similarly for critiques (maxlen=20)
        for i in range(25):
            result.critiques.append(
                CritiqueResult(
                    critic=f"critic_{i}",
                    feedback="Test",
                    suggestions=[],
                    needs_improvement=False,
                )
            )

        assert len(result.critiques) == 20

    def test_model_serialization(self):
        """Test model serialization."""
        result = SifakaResult(original_text="Test", final_text="Improved", iteration=1)

        # Should be able to dump to dict
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["original_text"] == "Test"
        assert data["final_text"] == "Improved"

        # Should be able to dump to JSON
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test" in json_str
        assert "Improved" in json_str
