"""Comprehensive unit tests for critic result Pydantic models.

This module tests all the new Pydantic models used in the critic architecture:
- SeverityLevel enum
- ConfidenceScore model with validation
- ViolationReport model with location tracking
- ImprovementSuggestion model with dependencies
- CritiqueFeedback model with structured components
- CriticResult model with consistency validation

Tests cover:
- Valid model creation and serialization
- Field validation and constraints
- Edge cases and error conditions
- Model consistency validation
- Frozen model behavior
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from pydantic import ValidationError

from sifaka.models.critic_results import (
    SeverityLevel,
    ConfidenceScore,
    ViolationReport,
    ImprovementSuggestion,
    CritiqueFeedback,
    CriticResult,
)


class TestSeverityLevel:
    """Test the SeverityLevel enum."""

    def test_severity_level_values(self):
        """Test that all severity levels have correct values."""
        assert SeverityLevel.CRITICAL == "critical"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.INFO == "info"

    def test_severity_level_ordering(self):
        """Test that severity levels can be compared."""
        # Note: String enums compare lexicographically, not by severity
        # This is expected behavior for this enum
        levels = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO,
        ]
        assert len(levels) == 5
        assert all(isinstance(level, SeverityLevel) for level in levels)


class TestConfidenceScore:
    """Test the ConfidenceScore model."""

    def test_valid_confidence_score_creation(self):
        """Test creating a valid ConfidenceScore."""
        score = ConfidenceScore(
            overall=0.85,
            content_quality=0.9,
            grammar_accuracy=0.8,
            factual_accuracy=0.7,
            coherence=0.95,
            calculation_method="weighted_average",
            factors_considered=["grammar", "content", "coherence"],
            uncertainty_sources=["ambiguous_context"],
            metadata={"model_version": "v1.0"},
        )

        assert score.overall == 0.85
        assert score.content_quality == 0.9
        assert score.grammar_accuracy == 0.8
        assert score.factual_accuracy == 0.7
        assert score.coherence == 0.95
        assert score.calculation_method == "weighted_average"
        assert "grammar" in score.factors_considered
        assert "ambiguous_context" in score.uncertainty_sources
        assert score.metadata["model_version"] == "v1.0"

    def test_confidence_score_minimal(self):
        """Test creating ConfidenceScore with only required fields."""
        score = ConfidenceScore(overall=0.5)

        assert score.overall == 0.5
        assert score.content_quality is None
        assert score.grammar_accuracy is None
        assert score.factual_accuracy is None
        assert score.coherence is None
        assert score.calculation_method is None
        assert score.factors_considered == []
        assert score.uncertainty_sources == []
        assert score.metadata == {}

    def test_confidence_score_validation_bounds(self):
        """Test that confidence scores are properly bounded between 0.0 and 1.0."""
        # Valid bounds
        ConfidenceScore(overall=0.0)
        ConfidenceScore(overall=1.0)
        ConfidenceScore(overall=0.5, content_quality=0.0, grammar_accuracy=1.0)

        # Invalid bounds - should raise ValidationError
        with pytest.raises(ValidationError):
            ConfidenceScore(overall=-0.1)

        with pytest.raises(ValidationError):
            ConfidenceScore(overall=1.1)

        with pytest.raises(ValidationError):
            ConfidenceScore(overall=0.5, content_quality=-0.1)

        with pytest.raises(ValidationError):
            ConfidenceScore(overall=0.5, grammar_accuracy=1.1)

    def test_confidence_score_frozen(self):
        """Test that ConfidenceScore is frozen (immutable)."""
        score = ConfidenceScore(overall=0.8)

        with pytest.raises(ValidationError):
            score.overall = 0.9


class TestViolationReport:
    """Test the ViolationReport model."""

    def test_valid_violation_report_creation(self):
        """Test creating a valid ViolationReport."""
        violation = ViolationReport(
            violation_type="grammar_error",
            description="Subject-verb disagreement",
            severity=SeverityLevel.HIGH,
            location="paragraph 2, sentence 3",
            start_position=45,
            end_position=52,
            rule_violated="subject_verb_agreement",
            evidence="The cats was sleeping",
            suggested_fix="Change 'was' to 'were'",
            confidence=0.95,
            metadata={"rule_id": "SVA001"},
        )

        assert violation.violation_type == "grammar_error"
        assert violation.description == "Subject-verb disagreement"
        assert violation.severity == SeverityLevel.HIGH
        assert violation.location == "paragraph 2, sentence 3"
        assert violation.start_position == 45
        assert violation.end_position == 52
        assert violation.rule_violated == "subject_verb_agreement"
        assert violation.evidence == "The cats was sleeping"
        assert violation.suggested_fix == "Change 'was' to 'were'"
        assert violation.confidence == 0.95
        assert violation.metadata["rule_id"] == "SVA001"

    def test_violation_report_minimal(self):
        """Test creating ViolationReport with only required fields."""
        violation = ViolationReport(
            violation_type="style_issue",
            description="Passive voice detected",
            severity=SeverityLevel.LOW,
        )

        assert violation.violation_type == "style_issue"
        assert violation.description == "Passive voice detected"
        assert violation.severity == SeverityLevel.LOW
        assert violation.location is None
        assert violation.start_position is None
        assert violation.end_position is None
        assert violation.rule_violated is None
        assert violation.evidence is None
        assert violation.suggested_fix is None
        assert violation.confidence == 1.0  # Default value
        assert violation.metadata == {}

    def test_violation_report_position_validation(self):
        """Test that position fields are properly validated."""
        # Valid positions
        ViolationReport(
            violation_type="test",
            description="test",
            severity=SeverityLevel.LOW,
            start_position=0,
            end_position=10,
        )

        # Invalid positions - should raise ValidationError
        with pytest.raises(ValidationError):
            ViolationReport(
                violation_type="test",
                description="test",
                severity=SeverityLevel.LOW,
                start_position=-1,
            )

        with pytest.raises(ValidationError):
            ViolationReport(
                violation_type="test",
                description="test",
                severity=SeverityLevel.LOW,
                end_position=-5,
            )

    def test_violation_report_confidence_validation(self):
        """Test that confidence is properly bounded."""
        # Valid confidence
        ViolationReport(
            violation_type="test", description="test", severity=SeverityLevel.LOW, confidence=0.0
        )

        ViolationReport(
            violation_type="test", description="test", severity=SeverityLevel.LOW, confidence=1.0
        )

        # Invalid confidence
        with pytest.raises(ValidationError):
            ViolationReport(
                violation_type="test",
                description="test",
                severity=SeverityLevel.LOW,
                confidence=-0.1,
            )

        with pytest.raises(ValidationError):
            ViolationReport(
                violation_type="test",
                description="test",
                severity=SeverityLevel.LOW,
                confidence=1.1,
            )

    def test_violation_report_frozen(self):
        """Test that ViolationReport is frozen (immutable)."""
        violation = ViolationReport(
            violation_type="test", description="test", severity=SeverityLevel.LOW
        )

        with pytest.raises(ValidationError):
            violation.violation_type = "changed"


class TestImprovementSuggestion:
    """Test the ImprovementSuggestion model."""

    def test_valid_improvement_suggestion_creation(self):
        """Test creating a valid ImprovementSuggestion."""
        suggestion = ImprovementSuggestion(
            suggestion="Add more specific examples to support your argument",
            category="clarity",
            priority=SeverityLevel.HIGH,
            rationale="Examples help readers understand abstract concepts",
            implementation="Insert 2-3 concrete examples after the main claim",
            example="For instance, when discussing AI safety, mention specific scenarios like autonomous vehicles or medical diagnosis systems",
            applies_to="paragraph 3",
            start_position=120,
            end_position=180,
            expected_impact="Improved reader comprehension and engagement",
            confidence=0.85,
            depends_on=["fix_grammar_first"],
            conflicts_with=["remove_examples"],
            metadata={"suggestion_id": "CLAR001"},
        )

        assert suggestion.suggestion == "Add more specific examples to support your argument"
        assert suggestion.category == "clarity"
        assert suggestion.priority == SeverityLevel.HIGH
        assert suggestion.rationale == "Examples help readers understand abstract concepts"
        assert suggestion.implementation == "Insert 2-3 concrete examples after the main claim"
        assert "For instance" in suggestion.example
        assert suggestion.applies_to == "paragraph 3"
        assert suggestion.start_position == 120
        assert suggestion.end_position == 180
        assert suggestion.expected_impact == "Improved reader comprehension and engagement"
        assert suggestion.confidence == 0.85
        assert "fix_grammar_first" in suggestion.depends_on
        assert "remove_examples" in suggestion.conflicts_with
        assert suggestion.metadata["suggestion_id"] == "CLAR001"

    def test_improvement_suggestion_minimal(self):
        """Test creating ImprovementSuggestion with only required fields."""
        suggestion = ImprovementSuggestion(suggestion="Fix typo", category="grammar")

        assert suggestion.suggestion == "Fix typo"
        assert suggestion.category == "grammar"
        assert suggestion.priority == SeverityLevel.MEDIUM  # Default value
        assert suggestion.rationale is None
        assert suggestion.implementation is None
        assert suggestion.example is None
        assert suggestion.applies_to is None
        assert suggestion.start_position is None
        assert suggestion.end_position is None
        assert suggestion.expected_impact is None
        assert suggestion.confidence == 1.0  # Default value
        assert suggestion.depends_on == []
        assert suggestion.conflicts_with == []
        assert suggestion.metadata == {}

    def test_improvement_suggestion_position_validation(self):
        """Test that position fields are properly validated."""
        # Valid positions
        ImprovementSuggestion(suggestion="test", category="test", start_position=0, end_position=10)

        # Invalid positions
        with pytest.raises(ValidationError):
            ImprovementSuggestion(suggestion="test", category="test", start_position=-1)

        with pytest.raises(ValidationError):
            ImprovementSuggestion(suggestion="test", category="test", end_position=-5)

    def test_improvement_suggestion_confidence_validation(self):
        """Test that confidence is properly bounded."""
        # Valid confidence
        ImprovementSuggestion(suggestion="test", category="test", confidence=0.0)

        ImprovementSuggestion(suggestion="test", category="test", confidence=1.0)

        # Invalid confidence
        with pytest.raises(ValidationError):
            ImprovementSuggestion(suggestion="test", category="test", confidence=-0.1)

        with pytest.raises(ValidationError):
            ImprovementSuggestion(suggestion="test", category="test", confidence=1.1)

    def test_improvement_suggestion_frozen(self):
        """Test that ImprovementSuggestion is frozen (immutable)."""
        suggestion = ImprovementSuggestion(suggestion="test", category="test")

        with pytest.raises(ValidationError):
            suggestion.suggestion = "changed"


class TestCritiqueFeedback:
    """Test the CritiqueFeedback model."""

    def test_valid_critique_feedback_creation(self):
        """Test creating a valid CritiqueFeedback."""
        violation = ViolationReport(
            violation_type="grammar", description="Grammar error", severity=SeverityLevel.HIGH
        )

        suggestion = ImprovementSuggestion(suggestion="Fix grammar", category="grammar")

        confidence = ConfidenceScore(overall=0.9)

        feedback = CritiqueFeedback(
            message="The text has several grammar issues that need attention",
            needs_improvement=True,
            violations=[violation],
            suggestions=[suggestion],
            confidence=confidence,
            critic_name="TestCritic",
            metadata={"critic_type": "constitutional"},
        )

        assert feedback.message == "The text has several grammar issues that need attention"
        assert feedback.needs_improvement is True
        assert len(feedback.violations) == 1
        assert feedback.violations[0].violation_type == "grammar"
        assert len(feedback.suggestions) == 1
        assert feedback.suggestions[0].suggestion == "Fix grammar"
        assert feedback.confidence.overall == 0.9
        assert feedback.critic_name == "TestCritic"
        assert feedback.metadata["critic_type"] == "constitutional"

    def test_critique_feedback_minimal(self):
        """Test creating CritiqueFeedback with only required fields."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="Text looks good",
            needs_improvement=False,
            confidence=confidence,
            critic_name="TestCritic",
        )

        assert feedback.message == "Text looks good"
        assert feedback.needs_improvement is False
        assert feedback.violations == []
        assert feedback.suggestions == []
        assert feedback.confidence.overall == 0.8
        assert feedback.critic_name == "TestCritic"
        assert feedback.metadata == {}

    def test_critique_feedback_frozen(self):
        """Test that CritiqueFeedback is frozen (immutable)."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="test", needs_improvement=False, confidence=confidence, critic_name="TestCritic"
        )

        with pytest.raises(ValidationError):
            feedback.message = "changed"


class TestCriticResult:
    """Test the CriticResult model."""

    def test_valid_critic_result_creation(self):
        """Test creating a valid CriticResult."""
        confidence = ConfidenceScore(overall=0.9)
        feedback = CritiqueFeedback(
            message="Text needs improvement",
            needs_improvement=True,
            confidence=confidence,
            critic_name="ConstitutionalCritic",
        )

        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            success=True,
            total_processing_time_ms=150.5,
            input_text_length=100,
            metadata={"version": "v1.0"},
        )

        assert result.feedback.message == "Text needs improvement"
        assert result.feedback.needs_improvement is True
        assert result.operation_type == "critique"
        assert result.success is True
        assert result.total_processing_time_ms == 150.5
        assert result.input_text_length == 100
        assert result.error_message is None
        assert result.metadata["version"] == "v1.0"
        assert isinstance(result.created_at, datetime)

    def test_critic_result_minimal(self):
        """Test creating CriticResult with only required fields."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="OK", needs_improvement=False, confidence=confidence, critic_name="TestCritic"
        )

        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            total_processing_time_ms=50.0,
            input_text_length=20,
        )

        assert result.feedback.message == "OK"
        assert result.operation_type == "critique"
        assert result.success is True  # Default value
        assert result.total_processing_time_ms == 50.0
        assert result.input_text_length == 20
        assert result.error_message is None
        assert result.metadata == {}
        assert isinstance(result.created_at, datetime)

    def test_critic_result_consistency_validation_success(self):
        """Test that successful results pass consistency validation."""
        # Successful result with improvement needed - should be valid
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="Needs work",
            needs_improvement=True,
            confidence=confidence,
            critic_name="TestCritic",
        )

        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            success=True,
            total_processing_time_ms=100.0,
            input_text_length=50,
        )

        # Should not raise any validation errors
        assert result.success is True
        assert result.feedback.needs_improvement is True

    def test_critic_result_consistency_validation_failure(self):
        """Test that failed results are properly validated for consistency."""
        # Failed result without improvement indication - should be invalid
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="All good",
            needs_improvement=False,
            confidence=confidence,
            critic_name="TestCritic",
        )

        with pytest.raises(
            ValidationError, match="Failed operations should indicate improvement is needed"
        ):
            CriticResult(
                feedback=feedback,
                operation_type="critique",
                success=False,
                error_message="Something went wrong",
                total_processing_time_ms=100.0,
                input_text_length=50,
            )

    def test_critic_result_error_message_validation(self):
        """Test that error messages are required for failed operations."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="Needs work",
            needs_improvement=True,
            confidence=confidence,
            critic_name="TestCritic",
        )

        # Failed result without error message - should be invalid
        with pytest.raises(ValidationError, match="Error message is required when success=False"):
            CriticResult(
                feedback=feedback,
                operation_type="critique",
                success=False,
                total_processing_time_ms=100.0,
                input_text_length=50,
            )

        # Failed result with error message - should be valid
        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            success=False,
            error_message="Processing failed",
            total_processing_time_ms=100.0,
            input_text_length=50,
        )

        assert result.success is False
        assert result.error_message == "Processing failed"

    def test_critic_result_processing_time_validation(self):
        """Test that processing time is properly validated."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="test", needs_improvement=False, confidence=confidence, critic_name="TestCritic"
        )

        # Valid processing times
        CriticResult(
            feedback=feedback,
            operation_type="critique",
            total_processing_time_ms=0.0,
            input_text_length=10,
        )

        CriticResult(
            feedback=feedback,
            operation_type="critique",
            total_processing_time_ms=1000.5,
            input_text_length=10,
        )

        # Invalid processing time
        with pytest.raises(ValidationError):
            CriticResult(
                feedback=feedback,
                operation_type="critique",
                total_processing_time_ms=-1.0,
                input_text_length=10,
            )

    def test_critic_result_frozen(self):
        """Test that CriticResult is frozen (immutable)."""
        confidence = ConfidenceScore(overall=0.8)
        feedback = CritiqueFeedback(
            message="test", needs_improvement=False, confidence=confidence, critic_name="TestCritic"
        )

        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            total_processing_time_ms=100.0,
            input_text_length=10,
        )

        with pytest.raises(ValidationError):
            result.success = False


class TestModelIntegration:
    """Test integration between different models."""

    def test_complex_critic_result_with_all_components(self):
        """Test creating a complex CriticResult with all nested components."""
        # Create confidence score
        confidence = ConfidenceScore(
            overall=0.85,
            content_quality=0.9,
            grammar_accuracy=0.8,
            calculation_method="weighted_average",
            factors_considered=["grammar", "content", "style"],
            uncertainty_sources=["ambiguous_context"],
        )

        # Create violation reports
        violation1 = ViolationReport(
            violation_type="grammar_error",
            description="Subject-verb disagreement",
            severity=SeverityLevel.HIGH,
            location="paragraph 1",
            start_position=10,
            end_position=20,
            confidence=0.95,
        )

        violation2 = ViolationReport(
            violation_type="style_issue",
            description="Passive voice overuse",
            severity=SeverityLevel.MEDIUM,
            location="paragraph 2",
            confidence=0.7,
        )

        # Create improvement suggestions
        suggestion1 = ImprovementSuggestion(
            suggestion="Fix subject-verb agreement",
            category="grammar",
            priority=SeverityLevel.HIGH,
            rationale="Grammatical correctness is essential",
            confidence=0.95,
        )

        suggestion2 = ImprovementSuggestion(
            suggestion="Use more active voice",
            category="style",
            priority=SeverityLevel.MEDIUM,
            rationale="Active voice is more engaging",
            confidence=0.8,
            depends_on=["fix_grammar_first"],
        )

        # Create critique feedback
        feedback = CritiqueFeedback(
            message="The text has grammar and style issues that should be addressed",
            needs_improvement=True,
            violations=[violation1, violation2],
            suggestions=[suggestion1, suggestion2],
            confidence=confidence,
            critic_name="ConstitutionalCritic",
        )

        # Create critic result
        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            success=True,
            total_processing_time_ms=245.7,
            input_text_length=150,
            metadata={"version": "v1.0", "rules_applied": ["grammar", "style"]},
        )

        # Verify the complete structure
        assert result.success is True
        assert result.feedback.needs_improvement is True
        assert len(result.feedback.violations) == 2
        assert len(result.feedback.suggestions) == 2
        assert result.feedback.confidence.overall == 0.85
        assert result.feedback.violations[0].severity == SeverityLevel.HIGH
        assert result.feedback.violations[1].severity == SeverityLevel.MEDIUM
        assert result.feedback.suggestions[0].priority == SeverityLevel.HIGH
        assert result.feedback.suggestions[1].priority == SeverityLevel.MEDIUM
        assert "fix_grammar_first" in result.feedback.suggestions[1].depends_on
        assert result.total_processing_time_ms == 245.7
        assert result.input_text_length == 150
        assert result.feedback.critic_name == "ConstitutionalCritic"

    def test_serialization_and_deserialization(self):
        """Test that models can be properly serialized and deserialized."""
        # Create a complex result
        confidence = ConfidenceScore(overall=0.8, content_quality=0.9)
        violation = ViolationReport(
            violation_type="test", description="test violation", severity=SeverityLevel.HIGH
        )
        suggestion = ImprovementSuggestion(suggestion="test suggestion", category="test")
        feedback = CritiqueFeedback(
            message="test feedback",
            needs_improvement=True,
            violations=[violation],
            suggestions=[suggestion],
            confidence=confidence,
            critic_name="TestCritic",
        )
        result = CriticResult(
            feedback=feedback,
            operation_type="critique",
            success=True,
            total_processing_time_ms=100.0,
            input_text_length=50,
        )

        # Serialize to dict
        result_dict = result.model_dump()

        # Verify structure
        assert result_dict["feedback"]["message"] == "test feedback"
        assert result_dict["feedback"]["needs_improvement"] is True
        assert len(result_dict["feedback"]["violations"]) == 1
        assert len(result_dict["feedback"]["suggestions"]) == 1
        assert result_dict["feedback"]["confidence"]["overall"] == 0.8
        assert result_dict["operation_type"] == "critique"
        assert result_dict["success"] is True
        assert result_dict["total_processing_time_ms"] == 100.0
        assert result_dict["input_text_length"] == 50

        # Deserialize back to model
        result_restored = CriticResult.model_validate(result_dict)

        # Verify restored model
        assert result_restored.feedback.message == "test feedback"
        assert result_restored.feedback.needs_improvement is True
        assert len(result_restored.feedback.violations) == 1
        assert len(result_restored.feedback.suggestions) == 1
        assert result_restored.feedback.confidence.overall == 0.8
        assert result_restored.operation_type == "critique"
        assert result_restored.success is True
        assert result_restored.total_processing_time_ms == 100.0
        assert result_restored.input_text_length == 50
