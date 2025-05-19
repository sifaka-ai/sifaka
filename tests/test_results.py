"""
Tests for the results module.

This module contains tests for the result classes in the Sifaka framework.
"""

from sifaka.results import ImprovementResult, Result, ValidationResult


class TestValidationResult:
    """Tests for the ValidationResult class."""

    def test_init_with_required_fields(self) -> None:
        """Test initializing a ValidationResult with only required fields."""
        result = ValidationResult(passed=True, message="Validation passed")
        assert result.passed is True
        assert result.message == "Validation passed"
        assert result.score is None
        assert result.issues is None
        assert result.suggestions is None
        assert result.details == {}

    def test_init_with_all_fields(self) -> None:
        """Test initializing a ValidationResult with all fields."""
        result = ValidationResult(
            passed=False,
            message="Validation failed",
            score=0.5,
            issues=["Issue 1", "Issue 2"],
            suggestions=["Suggestion 1", "Suggestion 2"],
            _details={"key": "value"},
        )
        assert result.passed is False
        assert result.message == "Validation failed"
        assert result.score == 0.5
        assert result.issues == ["Issue 1", "Issue 2"]
        assert result.suggestions == ["Suggestion 1", "Suggestion 2"]
        assert result.details == {"key": "value"}

    def test_bool_conversion(self) -> None:
        """Test converting ValidationResult to boolean."""
        result_true = ValidationResult(passed=True, message="Validation passed")
        result_false = ValidationResult(passed=False, message="Validation failed")

        assert bool(result_true) is True
        assert bool(result_false) is False

        # Test in if statement
        if result_true:
            assert True
        else:
            assert False

        if result_false:
            assert False
        else:
            assert True

    def test_details_property(self) -> None:
        """Test the details property."""
        result = ValidationResult(
            passed=True, message="Validation passed", _details={"key": "value"}
        )
        assert result.details == {"key": "value"}

        # Note: In the current implementation, details returns a reference, not a copy
        # This test just verifies that the details property works correctly
        assert result.details is result._details


class TestImprovementResult:
    """Tests for the ImprovementResult class."""

    def test_init_with_required_fields(self) -> None:
        """Test initializing an ImprovementResult with only required fields."""
        result = ImprovementResult(
            _original_text="Original text",
            _improved_text="Improved text",
            _changes_made=True,
            message="Text improved",
        )
        assert result.original_text == "Original text"
        assert result.improved_text == "Improved text"
        assert result.changes_made is True
        assert result.message == "Text improved"
        assert result.processing_time_ms is None
        assert result.details == {}

    def test_init_with_all_fields(self) -> None:
        """Test initializing an ImprovementResult with all fields."""
        result = ImprovementResult(
            _original_text="Original text",
            _improved_text="Improved text",
            _changes_made=True,
            message="Text improved",
            processing_time_ms=123.45,
            _details={"key": "value"},
        )
        assert result.original_text == "Original text"
        assert result.improved_text == "Improved text"
        assert result.changes_made is True
        assert result.message == "Text improved"
        assert result.processing_time_ms == 123.45
        assert result.details == {"key": "value"}

    def test_bool_conversion(self) -> None:
        """Test converting ImprovementResult to boolean."""
        result_true = ImprovementResult(
            _original_text="Original text",
            _improved_text="Improved text",
            _changes_made=True,
            message="Text improved",
        )
        result_false = ImprovementResult(
            _original_text="Original text",
            _improved_text="Original text",
            _changes_made=False,
            message="No improvements needed",
        )

        assert bool(result_true) is True
        assert bool(result_false) is False

        # Test in if statement
        if result_true:
            assert True
        else:
            assert False

        if result_false:
            assert False
        else:
            assert True

    def test_passed_property(self) -> None:
        """Test the passed property."""
        result = ImprovementResult(
            _original_text="Original text",
            _improved_text="Improved text",
            _changes_made=True,
            message="Text improved",
        )
        assert result.passed is True  # Always True for ImprovementResult

    def test_properties(self) -> None:
        """Test the properties of ImprovementResult."""
        result = ImprovementResult(
            _original_text="Original text",
            _improved_text="Improved text",
            _changes_made=True,
            message="Text improved",
            _details={"key": "value"},
        )
        assert result.original_text == "Original text"
        assert result.improved_text == "Improved text"
        assert result.changes_made is True
        assert result.details == {"key": "value"}

        # Note: In the current implementation, details returns a reference, not a copy
        # This test just verifies that the properties work correctly
        assert result.details is result._details


class TestResult:
    """Tests for the Result class."""

    def test_init_with_required_fields(self) -> None:
        """Test initializing a Result with only required fields."""
        result = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=[],
            improvement_results=[],
            metadata={},
        )
        assert result.text == "Final text"
        assert result.passed is True
        assert result.validation_results == []
        assert result.improvement_results == []
        assert result.metadata == {}
        assert result.execution_time_ms is None

    def test_init_with_all_fields(self) -> None:
        """Test initializing a Result with all fields."""
        validation_results = [
            ValidationResult(passed=True, message="Validation 1 passed"),
            ValidationResult(passed=True, message="Validation 2 passed"),
        ]
        improvement_results = [
            ImprovementResult(
                _original_text="Original text",
                _improved_text="Improved text",
                _changes_made=True,
                message="Text improved",
            )
        ]
        result = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=validation_results,
            improvement_results=improvement_results,
            metadata={"key": "value"},
            execution_time_ms=123.45,
        )
        assert result.text == "Final text"
        assert result.passed is True
        assert result.validation_results == validation_results
        assert result.improvement_results == improvement_results
        assert result.metadata == {"key": "value"}
        assert result.execution_time_ms == 123.45

    def test_bool_conversion(self) -> None:
        """Test converting Result to boolean."""
        result_true = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=[],
            improvement_results=[],
            metadata={},
        )
        result_false = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[],
            improvement_results=[],
            metadata={},
        )

        assert bool(result_true) is True
        assert bool(result_false) is False

        # Test in if statement
        if result_true:
            assert True
        else:
            assert False

        if result_false:
            assert False
        else:
            assert True

    def test_has_issues_property(self) -> None:
        """Test the has_issues property."""
        # No issues
        result1 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=[ValidationResult(passed=True, message="Validation passed")],
            improvement_results=[],
            metadata={},
        )
        assert result1.has_issues is False

        # With issues
        result2 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[
                ValidationResult(
                    passed=False,
                    message="Validation failed",
                    issues=["Issue 1", "Issue 2"],
                )
            ],
            improvement_results=[],
            metadata={},
        )
        assert result2.has_issues is True

    def test_all_issues_property(self) -> None:
        """Test the all_issues property."""
        # No issues
        result1 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=[
                ValidationResult(passed=True, message="Validation 1 passed"),
                ValidationResult(passed=True, message="Validation 2 passed"),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result1.all_issues == []

        # With issues from one validator
        result2 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[
                ValidationResult(passed=True, message="Validation 1 passed"),
                ValidationResult(
                    passed=False,
                    message="Validation 2 failed",
                    issues=["Issue 1", "Issue 2"],
                ),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result2.all_issues == ["Issue 1", "Issue 2"]

        # With issues from multiple validators
        result3 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[
                ValidationResult(
                    passed=False,
                    message="Validation 1 failed",
                    issues=["Issue A", "Issue B"],
                ),
                ValidationResult(
                    passed=False,
                    message="Validation 2 failed",
                    issues=["Issue C", "Issue D"],
                ),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result3.all_issues == ["Issue A", "Issue B", "Issue C", "Issue D"]

    def test_all_suggestions_property(self) -> None:
        """Test the all_suggestions property."""
        # No suggestions
        result1 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=True,
            validation_results=[
                ValidationResult(passed=True, message="Validation 1 passed"),
                ValidationResult(passed=True, message="Validation 2 passed"),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result1.all_suggestions == []

        # With suggestions from one validator
        result2 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[
                ValidationResult(passed=True, message="Validation 1 passed"),
                ValidationResult(
                    passed=False,
                    message="Validation 2 failed",
                    suggestions=["Suggestion 1", "Suggestion 2"],
                ),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result2.all_suggestions == ["Suggestion 1", "Suggestion 2"]

        # With suggestions from multiple validators
        result3 = Result(
            text="Final text",
            initial_text="Initial text",
            passed=False,
            validation_results=[
                ValidationResult(
                    passed=False,
                    message="Validation 1 failed",
                    suggestions=["Suggestion A", "Suggestion B"],
                ),
                ValidationResult(
                    passed=False,
                    message="Validation 2 failed",
                    suggestions=["Suggestion C", "Suggestion D"],
                ),
            ],
            improvement_results=[],
            metadata={},
        )
        assert result3.all_suggestions == [
            "Suggestion A",
            "Suggestion B",
            "Suggestion C",
            "Suggestion D",
        ]
