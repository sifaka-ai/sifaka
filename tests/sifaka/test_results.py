"""
Tests for the result types.
"""

import pytest

from sifaka.results import ValidationResult, ImprovementResult, Result


class TestValidationResult:
    """Tests for the ValidationResult class."""
    
    def test_initialization(self):
        """Test that a ValidationResult can be initialized."""
        result = ValidationResult(passed=True, message="Validation passed")
        assert result.passed is True
        assert result.message == "Validation passed"
        assert result.details is None
    
    def test_bool_conversion(self):
        """Test that a ValidationResult can be used in boolean context."""
        assert bool(ValidationResult(passed=True)) is True
        assert bool(ValidationResult(passed=False)) is False


class TestImprovementResult:
    """Tests for the ImprovementResult class."""
    
    def test_initialization(self):
        """Test that an ImprovementResult can be initialized."""
        result = ImprovementResult(
            original_text="Original",
            improved_text="Improved",
            changes_made=True,
            message="Improvement applied"
        )
        assert result.original_text == "Original"
        assert result.improved_text == "Improved"
        assert result.changes_made is True
        assert result.message == "Improvement applied"
        assert result.details is None
    
    def test_bool_conversion(self):
        """Test that an ImprovementResult can be used in boolean context."""
        assert bool(ImprovementResult(
            original_text="Original",
            improved_text="Improved",
            changes_made=True
        )) is True
        
        assert bool(ImprovementResult(
            original_text="Original",
            improved_text="Original",
            changes_made=False
        )) is False


class TestResult:
    """Tests for the Result class."""
    
    def test_initialization(self):
        """Test that a Result can be initialized."""
        validation_results = [
            ValidationResult(passed=True, message="Validation 1 passed"),
            ValidationResult(passed=True, message="Validation 2 passed")
        ]
        
        improvement_results = [
            ImprovementResult(
                original_text="Original",
                improved_text="Improved",
                changes_made=True,
                message="Improvement applied"
            )
        ]
        
        result = Result(
            text="Final text",
            passed=True,
            validation_results=validation_results,
            improvement_results=improvement_results,
            metadata={"tokens": 100}
        )
        
        assert result.text == "Final text"
        assert result.passed is True
        assert result.validation_results == validation_results
        assert result.improvement_results == improvement_results
        assert result.metadata == {"tokens": 100}
    
    def test_bool_conversion(self):
        """Test that a Result can be used in boolean context."""
        assert bool(Result(
            text="Text",
            passed=True,
            validation_results=[],
            improvement_results=[]
        )) is True
        
        assert bool(Result(
            text="Text",
            passed=False,
            validation_results=[],
            improvement_results=[]
        )) is False
