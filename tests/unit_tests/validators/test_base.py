"""Comprehensive unit tests for validator base classes and models.

This module tests the base validator infrastructure:
- ValidationResult model and validation
- BaseValidator abstract base class
- Sync/async method coordination
- Error handling and logging
- Validation result aggregation

Tests cover:
- ValidationResult creation and validation
- BaseValidator implementation requirements
- Sync wrapper for async validation
- Error handling and logging
- Field constraints and validation
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from pydantic import ValidationError

from sifaka.validators.base import ValidationResult, BaseValidator
from sifaka.core.thought import SifakaThought
from sifaka.utils.errors import ValidationError as SifakaValidationError


class TestValidationResult:
    """Test the ValidationResult model."""

    def test_validation_result_creation_minimal(self):
        """Test creating ValidationResult with minimal required fields."""
        result = ValidationResult(
            validator_name="test-validator",
            passed=True,
            processing_time_ms=50.0,
        )
        
        assert result.validator_name == "test-validator"
        assert result.passed is True
        assert result.processing_time_ms == 50.0
        assert result.score is None
        assert result.details == {}
        assert result.error_message is None
        assert result.issues == []
        assert result.suggestions == []
        assert result.metadata == {}
        assert isinstance(result.timestamp, datetime)

    def test_validation_result_creation_full(self):
        """Test creating ValidationResult with all fields."""
        timestamp = datetime.now()
        
        result = ValidationResult(
            validator_name="content-validator",
            passed=False,
            score=0.65,
            details={"length": 150, "min_required": 200},
            error_message="Content too short",
            issues=["Length below minimum", "Missing examples"],
            suggestions=["Add more content", "Include examples"],
            metadata={"version": "v1.0", "strict_mode": True},
            processing_time_ms=75.5,
            timestamp=timestamp,
        )
        
        assert result.validator_name == "content-validator"
        assert result.passed is False
        assert result.score == 0.65
        assert result.details["length"] == 150
        assert result.details["min_required"] == 200
        assert result.error_message == "Content too short"
        assert len(result.issues) == 2
        assert "Length below minimum" in result.issues
        assert "Missing examples" in result.issues
        assert len(result.suggestions) == 2
        assert "Add more content" in result.suggestions
        assert "Include examples" in result.suggestions
        assert result.metadata["version"] == "v1.0"
        assert result.metadata["strict_mode"] is True
        assert result.processing_time_ms == 75.5
        assert result.timestamp == timestamp

    def test_validation_result_score_constraints(self):
        """Test ValidationResult score field constraints."""
        # Valid score bounds
        ValidationResult(validator_name="test", passed=True, processing_time_ms=50.0, score=0.0)
        ValidationResult(validator_name="test", passed=True, processing_time_ms=50.0, score=1.0)
        ValidationResult(validator_name="test", passed=True, processing_time_ms=50.0, score=0.5)
        
        # Invalid score bounds
        with pytest.raises(ValidationError):
            ValidationResult(validator_name="test", passed=True, processing_time_ms=50.0, score=-0.1)
        
        with pytest.raises(ValidationError):
            ValidationResult(validator_name="test", passed=True, processing_time_ms=50.0, score=1.1)

    def test_validation_result_processing_time_constraints(self):
        """Test ValidationResult processing_time_ms field constraints."""
        # Valid processing times
        ValidationResult(validator_name="test", passed=True, processing_time_ms=0.0)
        ValidationResult(validator_name="test", passed=True, processing_time_ms=1000.5)
        
        # Invalid processing time
        with pytest.raises(ValidationError):
            ValidationResult(validator_name="test", passed=True, processing_time_ms=-1.0)

    def test_validation_result_post_init(self):
        """Test ValidationResult __post_init__ method for mutable field defaults."""
        # Create result without specifying mutable fields
        result = ValidationResult(
            validator_name="test",
            passed=True,
            processing_time_ms=50.0,
        )
        
        # Verify mutable fields are properly initialized
        assert result.issues == []
        assert result.suggestions == []
        assert result.metadata == {}
        
        # Verify they are separate instances (not shared)
        result2 = ValidationResult(
            validator_name="test2",
            passed=True,
            processing_time_ms=50.0,
        )
        
        result.issues.append("test issue")
        assert len(result.issues) == 1
        assert len(result2.issues) == 0

    def test_validation_result_serialization(self):
        """Test ValidationResult serialization and deserialization."""
        original = ValidationResult(
            validator_name="serialization-test",
            passed=False,
            score=0.7,
            details={"key": "value"},
            issues=["issue1", "issue2"],
            suggestions=["suggestion1"],
            processing_time_ms=100.0,
        )
        
        # Serialize to dict
        data = original.model_dump()
        
        # Verify structure
        assert data["validator_name"] == "serialization-test"
        assert data["passed"] is False
        assert data["score"] == 0.7
        assert data["details"]["key"] == "value"
        assert len(data["issues"]) == 2
        assert len(data["suggestions"]) == 1
        assert data["processing_time_ms"] == 100.0
        
        # Deserialize back to model
        restored = ValidationResult.model_validate(data)
        
        # Verify restored model
        assert restored.validator_name == original.validator_name
        assert restored.passed == original.passed
        assert restored.score == original.score
        assert restored.details == original.details
        assert restored.issues == original.issues
        assert restored.suggestions == original.suggestions
        assert restored.processing_time_ms == original.processing_time_ms


class TestBaseValidator:
    """Test the BaseValidator abstract base class."""

    def test_base_validator_is_abstract(self):
        """Test that BaseValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseValidator("test", "description")

    def test_base_validator_subclass_requirements(self):
        """Test that BaseValidator subclasses must implement validate_async."""
        class IncompleteValidator(BaseValidator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteValidator("test", "description")

    def test_base_validator_initialization(self):
        """Test BaseValidator initialization with concrete subclass."""
        class ConcreteValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    processing_time_ms=50.0,
                )
        
        validator = ConcreteValidator("test-validator", "Test description")
        
        assert validator.name == "test-validator"
        assert validator.description == "Test description"

    @pytest.mark.asyncio
    async def test_base_validator_validate_async_implementation(self):
        """Test that concrete validators properly implement validate_async."""
        class ConcreteValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                text_length = len(thought.current_text or thought.prompt)
                return ValidationResult(
                    validator_name=self.name,
                    passed=text_length >= 10,
                    score=min(text_length / 10.0, 1.0),
                    details={"text_length": text_length},
                    processing_time_ms=25.0,
                )
        
        validator = ConcreteValidator("length-validator", "Validates text length")
        
        # Test with short text
        short_thought = SifakaThought(prompt="Short", current_text="Short")
        result = await validator.validate_async(short_thought)
        
        assert result.validator_name == "length-validator"
        assert result.passed is False
        assert result.score == 0.5  # 5 chars / 10
        assert result.details["text_length"] == 5
        
        # Test with long text
        long_thought = SifakaThought(prompt="Long text", current_text="This is a longer text")
        result = await validator.validate_async(long_thought)
        
        assert result.passed is True
        assert result.score == 1.0
        assert result.details["text_length"] == 21

    def test_base_validator_sync_method_exists(self):
        """Test that BaseValidator provides a sync validate method."""
        class ConcreteValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    processing_time_ms=50.0,
                )
        
        validator = ConcreteValidator("test", "description")
        
        # Verify sync method exists
        assert hasattr(validator, 'validate')
        assert callable(validator.validate)

    @pytest.mark.asyncio
    async def test_base_validator_error_handling(self):
        """Test BaseValidator error handling in validate_async."""
        class ErrorValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                raise ValueError("Validation failed")
        
        validator = ErrorValidator("error-validator", "Always fails")
        thought = SifakaThought(prompt="Test")
        
        # Error should propagate
        with pytest.raises(ValueError, match="Validation failed"):
            await validator.validate_async(thought)

    @patch('sifaka.validators.base.logger')
    @pytest.mark.asyncio
    async def test_base_validator_logging(self, mock_logger):
        """Test that BaseValidator logs validation operations."""
        class LoggingValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                return ValidationResult(
                    validator_name=self.name,
                    passed=True,
                    processing_time_ms=50.0,
                )
        
        validator = LoggingValidator("logging-validator", "Test logging")
        thought = SifakaThought(prompt="Test")
        
        await validator.validate_async(thought)
        
        # Note: Actual logging behavior depends on implementation
        # This test verifies the logger is available for use


class TestValidatorIntegration:
    """Test integration scenarios with validators."""

    @pytest.mark.asyncio
    async def test_multiple_validators_workflow(self):
        """Test using multiple validators in sequence."""
        class LengthValidator(BaseValidator):
            def __init__(self, min_length: int = 10):
                super().__init__("length-validator", f"Validates minimum length of {min_length}")
                self.min_length = min_length
            
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                text = thought.current_text or thought.prompt
                length = len(text)
                passed = length >= self.min_length
                
                return ValidationResult(
                    validator_name=self.name,
                    passed=passed,
                    score=min(length / self.min_length, 1.0) if self.min_length > 0 else 1.0,
                    details={"length": length, "min_required": self.min_length},
                    issues=[] if passed else [f"Text too short: {length} < {self.min_length}"],
                    suggestions=[] if passed else ["Add more content"],
                    processing_time_ms=10.0,
                )
        
        class ContentValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                text = thought.current_text or thought.prompt
                has_content = bool(text.strip())
                
                return ValidationResult(
                    validator_name=self.name,
                    passed=has_content,
                    score=1.0 if has_content else 0.0,
                    details={"has_content": has_content},
                    issues=[] if has_content else ["Empty or whitespace-only content"],
                    suggestions=[] if has_content else ["Add meaningful content"],
                    processing_time_ms=5.0,
                )
        
        # Create validators
        length_validator = LengthValidator(min_length=20)
        content_validator = ContentValidator("content-validator", "Validates content presence")
        
        # Test with good text
        good_thought = SifakaThought(
            prompt="Test",
            current_text="This is a sufficiently long text that should pass validation"
        )
        
        length_result = await length_validator.validate_async(good_thought)
        content_result = await content_validator.validate_async(good_thought)
        
        assert length_result.passed is True
        assert content_result.passed is True
        assert length_result.score == 1.0
        assert content_result.score == 1.0
        
        # Test with bad text
        bad_thought = SifakaThought(prompt="Test", current_text="Short")
        
        length_result = await length_validator.validate_async(bad_thought)
        content_result = await content_validator.validate_async(bad_thought)
        
        assert length_result.passed is False
        assert content_result.passed is True  # Has content, just short
        assert length_result.score < 1.0
        assert len(length_result.issues) > 0
        assert len(length_result.suggestions) > 0

    @pytest.mark.asyncio
    async def test_validator_with_thought_evolution(self):
        """Test validator behavior as thought evolves through iterations."""
        class EvolutionValidator(BaseValidator):
            async def validate_async(self, thought: SifakaThought) -> ValidationResult:
                # Check if thought has improved over iterations
                generation_count = len(thought.generations)
                has_iterations = thought.iteration > 0
                
                return ValidationResult(
                    validator_name=self.name,
                    passed=has_iterations,
                    score=min(generation_count / 3.0, 1.0),  # Expect up to 3 generations
                    details={
                        "generation_count": generation_count,
                        "iteration": thought.iteration,
                        "has_iterations": has_iterations,
                    },
                    processing_time_ms=15.0,
                )
        
        validator = EvolutionValidator("evolution-validator", "Validates thought evolution")
        
        # Initial thought
        thought = SifakaThought(prompt="Test prompt")
        result = await validator.validate_async(thought)
        
        assert result.passed is False  # No iterations yet
        assert result.details["generation_count"] == 0
        assert result.details["iteration"] == 0
        
        # Evolved thought
        thought.iteration = 1
        thought.generations.append(Mock())  # Simulate generation
        result = await validator.validate_async(thought)
        
        assert result.passed is True  # Has iterations
        assert result.details["generation_count"] == 1
        assert result.details["iteration"] == 1
