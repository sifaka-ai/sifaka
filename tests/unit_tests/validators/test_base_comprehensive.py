"""Comprehensive unit tests for validator base classes and interfaces.

This module provides extensive testing for the validator infrastructure:
- BaseValidator abstract base class
- ValidationResult data structure
- Validator protocol compliance
- Error handling and edge cases
- Performance characteristics
- Integration patterns
"""

import asyncio
from typing import Any, Dict

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.validators.base import BaseValidator, ValidationResult


class MockValidator(BaseValidator):
    """Mock implementation of BaseValidator for testing."""

    def __init__(
        self,
        name: str = "mock_validator",
        always_pass: bool = True,
        metadata: Dict[str, Any] = None,
    ):
        super().__init__(name=name, description=f"Mock validator: {name}")
        self.always_pass = always_pass
        self.metadata = metadata or {}
        self.call_count = 0

    async def validate_async(self, text: str) -> ValidationResult:
        """Mock validation that can be configured to pass or fail."""
        self.call_count += 1

        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                validator_name=self.name,
                message="Empty text provided",
                metadata={"reason": "empty_text", "call_count": self.call_count},
            )

        return ValidationResult(
            is_valid=self.always_pass,
            validator_name=self.name,
            message="Validation passed" if self.always_pass else "Validation failed",
            metadata={**self.metadata, "call_count": self.call_count, "text_length": len(text)},
        )


class TestValidationResult:
    """Test ValidationResult data structure."""

    def test_validation_result_creation(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            validator_name="test_validator",
            message="Validation successful",
            metadata={"score": 0.95},
        )

        assert result.is_valid is True
        assert result.validator_name == "test_validator"
        assert result.message == "Validation successful"
        assert result.metadata["score"] == 0.95

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_valid=False, validator_name="test")

        assert result.is_valid is False
        assert result.validator_name == "test"
        assert result.message == ""
        assert result.metadata == {}

    def test_validation_result_immutability(self):
        """Test that ValidationResult can be modified (it's mutable)."""
        result = ValidationResult(is_valid=True, validator_name="test")

        # ValidationResult should be mutable for practical use
        result.message = "Updated message"
        result.metadata["new_key"] = "new_value"

        assert result.message == "Updated message"
        assert result.metadata["new_key"] == "new_value"

    def test_validation_result_string_representation(self):
        """Test string representations of ValidationResult."""
        result = ValidationResult(
            is_valid=True, validator_name="test_validator", message="All good"
        )

        str_repr = str(result)
        assert "test_validator" in str_repr
        assert "True" in str_repr or "valid" in str_repr.lower()

        repr_str = repr(result)
        assert "ValidationResult" in repr_str
        assert "test_validator" in repr_str


class TestBaseValidator:
    """Test BaseValidator abstract base class."""

    def test_base_validator_cannot_be_instantiated(self):
        """Test that BaseValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseValidator("test", "description")

    def test_concrete_validator_implementation(self):
        """Test a concrete implementation of BaseValidator."""
        validator = MockValidator("test_validator")

        assert validator.name == "test_validator"
        assert validator.description == "Mock validator: test_validator"

    @pytest.mark.asyncio
    async def test_validate_async_method(self):
        """Test the validate_async method implementation."""
        validator = MockValidator("test", always_pass=True)

        result = await validator.validate_async("Hello world")

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.validator_name == "test"
        assert result.metadata["text_length"] == 11
        assert validator.call_count == 1

    def test_validate_sync_method(self):
        """Test the synchronous validate method."""
        validator = MockValidator("sync_test", always_pass=True)

        result = validator.validate("Hello world")

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.validator_name == "sync_test"
        assert validator.call_count == 1

    @pytest.mark.asyncio
    async def test_validator_with_failing_validation(self):
        """Test validator that fails validation."""
        validator = MockValidator("failing", always_pass=False)

        result = await validator.validate_async("Some text")

        assert result.is_valid is False
        assert result.validator_name == "failing"
        assert result.message == "Validation failed"

    @pytest.mark.asyncio
    async def test_validator_with_empty_text(self):
        """Test validator behavior with empty text."""
        validator = MockValidator("empty_test")

        # Test empty string
        result = await validator.validate_async("")
        assert result.is_valid is False
        assert result.metadata["reason"] == "empty_text"

        # Test whitespace only
        result = await validator.validate_async("   ")
        assert result.is_valid is False

        # Test None (should be handled gracefully)
        result = await validator.validate_async(None)
        assert result.is_valid is False

    def test_validator_string_representations(self):
        """Test string representations of validator."""
        validator = MockValidator("repr_test")

        # Test __str__
        str_repr = str(validator)
        assert "repr_test" in str_repr

        # Test __repr__
        repr_str = repr(validator)
        assert "MockValidator" in repr_str
        assert "repr_test" in repr_str

    @pytest.mark.asyncio
    async def test_error_handling_in_validate_async(self):
        """Test error handling in validate_async method."""

        class FailingValidator(BaseValidator):
            async def validate_async(self, text: str) -> ValidationResult:
                raise ValueError("Validation error")

        validator = FailingValidator("failing", "Failing validator")

        with pytest.raises(ValueError, match="Validation error"):
            await validator.validate_async("test")

    def test_error_handling_in_validate_sync(self):
        """Test error handling in synchronous validate method."""

        class FailingValidator(BaseValidator):
            async def validate_async(self, text: str) -> ValidationResult:
                raise ValueError("Async validation error")

        validator = FailingValidator("failing", "Failing validator")

        # The sync method should wrap async errors appropriately
        with pytest.raises(
            Exception
        ):  # Could be ValueError or RuntimeError depending on implementation
            validator.validate("test")

    @pytest.mark.asyncio
    async def test_validator_metadata_handling(self):
        """Test that validators properly handle metadata."""
        custom_metadata = {"custom_key": "custom_value", "score": 0.85}
        validator = MockValidator("metadata_test", metadata=custom_metadata)

        result = await validator.validate_async("Test text")

        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["score"] == 0.85
        assert result.metadata["text_length"] == 9  # Length of "Test text"
        assert result.metadata["call_count"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_validation_calls(self):
        """Test concurrent validation calls."""
        validator = MockValidator("concurrent_test")

        # Make multiple concurrent validation calls
        texts = [f"Text {i}" for i in range(10)]
        tasks = [validator.validate_async(text) for text in texts]
        results = await asyncio.gather(*tasks)

        # Verify all validations completed
        assert len(results) == 10
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all(result.is_valid for result in results)
        assert validator.call_count == 10

    @pytest.mark.asyncio
    async def test_validator_state_isolation(self):
        """Test that validator instances maintain separate state."""
        validator1 = MockValidator("state_test_1")
        validator2 = MockValidator("state_test_2")

        await validator1.validate_async("Text 1")
        await validator1.validate_async("Text 2")
        await validator2.validate_async("Text 3")

        assert validator1.call_count == 2
        assert validator2.call_count == 1

    def test_validator_configuration(self):
        """Test validator configuration and initialization."""
        validator = MockValidator(
            name="config_test", always_pass=False, metadata={"threshold": 0.8, "strict_mode": True}
        )

        assert validator.name == "config_test"
        assert validator.always_pass is False
        assert validator.metadata["threshold"] == 0.8
        assert validator.metadata["strict_mode"] is True

    @pytest.mark.asyncio
    async def test_validation_with_thought_integration(self):
        """Test validator integration with SifakaThought objects."""
        validator = MockValidator("thought_integration")

        # Create a thought
        thought = SifakaThought(prompt="Test prompt", final_text="Test response")

        # Validate the thought's final text
        result = await validator.validate_async(thought.final_text)

        assert result.is_valid is True
        assert result.validator_name == "thought_integration"
        assert result.metadata["text_length"] == len(thought.final_text)

    @pytest.mark.asyncio
    async def test_validator_performance_characteristics(self):
        """Test validator performance and timing."""
        import time

        class SlowValidator(BaseValidator):
            async def validate_async(self, text: str) -> ValidationResult:
                await asyncio.sleep(0.01)  # Simulate slow validation
                return ValidationResult(
                    is_valid=True, validator_name=self.name, message="Slow validation completed"
                )

        validator = SlowValidator("slow_test", "Slow validator")

        start_time = time.time()
        result = await validator.validate_async("Test text")
        end_time = time.time()

        assert result.is_valid is True
        assert (end_time - start_time) >= 0.01  # Should take at least 10ms
        assert result.message == "Slow validation completed"
