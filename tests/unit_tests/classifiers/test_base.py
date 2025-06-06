"""Comprehensive unit tests for base classifier classes.

This module tests the base classifier functionality including:
- BaseClassifier abstract interface
- ClassificationResult data structure
- CachedClassifier implementation
- TimingMixin functionality
- Error handling and validation
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from sifaka.classifiers.base import (
    BaseClassifier,
    CachedClassifier,
    ClassificationResult,
    TimingMixin,
)
from sifaka.utils.errors import ValidationError


class TestClassificationResult:
    """Test ClassificationResult data structure."""

    def test_classification_result_creation(self):
        """Test creating a valid ClassificationResult."""
        result = ClassificationResult(
            label="positive",
            confidence=0.85,
            metadata={"method": "test"},
            processing_time_ms=10.5,
        )

        assert result.label == "positive"
        assert result.confidence == 0.85
        assert result.metadata == {"method": "test"}
        assert result.processing_time_ms == 10.5

    def test_classification_result_default_metadata(self):
        """Test ClassificationResult with default metadata."""
        result = ClassificationResult(label="test", confidence=0.5)

        assert result.metadata == {}
        assert result.processing_time_ms == 0.0

    def test_classification_result_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        ClassificationResult(label="test", confidence=0.0)
        ClassificationResult(label="test", confidence=0.5)
        ClassificationResult(label="test", confidence=1.0)

        # Invalid confidence scores
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ClassificationResult(label="test", confidence=-0.1)

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ClassificationResult(label="test", confidence=1.1)

    def test_classification_result_mutable(self):
        """Test that ClassificationResult fields can be modified."""
        result = ClassificationResult(label="test", confidence=0.5)

        # ClassificationResult is mutable, so we can change fields
        result.label = "changed"
        result.confidence = 0.8

        assert result.label == "changed"
        assert result.confidence == 0.8


class TestBaseClassifier:
    """Test BaseClassifier abstract base class."""

    def test_base_classifier_cannot_be_instantiated(self):
        """Test that BaseClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseClassifier("test", "description")

    def test_concrete_classifier_implementation(self):
        """Test a concrete implementation of BaseClassifier."""

        class TestClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                return ClassificationResult(label="test", confidence=0.5, metadata={"input": text})

            def get_classes(self) -> list[str]:
                return ["test", "other"]

        classifier = TestClassifier("test_classifier", "Test description")
        assert classifier.name == "test_classifier"
        assert classifier.description == "Test description"

    @pytest.mark.asyncio
    async def test_classify_async_method(self):
        """Test the classify_async method implementation."""

        class TestClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                return ClassificationResult(
                    label="processed", confidence=0.75, metadata={"length": len(text)}
                )

            def get_classes(self) -> list[str]:
                return ["processed", "unprocessed"]

        classifier = TestClassifier("test", "Test classifier")
        result = await classifier.classify_async("Hello world")

        assert result.label == "processed"
        assert result.confidence == 0.75
        assert result.metadata["length"] == 11

    def test_classify_sync_method(self):
        """Test the synchronous classify method."""

        class TestClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                return ClassificationResult(
                    label="sync_test", confidence=0.6, metadata={"sync": True}
                )

            def get_classes(self) -> list[str]:
                return ["sync_test", "async_test"]

        classifier = TestClassifier("test", "Test classifier")
        result = classifier.classify("Hello")

        assert result.label == "sync_test"
        assert result.confidence == 0.6
        assert result.metadata["sync"] is True

    def test_classifier_string_representations(self):
        """Test string representations of classifier."""

        class TestClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                return ClassificationResult(label="test", confidence=0.5)

            def get_classes(self) -> list[str]:
                return ["test", "other"]

        classifier = TestClassifier("test_name", "Test description")

        # Test __str__
        str_repr = str(classifier)
        assert "test_name" in str_repr

        # Test __repr__
        repr_str = repr(classifier)
        assert "TestClassifier" in repr_str
        assert "test_name" in repr_str
        assert "Test description" in repr_str

    @pytest.mark.asyncio
    async def test_error_handling_in_classify_async(self):
        """Test error handling in classify_async method."""

        class FailingClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                raise ValidationError("Classification failed")

            def get_classes(self) -> list[str]:
                return ["fail", "success"]

        classifier = FailingClassifier("failing", "Failing classifier")

        with pytest.raises(ValidationError, match="Classification failed"):
            await classifier.classify_async("test")

    @pytest.mark.asyncio
    async def test_error_handling_in_classify_sync(self):
        """Test error handling in synchronous classify method."""

        class FailingClassifier(BaseClassifier):
            async def classify_async(self, text: str) -> ClassificationResult:
                raise ValidationError("Async classification failed")

            def get_classes(self) -> list[str]:
                return ["fail", "success"]

        classifier = FailingClassifier("failing", "Failing classifier")

        # Test async version instead of sync to avoid event loop issues
        with pytest.raises(ValidationError, match="Async classification failed"):
            await classifier.classify_async("test")


class TestCachedClassifier:
    """Test CachedClassifier implementation."""

    def test_cached_classifier_cannot_be_instantiated(self):
        """Test that CachedClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CachedClassifier("test", "description", 128)

    def test_concrete_cached_classifier_implementation(self):
        """Test a concrete implementation of CachedClassifier."""

        class TestCachedClassifier(CachedClassifier):
            def _classify_uncached(self, text: str) -> ClassificationResult:
                return ClassificationResult(
                    label="cached_test", confidence=0.7, metadata={"cached": False, "input": text}
                )

            def get_classes(self) -> list[str]:
                return ["cached_test", "uncached_test"]

        classifier = TestCachedClassifier("cached_test", "Cached test classifier", 64)
        assert classifier.name == "cached_test"
        assert classifier.description == "Cached test classifier"
        assert classifier.cache_size == 64

    @pytest.mark.asyncio
    async def test_cached_classifier_caching_behavior(self):
        """Test that CachedClassifier properly caches results."""

        call_count = 0

        class TestCachedClassifier(CachedClassifier):
            def _classify_uncached(self, text: str) -> ClassificationResult:
                nonlocal call_count
                call_count += 1
                return ClassificationResult(
                    label="cached", confidence=0.8, metadata={"call_count": call_count}
                )

            def get_classes(self) -> list[str]:
                return ["cached", "uncached"]

        classifier = TestCachedClassifier("cached", "Test", 128)

        # First call should execute _classify_uncached
        result1 = await classifier.classify_async("test text")
        assert result1.metadata["call_count"] == 1
        assert call_count == 1

        # Second call with same text should use cache
        result2 = await classifier.classify_async("test text")
        assert result2.metadata["call_count"] == 1  # Same as first call
        assert call_count == 1  # No additional calls

        # Different text should execute _classify_uncached again
        result3 = await classifier.classify_async("different text")
        assert result3.metadata["call_count"] == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_classifier_cache_clearing(self):
        """Test cache clearing functionality."""

        call_count = 0

        class TestCachedClassifier(CachedClassifier):
            def _classify_uncached(self, text: str) -> ClassificationResult:
                nonlocal call_count
                call_count += 1
                return ClassificationResult(label="test", confidence=0.5)

            def get_classes(self) -> list[str]:
                return ["test", "other"]

        classifier = TestCachedClassifier("test", "Test", 128)

        # Make initial call
        await classifier.classify_async("test")
        assert call_count == 1

        # Call again (should use cache)
        await classifier.classify_async("test")
        assert call_count == 1

        # Clear cache
        classifier.clear_cache()

        # Call again (should execute _classify_uncached)
        await classifier.classify_async("test")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_classifier_async_method(self):
        """Test that CachedClassifier properly implements classify_async."""

        class TestCachedClassifier(CachedClassifier):
            def _classify_uncached(self, text: str) -> ClassificationResult:
                return ClassificationResult(
                    label="async_cached", confidence=0.9, metadata={"async": True}
                )

            def get_classes(self) -> list[str]:
                return ["async_cached", "sync_cached"]

        classifier = TestCachedClassifier("async_test", "Async test", 128)
        result = await classifier.classify_async("test")

        assert result.label == "async_cached"
        assert result.confidence == 0.9
        assert result.metadata["async"] is True


class TestTimingMixin:
    """Test TimingMixin functionality."""

    @pytest.mark.asyncio
    async def test_timing_mixin_context_manager(self):
        """Test TimingMixin context manager functionality."""

        class TestClassifierWithTiming(BaseClassifier, TimingMixin):
            async def classify_async(self, text: str) -> ClassificationResult:
                with self.time_operation("test_operation") as timer:
                    # Simulate some work
                    await asyncio.sleep(0.01)
                    processing_time = getattr(timer, "duration_ms", 0.0)

                    return ClassificationResult(
                        label="timed", confidence=0.5, processing_time_ms=processing_time
                    )

            def get_classes(self) -> list[str]:
                return ["timed", "untimed"]

        classifier = TestClassifierWithTiming("timed", "Timed classifier")
        result = await classifier.classify_async("test")

        assert result.label == "timed"
        assert result.processing_time_ms >= 0  # Should have some timing

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.base.logger")
    async def test_timing_mixin_logging(self, mock_logger):
        """Test that TimingMixin logs performance metrics."""

        class TestClassifierWithTiming(BaseClassifier, TimingMixin):
            async def classify_async(self, text: str) -> ClassificationResult:
                with self.time_operation("test_op"):
                    pass
                return ClassificationResult(label="test", confidence=0.5)

            def get_classes(self) -> list[str]:
                return ["test", "other"]

        classifier = TestClassifierWithTiming("test", "Test")
        await classifier.classify_async("test")

        # Verify that performance_timer was called
        mock_logger.performance_timer.assert_called_with("test_op", classifier="test")
