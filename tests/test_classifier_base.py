#!/usr/bin/env python3
"""Tests for Sifaka classifier base classes.

This test suite covers the base classifier functionality
to improve test coverage.
"""

import pytest
from unittest.mock import Mock, patch

from sifaka.classifiers.base import (
    ClassificationResult,
    TextClassifier,
    CachedTextClassifier,
    ClassifierError
)


class TestClassificationResult:
    """Test ClassificationResult functionality."""

    def test_classification_result_creation(self):
        """Test basic ClassificationResult creation."""
        result = ClassificationResult(
            label="positive",
            confidence=0.8,
            metadata={"score": 0.8}
        )
        
        assert result.label == "positive"
        assert result.confidence == 0.8
        assert result.metadata["score"] == 0.8

    def test_classification_result_validation(self):
        """Test ClassificationResult validation."""
        # Valid confidence range
        result = ClassificationResult(label="test", confidence=0.0)
        assert result.confidence == 0.0
        
        result = ClassificationResult(label="test", confidence=1.0)
        assert result.confidence == 1.0
        
        # Invalid confidence should raise validation error
        with pytest.raises(ValueError):
            ClassificationResult(label="test", confidence=-0.1)
        
        with pytest.raises(ValueError):
            ClassificationResult(label="test", confidence=1.1)

    def test_classification_result_immutable(self):
        """Test that ClassificationResult is immutable."""
        result = ClassificationResult(label="test", confidence=0.5)
        
        # Should not be able to modify
        with pytest.raises(ValueError):
            result.label = "new_label"

    def test_classification_result_default_metadata(self):
        """Test ClassificationResult with default metadata."""
        result = ClassificationResult(label="test", confidence=0.5)
        assert result.metadata == {}


class MockTextClassifier(TextClassifier):
    """Mock implementation of TextClassifier for testing."""
    
    def __init__(self, name="MockClassifier", description="Test classifier"):
        super().__init__(name, description)
        self.classify_calls = []
    
    def classify(self, text: str) -> ClassificationResult:
        self.classify_calls.append(text)
        
        if not text or not text.strip():
            return ClassificationResult(label="empty", confidence=1.0)
        
        # Simple mock logic
        if "positive" in text.lower():
            return ClassificationResult(label="positive", confidence=0.9)
        elif "negative" in text.lower():
            return ClassificationResult(label="negative", confidence=0.9)
        else:
            return ClassificationResult(label="neutral", confidence=0.5)


class TestTextClassifier:
    """Test TextClassifier base functionality."""

    def test_text_classifier_creation(self):
        """Test basic TextClassifier creation."""
        classifier = MockTextClassifier()
        assert classifier.name == "MockClassifier"
        assert classifier.description == "Test classifier"

    def test_text_classifier_custom_params(self):
        """Test TextClassifier with custom parameters."""
        classifier = MockTextClassifier(name="Custom", description="Custom desc")
        assert classifier.name == "Custom"
        assert classifier.description == "Custom desc"

    def test_classify_method(self):
        """Test the classify method."""
        classifier = MockTextClassifier()
        
        # Test positive classification
        result = classifier.classify("This is positive")
        assert result.label == "positive"
        assert result.confidence == 0.9
        
        # Test negative classification
        result = classifier.classify("This is negative")
        assert result.label == "negative"
        assert result.confidence == 0.9
        
        # Test neutral classification
        result = classifier.classify("This is neutral text")
        assert result.label == "neutral"
        assert result.confidence == 0.5

    def test_batch_classify(self):
        """Test batch classification."""
        classifier = MockTextClassifier()
        
        texts = [
            "This is positive",
            "This is negative", 
            "Neutral text"
        ]
        
        results = classifier.batch_classify(texts)
        
        assert len(results) == 3
        assert results[0].label == "positive"
        assert results[1].label == "negative"
        assert results[2].label == "neutral"

    def test_batch_classify_with_errors(self):
        """Test batch classification with errors."""
        classifier = MockTextClassifier()
        
        # Mock classify to raise an error for specific text
        original_classify = classifier.classify
        def mock_classify(text):
            if text == "error":
                raise ValueError("Test error")
            return original_classify(text)
        
        classifier.classify = mock_classify
        
        texts = ["positive text", "error", "negative text"]
        results = classifier.batch_classify(texts)
        
        assert len(results) == 3
        assert results[0].label == "positive"
        assert results[1].label == "error"  # Error result
        assert results[1].confidence == 0.0
        assert results[2].label == "negative"

    def test_predict_interface(self):
        """Test scikit-learn compatible predict interface."""
        classifier = MockTextClassifier()
        
        texts = ["positive text", "negative text", "neutral text"]
        labels = classifier.predict(texts)
        
        assert len(labels) == 3
        assert labels[0] == "positive"
        assert labels[1] == "negative"
        assert labels[2] == "neutral"

    def test_predict_proba_interface(self):
        """Test scikit-learn compatible predict_proba interface."""
        classifier = MockTextClassifier()
        
        texts = ["positive text", "negative text"]
        probabilities = classifier.predict_proba(texts)
        
        assert len(probabilities) == 2
        assert len(probabilities[0]) == 2  # Binary probabilities
        assert len(probabilities[1]) == 2
        
        # Check that probabilities sum to 1
        assert abs(sum(probabilities[0]) - 1.0) < 0.001
        assert abs(sum(probabilities[1]) - 1.0) < 0.001

    def test_get_classes(self):
        """Test get_classes method."""
        classifier = MockTextClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "negative" in classes
        assert "positive" in classes

    def test_string_representations(self):
        """Test string representations."""
        classifier = MockTextClassifier()
        
        str_repr = str(classifier)
        assert "MockClassifier" in str_repr
        assert "Test classifier" in str_repr
        
        repr_str = repr(classifier)
        assert "MockTextClassifier" in repr_str
        assert "MockClassifier" in repr_str


class MockCachedClassifier(CachedTextClassifier):
    """Mock implementation of CachedTextClassifier for testing."""
    
    def __init__(self, name="MockCached", description="Test cached classifier", cache_size=128):
        super().__init__(name, description, cache_size)
        self.classify_calls = []
    
    def _classify_uncached(self, text: str) -> ClassificationResult:
        self.classify_calls.append(text)
        
        if not text or not text.strip():
            return ClassificationResult(label="empty", confidence=1.0)
        
        # Simple mock logic
        if "positive" in text.lower():
            return ClassificationResult(label="positive", confidence=0.9)
        elif "negative" in text.lower():
            return ClassificationResult(label="negative", confidence=0.9)
        else:
            return ClassificationResult(label="neutral", confidence=0.5)


class TestCachedTextClassifier:
    """Test CachedTextClassifier functionality."""

    def test_cached_classifier_creation(self):
        """Test basic CachedTextClassifier creation."""
        classifier = MockCachedClassifier()
        assert classifier.name == "MockCached"
        assert classifier.cache_size == 128

    def test_cached_classifier_custom_cache_size(self):
        """Test CachedTextClassifier with custom cache size."""
        classifier = MockCachedClassifier(cache_size=64)
        assert classifier.cache_size == 64

    def test_caching_behavior(self):
        """Test that caching works correctly."""
        classifier = MockCachedClassifier()
        
        # First call should invoke _classify_uncached
        result1 = classifier.classify("test text")
        assert len(classifier.classify_calls) == 1
        
        # Second call with same text should use cache
        result2 = classifier.classify("test text")
        assert len(classifier.classify_calls) == 1  # No additional call
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cache_different_texts(self):
        """Test caching with different texts."""
        classifier = MockCachedClassifier()
        
        # Different texts should each be cached separately
        result1 = classifier.classify("positive text")
        result2 = classifier.classify("negative text")
        result3 = classifier.classify("positive text")  # Should use cache
        
        assert len(classifier.classify_calls) == 2  # Only 2 unique texts
        assert result1.label == result3.label  # Same result from cache

    def test_empty_text_not_cached(self):
        """Test that empty text is not cached."""
        classifier = MockCachedClassifier()
        
        # Empty text should not be cached
        result1 = classifier.classify("")
        result2 = classifier.classify("")
        
        assert len(classifier.classify_calls) == 2  # Both calls made
        assert result1.label == "empty"
        assert result2.label == "empty"

    def test_clear_cache(self):
        """Test clearing the cache."""
        classifier = MockCachedClassifier()
        
        # Make some cached calls
        classifier.classify("test text 1")
        classifier.classify("test text 2")
        assert len(classifier.classify_calls) == 2
        
        # Clear cache
        classifier.clear_cache()
        
        # Same text should now invoke _classify_uncached again
        classifier.classify("test text 1")
        assert len(classifier.classify_calls) == 3

    def test_get_cache_info(self):
        """Test getting cache statistics."""
        classifier = MockCachedClassifier()
        
        # Initial cache info
        info = classifier.get_cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["current_size"] == 0
        assert info["hit_rate"] == 0.0
        
        # Make some calls
        classifier.classify("test text")  # Miss
        classifier.classify("test text")  # Hit
        classifier.classify("other text")  # Miss
        
        info = classifier.get_cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2
        assert info["current_size"] == 2
        assert info["hit_rate"] == 1/3

    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        classifier = MockCachedClassifier(cache_size=2)
        
        # Fill cache beyond limit
        classifier.classify("text 1")
        classifier.classify("text 2")
        classifier.classify("text 3")  # Should evict oldest
        
        # First text should have been evicted
        classifier.classify("text 1")  # Should be a miss now
        
        info = classifier.get_cache_info()
        assert info["current_size"] <= 2


class TestClassifierError:
    """Test ClassifierError functionality."""

    def test_classifier_error_creation(self):
        """Test ClassifierError creation."""
        error = ClassifierError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
