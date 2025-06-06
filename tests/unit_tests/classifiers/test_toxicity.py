"""Comprehensive unit tests for ToxicityClassifier.

This module tests the toxicity detection classifier including:
- Toxicity detection using transformers pipeline
- Rule-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.toxicity import (
    ToxicityClassifier,
    CachedToxicityClassifier,
    create_toxicity_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestToxicityClassifier:
    """Test ToxicityClassifier implementation."""

    def test_toxicity_classifier_initialization(self):
        """Test ToxicityClassifier initialization with default parameters."""
        classifier = ToxicityClassifier()
        
        assert classifier.name == "toxicity_detection"
        assert "toxicity" in classifier.description.lower()
        assert classifier.model_name == "unitary/toxic-bert-base"
        assert classifier.threshold == 0.7

    def test_toxicity_classifier_custom_parameters(self):
        """Test ToxicityClassifier initialization with custom parameters."""
        classifier = ToxicityClassifier(
            model_name="custom/toxicity-model",
            threshold=0.8,
            name="custom_toxicity",
            description="Custom toxicity detector"
        )
        
        assert classifier.name == "custom_toxicity"
        assert classifier.description == "Custom toxicity detector"
        assert classifier.model_name == "custom/toxicity-model"
        assert classifier.threshold == 0.8

    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    def test_toxicity_classifier_transformers_initialization(self, mock_import):
        """Test ToxicityClassifier initialization with transformers available."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers
        
        classifier = ToxicityClassifier()
        
        # Verify transformers was imported and pipeline created
        mock_import.assert_called_with("transformers")
        mock_transformers.pipeline.assert_called_with(
            "text-classification",
            model="unitary/toxic-bert-base",
            return_all_scores=True
        )
        assert classifier.pipeline is not None

    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    def test_toxicity_classifier_no_transformers(self, mock_import):
        """Test ToxicityClassifier when transformers is not available."""
        mock_import.side_effect = ImportError("transformers not available")
        
        classifier = ToxicityClassifier()
        
        # Should still initialize but with no pipeline
        assert classifier.pipeline is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = ToxicityClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "non_toxic"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = ToxicityClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "non_toxic"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    async def test_classify_with_pipeline_toxic(self, mock_import):
        """Test toxic classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "TOXIC", "score": 0.95},
                {"label": "NON_TOXIC", "score": 0.05}
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = ToxicityClassifier()
        result = await classifier.classify_async("You are stupid and worthless!")
        
        assert result.label == "toxic"
        assert result.confidence == 0.95
        assert result.metadata["method"] == "transformers_pipeline"
        assert "toxic_score" in result.metadata
        assert "non_toxic_score" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    async def test_classify_with_pipeline_non_toxic(self, mock_import):
        """Test non-toxic classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "TOXIC", "score": 0.1},
                {"label": "NON_TOXIC", "score": 0.9}
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = ToxicityClassifier()
        result = await classifier.classify_async("Have a great day!")
        
        assert result.label == "non_toxic"
        assert result.confidence == 0.9
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    async def test_classify_below_threshold(self, mock_import):
        """Test classification when toxic score is below threshold."""
        # Mock transformers with toxic score below threshold
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "TOXIC", "score": 0.5},  # Below default threshold of 0.7
                {"label": "NON_TOXIC", "score": 0.5}
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = ToxicityClassifier()
        result = await classifier.classify_async("This is borderline text")
        
        assert result.label == "non_toxic"
        assert result.confidence == 0.5
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    async def test_classify_with_rules_fallback(self):
        """Test classification using rule-based fallback method."""
        # Create classifier with no transformers
        with patch('sifaka.classifiers.toxicity.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = ToxicityClassifier()
        
        # Test toxic text with profanity
        toxic_text = "You are an idiot and a moron!"
        result = await classifier.classify_async(toxic_text)
        
        assert result.label == "toxic"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["profanity_count"] > 0

    @pytest.mark.asyncio
    async def test_classify_non_toxic_rules_fallback(self):
        """Test classification of non-toxic text using rule-based fallback."""
        with patch('sifaka.classifiers.toxicity.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = ToxicityClassifier()
        
        # Test non-toxic text
        non_toxic_text = "Have a wonderful day!"
        result = await classifier.classify_async(non_toxic_text)
        
        assert result.label == "non_toxic"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["profanity_count"] == 0

    @pytest.mark.asyncio
    async def test_classify_caps_detection_fallback(self):
        """Test classification with excessive capitalization using fallback."""
        with patch('sifaka.classifiers.toxicity.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = ToxicityClassifier()
        
        # Test text with excessive caps
        caps_text = "YOU ARE TERRIBLE AND AWFUL!!!"
        result = await classifier.classify_async(caps_text)
        
        assert result.label == "toxic"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["caps_ratio"] > 0.5

    def test_get_classes(self):
        """Test get_classes method returns correct labels."""
        classifier = ToxicityClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "toxic" in classes
        assert "non_toxic" in classes
        assert len(classes) == 2

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = ToxicityClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedToxicityClassifier:
    """Test CachedToxicityClassifier implementation."""

    def test_cached_toxicity_classifier_initialization(self):
        """Test CachedToxicityClassifier initialization."""
        classifier = CachedToxicityClassifier(cache_size=64)
        
        assert classifier.name == "cached_toxicity"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_toxicity_classifier_caching(self):
        """Test that CachedToxicityClassifier properly caches results."""
        classifier = CachedToxicityClassifier()
        
        # First call
        result1 = classifier.classify("Test message")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("Test message")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_toxicity_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedToxicityClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "toxic" in classes
        assert "non_toxic" in classes


class TestToxicityClassifierFactory:
    """Test toxicity classifier factory function."""

    def test_create_toxicity_classifier_default(self):
        """Test creating toxicity classifier with default parameters."""
        classifier = create_toxicity_classifier()
        
        assert isinstance(classifier, ToxicityClassifier)
        assert classifier.model_name == "unitary/toxic-bert-base"
        assert classifier.threshold == 0.7

    def test_create_toxicity_classifier_cached(self):
        """Test creating cached toxicity classifier."""
        classifier = create_toxicity_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedToxicityClassifier)
        assert classifier.cache_size == 64

    def test_create_toxicity_classifier_custom_params(self):
        """Test creating toxicity classifier with custom parameters."""
        classifier = create_toxicity_classifier(
            model_name="custom/toxicity-model",
            threshold=0.8
        )
        
        assert isinstance(classifier, ToxicityClassifier)
        assert classifier.model_name == "custom/toxicity-model"
        assert classifier.threshold == 0.8


class TestToxicityClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = ToxicityClassifier()
        long_text = "This is a test message. " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = ToxicityClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = ToxicityClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.toxicity.importlib.import_module')
    async def test_pipeline_error_fallback(self, mock_import):
        """Test fallback when pipeline raises an error."""
        # Mock transformers with failing pipeline
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline failed")
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = ToxicityClassifier()
        result = await classifier.classify_async("Test message")
        
        # Should fallback to rule-based analysis
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "rule_based"
