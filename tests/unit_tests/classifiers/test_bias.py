"""Comprehensive unit tests for BiasClassifier.

This module tests the bias detection classifier including:
- Bias detection using detoxify library
- Fallback to transformers pipeline
- Simple heuristic-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.bias import (
    BiasClassifier,
    CachedBiasClassifier,
    create_bias_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestBiasClassifier:
    """Test BiasClassifier implementation."""

    def test_bias_classifier_initialization(self):
        """Test BiasClassifier initialization with default parameters."""
        classifier = BiasClassifier()
        
        assert classifier.name == "bias_detection"
        assert "bias" in classifier.description.lower()
        assert classifier.model_name == "unitary/unbiased-toxic-roberta"
        assert classifier.threshold == 0.7
        assert classifier.detoxify_model_name == "unbiased"

    def test_bias_classifier_custom_parameters(self):
        """Test BiasClassifier initialization with custom parameters."""
        classifier = BiasClassifier(
            model_name="custom/model",
            threshold=0.8,
            detoxify_model="original",
            name="custom_bias",
            description="Custom bias detector"
        )
        
        assert classifier.name == "custom_bias"
        assert classifier.description == "Custom bias detector"
        assert classifier.model_name == "custom/model"
        assert classifier.threshold == 0.8
        assert classifier.detoxify_model_name == "original"

    @patch('sifaka.classifiers.bias.importlib.import_module')
    def test_bias_classifier_detoxify_initialization(self, mock_import):
        """Test BiasClassifier initialization with detoxify available."""
        # Mock detoxify module
        mock_detoxify = Mock()
        mock_detoxify.Detoxify = Mock()
        mock_import.return_value = mock_detoxify
        
        classifier = BiasClassifier()
        
        # Verify detoxify was imported and initialized
        mock_import.assert_called_with("detoxify")
        mock_detoxify.Detoxify.assert_called_with("unbiased")
        assert classifier.detoxify_model is not None

    @patch('sifaka.classifiers.bias.importlib.import_module')
    def test_bias_classifier_transformers_fallback(self, mock_import):
        """Test BiasClassifier fallback to transformers when detoxify unavailable."""
        # Mock detoxify import failure, transformers success
        def side_effect(module_name):
            if module_name == "detoxify":
                raise ImportError("detoxify not available")
            elif module_name == "transformers":
                mock_transformers = Mock()
                mock_transformers.pipeline = Mock()
                return mock_transformers
            return Mock()
        
        mock_import.side_effect = side_effect
        
        classifier = BiasClassifier()
        
        # Verify fallback to transformers
        assert classifier.detoxify_model is None
        assert classifier.pipeline is not None

    @patch('sifaka.classifiers.bias.importlib.import_module')
    def test_bias_classifier_no_dependencies(self, mock_import):
        """Test BiasClassifier when no dependencies are available."""
        mock_import.side_effect = ImportError("No dependencies available")
        
        classifier = BiasClassifier()
        
        # Should still initialize but with no external models
        assert classifier.detoxify_model is None
        assert classifier.pipeline is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = BiasClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "unbiased"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = BiasClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "unbiased"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.bias.importlib.import_module')
    async def test_classify_with_detoxify(self, mock_import):
        """Test classification using detoxify model."""
        # Mock detoxify
        mock_detoxify = Mock()
        mock_instance = Mock()
        mock_instance.predict.return_value = {
            "toxicity": 0.1,
            "severe_toxicity": 0.05,
            "obscene": 0.02,
            "threat": 0.01,
            "insult": 0.03,
            "identity_attack": 0.02,
        }
        mock_detoxify.Detoxify.return_value = mock_instance
        mock_import.return_value = mock_detoxify
        
        classifier = BiasClassifier()
        result = await classifier.classify_async("This is a test message")
        
        assert result.label in ["biased", "unbiased"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.metadata["method"] == "detoxify"
        assert "toxicity_scores" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.bias.importlib.import_module')
    async def test_classify_with_pipeline(self, mock_import):
        """Test classification using transformers pipeline."""
        # Mock transformers but not detoxify
        def side_effect(module_name):
            if module_name == "detoxify":
                raise ImportError("detoxify not available")
            elif module_name == "transformers":
                mock_transformers = Mock()
                mock_pipeline = Mock()
                mock_pipeline.return_value = [
                    {"label": "BIASED", "score": 0.85},
                    {"label": "UNBIASED", "score": 0.15}
                ]
                mock_transformers.pipeline.return_value = mock_pipeline
                return mock_transformers
            return Mock()
        
        mock_import.side_effect = side_effect
        
        classifier = BiasClassifier()
        result = await classifier.classify_async("This is a test message")
        
        assert result.label in ["biased", "unbiased"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    async def test_classify_with_fallback(self):
        """Test classification using simple fallback method."""
        # Create classifier with no external dependencies
        with patch('sifaka.classifiers.bias.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = BiasClassifier()
        
        # Test biased text
        biased_text = "All women are bad drivers"
        result = await classifier.classify_async(biased_text)
        
        assert result.label == "biased"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "simple_heuristics"
        assert result.metadata["bias_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_unbiased_fallback(self):
        """Test classification of unbiased text using fallback."""
        with patch('sifaka.classifiers.bias.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = BiasClassifier()
        
        # Test unbiased text
        unbiased_text = "The weather is nice today"
        result = await classifier.classify_async(unbiased_text)
        
        assert result.label == "unbiased"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "simple_heuristics"
        assert result.metadata["bias_indicators"] == 0

    def test_get_classes(self):
        """Test get_classes method returns correct labels."""
        classifier = BiasClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "biased" in classes
        assert "unbiased" in classes
        assert len(classes) == 2

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = BiasClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedBiasClassifier:
    """Test CachedBiasClassifier implementation."""

    def test_cached_bias_classifier_initialization(self):
        """Test CachedBiasClassifier initialization."""
        classifier = CachedBiasClassifier(cache_size=64)
        
        assert classifier.name == "cached_bias"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_bias_classifier_caching(self):
        """Test that CachedBiasClassifier properly caches results."""
        classifier = CachedBiasClassifier()
        
        # First call
        result1 = classifier.classify("Test message")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("Test message")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_bias_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedBiasClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "biased" in classes
        assert "unbiased" in classes


class TestBiasClassifierFactory:
    """Test bias classifier factory function."""

    def test_create_bias_classifier_default(self):
        """Test creating bias classifier with default parameters."""
        classifier = create_bias_classifier()
        
        assert isinstance(classifier, BiasClassifier)
        assert classifier.model_name == "unitary/unbiased-toxic-roberta"
        assert classifier.threshold == 0.7

    def test_create_bias_classifier_cached(self):
        """Test creating cached bias classifier."""
        classifier = create_bias_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedBiasClassifier)
        assert classifier.cache_size == 64

    def test_create_bias_classifier_custom_params(self):
        """Test creating bias classifier with custom parameters."""
        classifier = create_bias_classifier(
            model_name="custom/model",
            threshold=0.8,
            detoxify_model="original"
        )
        
        assert isinstance(classifier, BiasClassifier)
        assert classifier.model_name == "custom/model"
        assert classifier.threshold == 0.8
        assert classifier.detoxify_model_name == "original"


class TestBiasClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = BiasClassifier()
        long_text = "This is a test message. " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in ["biased", "unbiased"]
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = BiasClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in ["biased", "unbiased"]
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = BiasClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in ["biased", "unbiased"]
        assert 0.0 <= result.confidence <= 1.0
