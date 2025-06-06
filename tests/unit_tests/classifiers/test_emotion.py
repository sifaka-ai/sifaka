"""Comprehensive unit tests for EmotionClassifier.

This module tests the emotion detection classifier including:
- Emotion detection using transformers pipeline
- Simple keyword-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.emotion import (
    EmotionClassifier,
    CachedEmotionClassifier,
    create_emotion_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestEmotionClassifier:
    """Test EmotionClassifier implementation."""

    def test_emotion_classifier_initialization(self):
        """Test EmotionClassifier initialization with default parameters."""
        classifier = EmotionClassifier()
        
        assert classifier.name == "emotion_detection"
        assert "emotion" in classifier.description.lower()
        assert classifier.model_name == "j-hartmann/emotion-english-distilroberta-base"
        assert classifier.threshold == 0.3

    def test_emotion_classifier_custom_parameters(self):
        """Test EmotionClassifier initialization with custom parameters."""
        classifier = EmotionClassifier(
            model_name="custom/emotion-model",
            threshold=0.5,
            name="custom_emotion",
            description="Custom emotion detector"
        )
        
        assert classifier.name == "custom_emotion"
        assert classifier.description == "Custom emotion detector"
        assert classifier.model_name == "custom/emotion-model"
        assert classifier.threshold == 0.5

    @patch('sifaka.classifiers.emotion.importlib.import_module')
    def test_emotion_classifier_transformers_initialization(self, mock_import):
        """Test EmotionClassifier initialization with transformers available."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers
        
        classifier = EmotionClassifier()
        
        # Verify transformers was imported and pipeline created
        mock_import.assert_called_with("transformers")
        mock_transformers.pipeline.assert_called_with(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        assert classifier.pipeline is not None

    @patch('sifaka.classifiers.emotion.importlib.import_module')
    def test_emotion_classifier_no_transformers(self, mock_import):
        """Test EmotionClassifier when transformers is not available."""
        mock_import.side_effect = ImportError("transformers not available")
        
        classifier = EmotionClassifier()
        
        # Should still initialize but with no pipeline
        assert classifier.pipeline is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = EmotionClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "neutral"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = EmotionClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "neutral"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.emotion.importlib.import_module')
    async def test_classify_with_pipeline(self, mock_import):
        """Test classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "joy", "score": 0.85},
                {"label": "sadness", "score": 0.10},
                {"label": "anger", "score": 0.05}
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = EmotionClassifier()
        result = await classifier.classify_async("I'm so happy today!")
        
        assert result.label == "joy"
        assert result.confidence == 0.85
        assert result.metadata["method"] == "transformers_pipeline"
        assert "all_emotions" in result.metadata
        assert "detected_emotions" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.emotion.importlib.import_module')
    async def test_classify_below_threshold(self, mock_import):
        """Test classification when all emotions are below threshold."""
        # Mock transformers with low confidence scores
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "joy", "score": 0.2},
                {"label": "sadness", "score": 0.15},
                {"label": "anger", "score": 0.1}
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = EmotionClassifier(threshold=0.3)
        result = await classifier.classify_async("This is neutral text")
        
        assert result.label == "neutral"
        assert result.confidence > 0
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    async def test_classify_with_fallback(self):
        """Test classification using simple fallback method."""
        # Create classifier with no transformers
        with patch('sifaka.classifiers.emotion.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = EmotionClassifier()
        
        # Test happy text
        happy_text = "I'm so excited and joyful!"
        result = await classifier.classify_async(happy_text)
        
        assert result.label == "joy"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "keyword_analysis"
        assert result.metadata["keyword_matches"] > 0

    @pytest.mark.asyncio
    async def test_classify_sad_fallback(self):
        """Test classification of sad text using fallback."""
        with patch('sifaka.classifiers.emotion.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = EmotionClassifier()
        
        # Test sad text
        sad_text = "I feel devastated and heartbroken"
        result = await classifier.classify_async(sad_text)
        
        assert result.label == "sadness"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "keyword_analysis"

    @pytest.mark.asyncio
    async def test_classify_angry_fallback(self):
        """Test classification of angry text using fallback."""
        with patch('sifaka.classifiers.emotion.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = EmotionClassifier()
        
        # Test angry text
        angry_text = "I'm furious and outraged!"
        result = await classifier.classify_async(angry_text)
        
        assert result.label == "anger"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "keyword_analysis"

    @pytest.mark.asyncio
    async def test_classify_neutral_fallback(self):
        """Test classification of neutral text using fallback."""
        with patch('sifaka.classifiers.emotion.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = EmotionClassifier()
        
        # Test neutral text
        neutral_text = "The weather is nice today"
        result = await classifier.classify_async(neutral_text)
        
        assert result.label == "neutral"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "keyword_analysis"
        assert result.metadata["keyword_matches"] == 0

    def test_get_classes(self):
        """Test get_classes method returns correct labels."""
        classifier = EmotionClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        expected_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        for emotion in expected_emotions:
            assert emotion in classes

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = EmotionClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedEmotionClassifier:
    """Test CachedEmotionClassifier implementation."""

    def test_cached_emotion_classifier_initialization(self):
        """Test CachedEmotionClassifier initialization."""
        classifier = CachedEmotionClassifier(cache_size=64)
        
        assert classifier.name == "cached_emotion"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_emotion_classifier_caching(self):
        """Test that CachedEmotionClassifier properly caches results."""
        classifier = CachedEmotionClassifier()
        
        # First call
        result1 = classifier.classify("I'm happy!")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("I'm happy!")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_emotion_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedEmotionClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "joy" in classes
        assert "sadness" in classes
        assert "neutral" in classes


class TestEmotionClassifierFactory:
    """Test emotion classifier factory function."""

    def test_create_emotion_classifier_default(self):
        """Test creating emotion classifier with default parameters."""
        classifier = create_emotion_classifier()
        
        assert isinstance(classifier, EmotionClassifier)
        assert classifier.model_name == "j-hartmann/emotion-english-distilroberta-base"
        assert classifier.threshold == 0.3

    def test_create_emotion_classifier_cached(self):
        """Test creating cached emotion classifier."""
        classifier = create_emotion_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedEmotionClassifier)
        assert classifier.cache_size == 64

    def test_create_emotion_classifier_custom_params(self):
        """Test creating emotion classifier with custom parameters."""
        classifier = create_emotion_classifier(
            model_name="custom/emotion-model",
            threshold=0.5
        )
        
        assert isinstance(classifier, EmotionClassifier)
        assert classifier.model_name == "custom/emotion-model"
        assert classifier.threshold == 0.5


class TestEmotionClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = EmotionClassifier()
        long_text = "I'm so happy today! " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = EmotionClassifier()
        special_text = "Happy! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = EmotionClassifier()
        unicode_text = "Happy 世界 🌍 😊 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.emotion.importlib.import_module')
    async def test_pipeline_error_fallback(self, mock_import):
        """Test fallback when pipeline raises an error."""
        # Mock transformers with failing pipeline
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline failed")
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = EmotionClassifier()
        result = await classifier.classify_async("Test message")
        
        # Should fallback to keyword analysis
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "keyword_analysis"
