"""Comprehensive unit tests for LanguageClassifier.

This module tests the language detection classifier including:
- Language detection using langdetect library
- Pattern-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.language import (
    LanguageClassifier,
    CachedLanguageClassifier,
    create_language_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestLanguageClassifier:
    """Test LanguageClassifier implementation."""

    def test_language_classifier_initialization(self):
        """Test LanguageClassifier initialization with default parameters."""
        classifier = LanguageClassifier()
        
        assert classifier.name == "language_detection"
        assert "language" in classifier.description.lower()
        assert classifier.min_confidence == 0.7

    def test_language_classifier_custom_parameters(self):
        """Test LanguageClassifier initialization with custom parameters."""
        classifier = LanguageClassifier(
            min_confidence=0.8,
            name="custom_language",
            description="Custom language detector"
        )
        
        assert classifier.name == "custom_language"
        assert classifier.description == "Custom language detector"
        assert classifier.min_confidence == 0.8

    @patch('sifaka.classifiers.language.importlib.import_module')
    def test_language_classifier_langdetect_initialization(self, mock_import):
        """Test LanguageClassifier initialization with langdetect available."""
        # Mock langdetect module
        mock_langdetect = Mock()
        mock_langdetect.DetectorFactory = Mock()
        mock_import.return_value = mock_langdetect
        
        classifier = LanguageClassifier()
        
        # Verify langdetect was imported
        mock_import.assert_called_with("langdetect")
        assert classifier.detector is not None

    @patch('sifaka.classifiers.language.importlib.import_module')
    def test_language_classifier_no_langdetect(self, mock_import):
        """Test LanguageClassifier when langdetect is not available."""
        mock_import.side_effect = ImportError("langdetect not available")
        
        classifier = LanguageClassifier()
        
        # Should still initialize but with no detector
        assert classifier.detector is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = LanguageClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = LanguageClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.language.importlib.import_module')
    async def test_classify_with_langdetect_english(self, mock_import):
        """Test English language detection using langdetect."""
        # Mock langdetect
        mock_langdetect = Mock()
        mock_detector = Mock()
        
        # Mock language probability objects
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "en"
        mock_lang_prob.prob = 0.95
        
        mock_detector.detect_langs.return_value = [mock_lang_prob]
        mock_langdetect.DetectorFactory.create.return_value = mock_detector
        mock_import.return_value = mock_langdetect
        
        classifier = LanguageClassifier()
        result = await classifier.classify_async("Hello, how are you doing today?")
        
        assert result.label == "en"
        assert result.confidence == 0.95
        assert result.metadata["method"] == "langdetect"
        assert result.metadata["language_name"] == "English"

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.language.importlib.import_module')
    async def test_classify_with_langdetect_spanish(self, mock_import):
        """Test Spanish language detection using langdetect."""
        # Mock langdetect
        mock_langdetect = Mock()
        mock_detector = Mock()
        
        # Mock language probability objects
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "es"
        mock_lang_prob.prob = 0.88
        
        mock_detector.detect_langs.return_value = [mock_lang_prob]
        mock_langdetect.DetectorFactory.create.return_value = mock_detector
        mock_import.return_value = mock_langdetect
        
        classifier = LanguageClassifier()
        result = await classifier.classify_async("Hola, ¿cómo estás hoy?")
        
        assert result.label == "es"
        assert result.confidence == 0.88
        assert result.metadata["method"] == "langdetect"
        assert result.metadata["language_name"] == "Spanish"

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.language.importlib.import_module')
    async def test_classify_below_confidence_threshold(self, mock_import):
        """Test classification when confidence is below threshold."""
        # Mock langdetect with low confidence
        mock_langdetect = Mock()
        mock_detector = Mock()
        
        mock_lang_prob = Mock()
        mock_lang_prob.lang = "en"
        mock_lang_prob.prob = 0.5  # Below default threshold of 0.7
        
        mock_detector.detect_langs.return_value = [mock_lang_prob]
        mock_langdetect.DetectorFactory.create.return_value = mock_detector
        mock_import.return_value = mock_langdetect
        
        classifier = LanguageClassifier()
        result = await classifier.classify_async("Short text")
        
        assert result.label == "unknown"
        assert result.metadata["method"] == "langdetect"

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_english(self):
        """Test English detection using pattern-based fallback."""
        # Create classifier with no langdetect
        with patch('sifaka.classifiers.language.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No langdetect")
            classifier = LanguageClassifier()
        
        # Test English text with common English words
        english_text = "The quick brown fox jumps over the lazy dog"
        result = await classifier.classify_async(english_text)
        
        assert result.label == "en"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"
        assert result.metadata["english_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_spanish(self):
        """Test Spanish detection using pattern-based fallback."""
        with patch('sifaka.classifiers.language.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No langdetect")
            classifier = LanguageClassifier()
        
        # Test Spanish text with Spanish indicators
        spanish_text = "El rápido zorro marrón salta sobre el perro perezoso"
        result = await classifier.classify_async(spanish_text)
        
        assert result.label == "es"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"
        assert result.metadata["spanish_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_unknown(self):
        """Test unknown language detection using pattern-based fallback."""
        with patch('sifaka.classifiers.language.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No langdetect")
            classifier = LanguageClassifier()
        
        # Test text without clear language indicators
        unknown_text = "xyz abc def 123 456"
        result = await classifier.classify_async(unknown_text)
        
        assert result.label == "unknown"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"

    def test_get_language_name(self):
        """Test get_language_name method."""
        classifier = LanguageClassifier()
        
        assert classifier.get_language_name("en") == "English"
        assert classifier.get_language_name("es") == "Spanish"
        assert classifier.get_language_name("fr") == "French"
        assert classifier.get_language_name("de") == "German"
        assert classifier.get_language_name("unknown") == "Unknown"
        assert classifier.get_language_name("xyz") == "Unknown"

    def test_get_classes(self):
        """Test get_classes method returns supported languages."""
        classifier = LanguageClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "en" in classes
        assert "es" in classes
        assert "fr" in classes
        assert "de" in classes
        assert "unknown" in classes

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = LanguageClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedLanguageClassifier:
    """Test CachedLanguageClassifier implementation."""

    def test_cached_language_classifier_initialization(self):
        """Test CachedLanguageClassifier initialization."""
        classifier = CachedLanguageClassifier(cache_size=64)
        
        assert classifier.name == "cached_language"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_language_classifier_caching(self):
        """Test that CachedLanguageClassifier properly caches results."""
        classifier = CachedLanguageClassifier()
        
        # First call
        result1 = classifier.classify("Hello world")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("Hello world")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_language_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedLanguageClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "en" in classes
        assert "unknown" in classes


class TestLanguageClassifierFactory:
    """Test language classifier factory function."""

    def test_create_language_classifier_default(self):
        """Test creating language classifier with default parameters."""
        classifier = create_language_classifier()
        
        assert isinstance(classifier, LanguageClassifier)
        assert classifier.min_confidence == 0.7

    def test_create_language_classifier_cached(self):
        """Test creating cached language classifier."""
        classifier = create_language_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedLanguageClassifier)
        assert classifier.cache_size == 64

    def test_create_language_classifier_custom_params(self):
        """Test creating language classifier with custom parameters."""
        classifier = create_language_classifier(min_confidence=0.8)
        
        assert isinstance(classifier, LanguageClassifier)
        assert classifier.min_confidence == 0.8


class TestLanguageClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = LanguageClassifier()
        long_text = "This is a test message in English. " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = LanguageClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = LanguageClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_very_short_text(self):
        """Test classification with very short text."""
        classifier = LanguageClassifier()
        short_text = "Hi"
        
        result = await classifier.classify_async(short_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.language.importlib.import_module')
    async def test_langdetect_error_fallback(self, mock_import):
        """Test fallback when langdetect raises an error."""
        # Mock langdetect with failing detector
        mock_langdetect = Mock()
        mock_detector = Mock()
        mock_detector.detect_langs.side_effect = Exception("Detection failed")
        mock_langdetect.DetectorFactory.create.return_value = mock_detector
        mock_import.return_value = mock_langdetect
        
        classifier = LanguageClassifier()
        result = await classifier.classify_async("Test message")
        
        # Should fallback to pattern-based detection
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "pattern_based"
