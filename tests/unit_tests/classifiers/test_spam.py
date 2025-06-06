"""Comprehensive unit tests for SpamClassifier.

This module tests the spam detection classifier including:
- Spam detection using machine learning models
- Rule-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.spam import (
    SpamClassifier,
    CachedSpamClassifier,
    create_spam_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestSpamClassifier:
    """Test SpamClassifier implementation."""

    def test_spam_classifier_initialization(self):
        """Test SpamClassifier initialization with default parameters."""
        classifier = SpamClassifier()
        
        assert classifier.name == "spam_detection"
        assert "spam" in classifier.description.lower()
        assert classifier.threshold == 0.5

    def test_spam_classifier_custom_parameters(self):
        """Test SpamClassifier initialization with custom parameters."""
        classifier = SpamClassifier(
            threshold=0.7,
            name="custom_spam",
            description="Custom spam detector"
        )
        
        assert classifier.name == "custom_spam"
        assert classifier.description == "Custom spam detector"
        assert classifier.threshold == 0.7

    @patch('sifaka.classifiers.spam.importlib.import_module')
    def test_spam_classifier_sklearn_initialization(self, mock_import):
        """Test SpamClassifier initialization with sklearn available."""
        # Mock sklearn modules
        mock_sklearn = Mock()
        mock_feature_extraction = Mock()
        mock_naive_bayes = Mock()
        
        def side_effect(module_name):
            if module_name == "sklearn.feature_extraction.text":
                return mock_feature_extraction
            elif module_name == "sklearn.naive_bayes":
                return mock_naive_bayes
            return mock_sklearn
        
        mock_import.side_effect = side_effect
        
        classifier = SpamClassifier()
        
        # Verify sklearn modules were imported
        assert mock_import.call_count >= 2

    @patch('sifaka.classifiers.spam.importlib.import_module')
    def test_spam_classifier_no_sklearn(self, mock_import):
        """Test SpamClassifier when sklearn is not available."""
        mock_import.side_effect = ImportError("sklearn not available")
        
        classifier = SpamClassifier()
        
        # Should still initialize but with no ML models
        assert classifier.vectorizer is None
        assert classifier.model is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = SpamClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "ham"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = SpamClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "ham"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.spam.importlib.import_module')
    async def test_classify_with_ml_model_spam(self, mock_import):
        """Test spam classification using ML model."""
        # Mock sklearn components
        mock_feature_extraction = Mock()
        mock_naive_bayes = Mock()
        
        # Mock vectorizer and model
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
        mock_feature_extraction.TfidfVectorizer.return_value = mock_vectorizer
        
        mock_model = Mock()
        mock_model.predict.return_value = [1]  # Spam
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # [ham_prob, spam_prob]
        mock_naive_bayes.MultinomialNB.return_value = mock_model
        
        def side_effect(module_name):
            if module_name == "sklearn.feature_extraction.text":
                return mock_feature_extraction
            elif module_name == "sklearn.naive_bayes":
                return mock_naive_bayes
            return Mock()
        
        mock_import.side_effect = side_effect
        
        classifier = SpamClassifier()
        # Manually set up the model to simulate training
        classifier.vectorizer = mock_vectorizer
        classifier.model = mock_model
        
        result = await classifier.classify_async("WIN $1000000 NOW!!! CLICK HERE!!!")
        
        assert result.label == "spam"
        assert result.confidence == 0.8
        assert result.metadata["method"] == "machine_learning"

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.spam.importlib.import_module')
    async def test_classify_with_ml_model_ham(self, mock_import):
        """Test ham classification using ML model."""
        # Mock sklearn components
        mock_feature_extraction = Mock()
        mock_naive_bayes = Mock()
        
        # Mock vectorizer and model
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
        mock_feature_extraction.TfidfVectorizer.return_value = mock_vectorizer
        
        mock_model = Mock()
        mock_model.predict.return_value = [0]  # Ham
        mock_model.predict_proba.return_value = [[0.9, 0.1]]  # [ham_prob, spam_prob]
        mock_naive_bayes.MultinomialNB.return_value = mock_model
        
        def side_effect(module_name):
            if module_name == "sklearn.feature_extraction.text":
                return mock_feature_extraction
            elif module_name == "sklearn.naive_bayes":
                return mock_naive_bayes
            return Mock()
        
        mock_import.side_effect = side_effect
        
        classifier = SpamClassifier()
        # Manually set up the model to simulate training
        classifier.vectorizer = mock_vectorizer
        classifier.model = mock_model
        
        result = await classifier.classify_async("Hi John, can we meet tomorrow?")
        
        assert result.label == "ham"
        assert result.confidence == 0.9
        assert result.metadata["method"] == "machine_learning"

    @pytest.mark.asyncio
    async def test_classify_with_rules_fallback_spam(self):
        """Test spam classification using rule-based fallback."""
        # Create classifier with no ML models
        with patch('sifaka.classifiers.spam.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No sklearn")
            classifier = SpamClassifier()
        
        # Test obvious spam text
        spam_text = "URGENT!!! WIN $1000000 NOW!!! CLICK HERE!!! FREE MONEY!!!"
        result = await classifier.classify_async(spam_text)
        
        assert result.label == "spam"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["indicator_count"] > 0
        assert result.metadata["pattern_count"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_rules_fallback_ham(self):
        """Test ham classification using rule-based fallback."""
        with patch('sifaka.classifiers.spam.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No sklearn")
            classifier = SpamClassifier()
        
        # Test normal text
        ham_text = "Hi John, can we schedule a meeting for tomorrow at 2 PM?"
        result = await classifier.classify_async(ham_text)
        
        assert result.label == "ham"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["indicator_count"] == 0

    @pytest.mark.asyncio
    async def test_classify_caps_detection_fallback(self):
        """Test classification with excessive capitalization using fallback."""
        with patch('sifaka.classifiers.spam.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No sklearn")
            classifier = SpamClassifier()
        
        # Test text with excessive caps
        caps_text = "BUY NOW!!! AMAZING DEAL!!! LIMITED TIME!!!"
        result = await classifier.classify_async(caps_text)
        
        assert result.label == "spam"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["caps_ratio"] > 0.5

    @pytest.mark.asyncio
    async def test_classify_exclamation_detection_fallback(self):
        """Test classification with excessive exclamation marks using fallback."""
        with patch('sifaka.classifiers.spam.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No sklearn")
            classifier = SpamClassifier()
        
        # Test text with many exclamation marks
        exclamation_text = "Amazing deal!!! Buy now!!! Don't miss out!!!"
        result = await classifier.classify_async(exclamation_text)
        
        assert result.label == "spam"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["exclamation_count"] > 3

    def test_get_classes(self):
        """Test get_classes method returns correct labels."""
        classifier = SpamClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "spam" in classes
        assert "ham" in classes
        assert len(classes) == 2

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = SpamClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedSpamClassifier:
    """Test CachedSpamClassifier implementation."""

    def test_cached_spam_classifier_initialization(self):
        """Test CachedSpamClassifier initialization."""
        classifier = CachedSpamClassifier(cache_size=64)
        
        assert classifier.name == "cached_spam"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_spam_classifier_caching(self):
        """Test that CachedSpamClassifier properly caches results."""
        classifier = CachedSpamClassifier()
        
        # First call
        result1 = classifier.classify("Test message")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("Test message")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_spam_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedSpamClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "spam" in classes
        assert "ham" in classes


class TestSpamClassifierFactory:
    """Test spam classifier factory function."""

    def test_create_spam_classifier_default(self):
        """Test creating spam classifier with default parameters."""
        classifier = create_spam_classifier()
        
        assert isinstance(classifier, SpamClassifier)
        assert classifier.threshold == 0.5

    def test_create_spam_classifier_cached(self):
        """Test creating cached spam classifier."""
        classifier = create_spam_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedSpamClassifier)
        assert classifier.cache_size == 64

    def test_create_spam_classifier_custom_params(self):
        """Test creating spam classifier with custom parameters."""
        classifier = create_spam_classifier(threshold=0.7)
        
        assert isinstance(classifier, SpamClassifier)
        assert classifier.threshold == 0.7


class TestSpamClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = SpamClassifier()
        long_text = "This is a test message. " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = SpamClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = SpamClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_numbers_and_symbols(self):
        """Test classification with numbers and symbols."""
        classifier = SpamClassifier()
        symbols_text = "$$$$ 1000000 $$$ FREE $$$ MONEY $$$"
        
        result = await classifier.classify_async(symbols_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0
