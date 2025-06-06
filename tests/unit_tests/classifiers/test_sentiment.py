"""Comprehensive unit tests for SentimentClassifier.

This module tests the sentiment analysis classifier including:
- Sentiment analysis using transformers pipeline
- TextBlob fallback
- Lexicon-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.sentiment import (
    SentimentClassifier,
    CachedSentimentClassifier,
    create_sentiment_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestSentimentClassifier:
    """Test SentimentClassifier implementation."""

    def test_sentiment_classifier_initialization(self):
        """Test SentimentClassifier initialization with default parameters."""
        classifier = SentimentClassifier()

        assert classifier.name == "pretrained_sentiment"
        assert "sentiment" in classifier.description.lower()
        assert classifier.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert classifier.positive_threshold == 0.1
        assert classifier.negative_threshold == -0.1

    def test_sentiment_classifier_custom_parameters(self):
        """Test SentimentClassifier initialization with custom parameters."""
        classifier = SentimentClassifier(
            model_name="custom/sentiment-model",
            positive_threshold=0.7,
            negative_threshold=0.8,
            name="custom_sentiment",
            description="Custom sentiment analyzer",
        )

        assert classifier.name == "custom_sentiment"
        assert classifier.description == "Custom sentiment analyzer"
        assert classifier.model_name == "custom/sentiment-model"
        assert classifier.positive_threshold == 0.7
        assert classifier.negative_threshold == 0.8

    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    def test_sentiment_classifier_transformers_initialization(self, mock_import):
        """Test SentimentClassifier initialization with transformers available."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers

        classifier = SentimentClassifier()

        # Verify transformers was imported and pipeline created
        mock_import.assert_called_with("transformers")
        mock_transformers.pipeline.assert_called_with(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True,
        )
        assert classifier.pipeline is not None

    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    def test_sentiment_classifier_textblob_fallback(self, mock_import):
        """Test SentimentClassifier fallback to TextBlob when transformers unavailable."""

        # Mock transformers import failure, TextBlob success
        def side_effect(module_name):
            if module_name == "transformers":
                raise ImportError("transformers not available")
            elif module_name == "textblob":
                mock_textblob = Mock()
                return mock_textblob
            return Mock()

        mock_import.side_effect = side_effect

        classifier = SentimentClassifier()

        # Verify fallback to TextBlob
        assert classifier.pipeline is None
        assert classifier.textblob is not None

    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    def test_sentiment_classifier_no_dependencies(self, mock_import):
        """Test SentimentClassifier when no dependencies are available."""
        mock_import.side_effect = ImportError("No dependencies available")

        classifier = SentimentClassifier()

        # Should still initialize but with no external models
        assert classifier.pipeline is None
        assert classifier.textblob is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = SentimentClassifier()

        result = await classifier.classify_async("")

        assert result.label == "neutral"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = SentimentClassifier()

        result = await classifier.classify_async("   \n\t   ")

        assert result.label == "neutral"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    async def test_classify_with_pipeline_positive(self, mock_import):
        """Test positive sentiment classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "LABEL_2", "score": 0.85},  # Positive
                {"label": "LABEL_1", "score": 0.10},  # Neutral
                {"label": "LABEL_0", "score": 0.05},  # Negative
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = SentimentClassifier()
        result = await classifier.classify_async("I love this product!")

        assert result.label == "positive"
        assert result.confidence == 0.85
        assert result.metadata["method"] == "transformers_pipeline"
        assert "all_scores" in result.metadata

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    async def test_classify_with_pipeline_negative(self, mock_import):
        """Test negative sentiment classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "LABEL_0", "score": 0.90},  # Negative
                {"label": "LABEL_1", "score": 0.08},  # Neutral
                {"label": "LABEL_2", "score": 0.02},  # Positive
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = SentimentClassifier()
        result = await classifier.classify_async("I hate this product!")

        assert result.label == "negative"
        assert result.confidence == 0.90
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    async def test_classify_with_pipeline_neutral(self, mock_import):
        """Test neutral sentiment classification using transformers pipeline."""
        # Mock transformers with low confidence scores
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [
                {"label": "LABEL_2", "score": 0.4},  # Positive (below threshold)
                {"label": "LABEL_1", "score": 0.35},  # Neutral
                {"label": "LABEL_0", "score": 0.25},  # Negative (below threshold)
            ]
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = SentimentClassifier()
        result = await classifier.classify_async("This is okay.")

        assert result.label == "neutral"
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.sentiment.importlib.import_module")
    async def test_classify_with_textblob(self, mock_import):
        """Test classification using TextBlob."""

        # Mock TextBlob but not transformers
        def side_effect(module_name):
            if module_name == "transformers":
                raise ImportError("transformers not available")
            elif module_name == "textblob":
                mock_textblob = Mock()
                mock_blob = Mock()
                mock_blob.sentiment.polarity = 0.8  # Positive sentiment
                mock_textblob.TextBlob.return_value = mock_blob
                return mock_textblob
            return Mock()

        mock_import.side_effect = side_effect

        classifier = SentimentClassifier()
        result = await classifier.classify_async("I love this!")

        assert result.label == "positive"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "textblob"
        assert "polarity" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_with_lexicon_fallback(self):
        """Test classification using lexicon-based fallback."""
        # Create classifier with no external dependencies
        with patch("sifaka.classifiers.sentiment.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = SentimentClassifier()

        # Test positive text
        positive_text = "I love this amazing wonderful fantastic product!"
        result = await classifier.classify_async(positive_text)

        assert result.label == "positive"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "lexicon_based"
        assert result.metadata["positive_words"] > 0

    @pytest.mark.asyncio
    async def test_classify_negative_lexicon_fallback(self):
        """Test negative sentiment classification using lexicon fallback."""
        with patch("sifaka.classifiers.sentiment.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = SentimentClassifier()

        # Test negative text
        negative_text = "I hate this terrible awful horrible product!"
        result = await classifier.classify_async(negative_text)

        assert result.label == "negative"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "lexicon_based"
        assert result.metadata["negative_words"] > 0

    @pytest.mark.asyncio
    async def test_classify_neutral_lexicon_fallback(self):
        """Test neutral sentiment classification using lexicon fallback."""
        with patch("sifaka.classifiers.sentiment.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = SentimentClassifier()

        # Test neutral text
        neutral_text = "The weather is nice today."
        result = await classifier.classify_async(neutral_text)

        assert result.label == "neutral"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "lexicon_based"

    def test_get_classes(self):
        """Test get_classes method returns correct labels."""
        classifier = SentimentClassifier()
        classes = classifier.get_classes()

        assert isinstance(classes, list)
        assert "positive" in classes
        assert "negative" in classes
        assert "neutral" in classes
        assert len(classes) == 3

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = SentimentClassifier()
        result = await classifier.classify_async("Test message")

        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedSentimentClassifier:
    """Test CachedSentimentClassifier implementation."""

    def test_cached_sentiment_classifier_initialization(self):
        """Test CachedSentimentClassifier initialization."""
        classifier = CachedSentimentClassifier(cache_size=64)

        assert classifier.name == "cached_sentiment"
        assert classifier.cache_size == 64
        assert hasattr(classifier, "_classifier")

    def test_cached_sentiment_classifier_caching(self):
        """Test that CachedSentimentClassifier properly caches results."""
        classifier = CachedSentimentClassifier()

        # First call
        result1 = classifier.classify("I love this!")

        # Second call with same text (should use cache)
        result2 = classifier.classify("I love this!")

        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_sentiment_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedSentimentClassifier()
        classes = classifier.get_classes()

        assert isinstance(classes, list)
        assert "positive" in classes
        assert "negative" in classes
        assert "neutral" in classes


class TestSentimentClassifierFactory:
    """Test sentiment classifier factory function."""

    def test_create_sentiment_classifier_default(self):
        """Test creating sentiment classifier with default parameters."""
        classifier = create_sentiment_classifier()

        assert isinstance(classifier, SentimentClassifier)
        assert classifier.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert classifier.positive_threshold == 0.1
        assert classifier.negative_threshold == -0.1

    def test_create_sentiment_classifier_cached(self):
        """Test creating cached sentiment classifier."""
        classifier = create_sentiment_classifier(cached=True, cache_size=64)

        assert isinstance(classifier, CachedSentimentClassifier)
        assert classifier.cache_size == 64

    def test_create_sentiment_classifier_custom_params(self):
        """Test creating sentiment classifier with custom parameters."""
        classifier = create_sentiment_classifier(
            model_name="custom/sentiment-model", positive_threshold=0.7, negative_threshold=0.8
        )

        assert isinstance(classifier, SentimentClassifier)
        assert classifier.model_name == "custom/sentiment-model"
        assert classifier.positive_threshold == 0.7
        assert classifier.negative_threshold == 0.8


class TestSentimentClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = SentimentClassifier()
        long_text = "I love this product! " * 1000

        result = await classifier.classify_async(long_text)

        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = SentimentClassifier()
        special_text = "Great! @#$%^&*()_+ 123 ñáéíóú"

        result = await classifier.classify_async(special_text)

        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = SentimentClassifier()
        unicode_text = "Great 世界 🌍 😊 emoji test"

        result = await classifier.classify_async(unicode_text)

        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0
