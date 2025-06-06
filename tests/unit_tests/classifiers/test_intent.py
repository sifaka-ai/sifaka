"""Comprehensive unit tests for IntentClassifier.

This module tests the intent detection classifier including:
- Intent detection using transformers pipeline
- Pattern-based fallback
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.intent import (
    IntentClassifier,
    CachedIntentClassifier,
    create_intent_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestIntentClassifier:
    """Test IntentClassifier implementation."""

    def test_intent_classifier_initialization(self):
        """Test IntentClassifier initialization with default parameters."""
        classifier = IntentClassifier()
        
        assert classifier.name == "intent_detection"
        assert "intent" in classifier.description.lower()
        assert classifier.model_name == "microsoft/DialoGPT-medium"
        assert classifier.threshold == 0.4
        assert "question" in classifier.intents
        assert "statement" in classifier.intents

    def test_intent_classifier_custom_parameters(self):
        """Test IntentClassifier initialization with custom parameters."""
        classifier = IntentClassifier(
            model_name="custom/intent-model",
            threshold=0.6,
            name="custom_intent",
            description="Custom intent detector"
        )
        
        assert classifier.name == "custom_intent"
        assert classifier.description == "Custom intent detector"
        assert classifier.model_name == "custom/intent-model"
        assert classifier.threshold == 0.6

    @patch('sifaka.classifiers.intent.importlib.import_module')
    def test_intent_classifier_transformers_initialization(self, mock_import):
        """Test IntentClassifier initialization with transformers available."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers
        
        classifier = IntentClassifier()
        
        # Verify transformers was imported and pipeline created
        mock_import.assert_called_with("transformers")
        assert classifier.pipeline is not None

    @patch('sifaka.classifiers.intent.importlib.import_module')
    def test_intent_classifier_no_transformers(self, mock_import):
        """Test IntentClassifier when transformers is not available."""
        mock_import.side_effect = ImportError("transformers not available")
        
        classifier = IntentClassifier()
        
        # Should still initialize but with no pipeline
        assert classifier.pipeline is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = IntentClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = IntentClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.intent.importlib.import_module')
    async def test_classify_with_pipeline_question(self, mock_import):
        """Test question intent classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "QUESTION", "score": 0.85},
            {"label": "STATEMENT", "score": 0.10},
            {"label": "REQUEST", "score": 0.05}
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = IntentClassifier()
        result = await classifier.classify_async("What time is the meeting?")
        
        assert result.label == "question"
        assert result.confidence == 0.85
        assert result.metadata["method"] == "transformers_pipeline"
        assert "all_intents" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.intent.importlib.import_module')
    async def test_classify_with_pipeline_statement(self, mock_import):
        """Test statement intent classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "STATEMENT", "score": 0.90},
            {"label": "QUESTION", "score": 0.05},
            {"label": "REQUEST", "score": 0.05}
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = IntentClassifier()
        result = await classifier.classify_async("The meeting is at 2 PM.")
        
        assert result.label == "statement"
        assert result.confidence == 0.90
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.intent.importlib.import_module')
    async def test_classify_below_threshold(self, mock_import):
        """Test classification when all intents are below threshold."""
        # Mock transformers with low confidence scores
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "QUESTION", "score": 0.3},  # Below threshold of 0.4
            {"label": "STATEMENT", "score": 0.25},
            {"label": "REQUEST", "score": 0.2}
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = IntentClassifier()
        result = await classifier.classify_async("Hmm, maybe...")
        
        assert result.label == "unknown"
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_question(self):
        """Test question intent detection using pattern-based fallback."""
        # Create classifier with no transformers
        with patch('sifaka.classifiers.intent.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = IntentClassifier()
        
        # Test question text
        question_text = "What time is the meeting scheduled for tomorrow?"
        result = await classifier.classify_async(question_text)
        
        assert result.label == "question"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"
        assert result.metadata["question_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_request(self):
        """Test request intent detection using pattern-based fallback."""
        with patch('sifaka.classifiers.intent.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = IntentClassifier()
        
        # Test request text
        request_text = "Please send me the report by end of day."
        result = await classifier.classify_async(request_text)
        
        assert result.label == "request"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"
        assert result.metadata["request_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_greeting(self):
        """Test greeting intent detection using pattern-based fallback."""
        with patch('sifaka.classifiers.intent.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = IntentClassifier()
        
        # Test greeting text
        greeting_text = "Hello there! How are you doing today?"
        result = await classifier.classify_async(greeting_text)
        
        assert result.label == "greeting"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"
        assert result.metadata["greeting_indicators"] > 0

    @pytest.mark.asyncio
    async def test_classify_with_patterns_fallback_statement(self):
        """Test statement intent detection using pattern-based fallback."""
        with patch('sifaka.classifiers.intent.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No transformers")
            classifier = IntentClassifier()
        
        # Test statement text
        statement_text = "The weather is nice today."
        result = await classifier.classify_async(statement_text)
        
        assert result.label == "statement"
        assert result.confidence > 0.5
        assert result.metadata["method"] == "pattern_based"

    def test_get_classes(self):
        """Test get_classes method returns intent types."""
        classifier = IntentClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "question" in classes
        assert "statement" in classes
        assert "request" in classes
        assert "greeting" in classes
        assert "unknown" in classes

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = IntentClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedIntentClassifier:
    """Test CachedIntentClassifier implementation."""

    def test_cached_intent_classifier_initialization(self):
        """Test CachedIntentClassifier initialization."""
        classifier = CachedIntentClassifier(cache_size=64)
        
        assert classifier.name == "cached_intent"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_intent_classifier_caching(self):
        """Test that CachedIntentClassifier properly caches results."""
        classifier = CachedIntentClassifier()
        
        # First call
        result1 = classifier.classify("What time is it?")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("What time is it?")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_intent_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedIntentClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "question" in classes
        assert "statement" in classes


class TestIntentClassifierFactory:
    """Test intent classifier factory function."""

    def test_create_intent_classifier_default(self):
        """Test creating intent classifier with default parameters."""
        classifier = create_intent_classifier()
        
        assert isinstance(classifier, IntentClassifier)
        assert classifier.model_name == "microsoft/DialoGPT-medium"
        assert classifier.threshold == 0.4

    def test_create_intent_classifier_cached(self):
        """Test creating cached intent classifier."""
        classifier = create_intent_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedIntentClassifier)
        assert classifier.cache_size == 64

    def test_create_intent_classifier_custom_params(self):
        """Test creating intent classifier with custom parameters."""
        classifier = create_intent_classifier(
            model_name="custom/intent-model",
            threshold=0.6
        )
        
        assert isinstance(classifier, IntentClassifier)
        assert classifier.model_name == "custom/intent-model"
        assert classifier.threshold == 0.6


class TestIntentClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = IntentClassifier()
        long_text = "What time is the meeting? " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = IntentClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = IntentClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.intent.importlib.import_module')
    async def test_pipeline_error_fallback(self, mock_import):
        """Test fallback when pipeline raises an error."""
        # Mock transformers with failing pipeline
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Pipeline failed")
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = IntentClassifier()
        result = await classifier.classify_async("Test message")
        
        # Should fallback to pattern-based analysis
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "pattern_based"
