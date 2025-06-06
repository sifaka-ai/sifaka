"""Comprehensive unit tests for ReadabilityClassifier.

This module tests the readability assessment classifier including:
- Readability assessment using transformers pipeline
- Traditional metrics using textstat
- Grade level determination
- Caching functionality
- Error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from sifaka.classifiers.readability import (
    ReadabilityClassifier,
    CachedReadabilityClassifier,
    create_readability_classifier,
)
from sifaka.classifiers.base import ClassificationResult
from sifaka.utils.errors import ValidationError


class TestReadabilityClassifier:
    """Test ReadabilityClassifier implementation."""

    def test_readability_classifier_initialization(self):
        """Test ReadabilityClassifier initialization with default parameters."""
        classifier = ReadabilityClassifier()
        
        assert classifier.name == "readability"
        assert "readability" in classifier.description.lower()
        assert classifier.model_name == "microsoft/DialoGPT-medium"
        assert "elementary" in classifier.grade_levels
        assert "college" in classifier.grade_levels

    def test_readability_classifier_custom_parameters(self):
        """Test ReadabilityClassifier initialization with custom parameters."""
        custom_grades = ["basic", "intermediate", "advanced"]
        classifier = ReadabilityClassifier(
            model_name="custom/readability-model",
            grade_levels=custom_grades,
            name="custom_readability",
            description="Custom readability analyzer"
        )
        
        assert classifier.name == "custom_readability"
        assert classifier.description == "Custom readability analyzer"
        assert classifier.model_name == "custom/readability-model"
        assert classifier.grade_levels == custom_grades

    @patch('sifaka.classifiers.readability.importlib.import_module')
    def test_readability_classifier_transformers_initialization(self, mock_import):
        """Test ReadabilityClassifier initialization with transformers available."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers
        
        classifier = ReadabilityClassifier()
        
        # Verify transformers was imported and pipeline created
        mock_import.assert_called_with("transformers")
        assert classifier.pipeline is not None

    @patch('sifaka.classifiers.readability.importlib.import_module')
    def test_readability_classifier_no_transformers(self, mock_import):
        """Test ReadabilityClassifier when transformers is not available."""
        mock_import.side_effect = ImportError("transformers not available")
        
        classifier = ReadabilityClassifier()
        
        # Should still initialize but with no pipeline
        assert classifier.pipeline is None

    @pytest.mark.asyncio
    async def test_classify_async_empty_text(self):
        """Test classification with empty text."""
        classifier = ReadabilityClassifier()
        
        result = await classifier.classify_async("")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = ReadabilityClassifier()
        
        result = await classifier.classify_async("   \n\t   ")
        
        assert result.label == "unknown"
        assert result.confidence > 0
        assert "empty_text" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.readability.importlib.import_module')
    async def test_classify_with_pipeline(self, mock_import):
        """Test readability classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "EASY", "score": 0.8},
            {"label": "MEDIUM", "score": 0.15},
            {"label": "HARD", "score": 0.05}
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers
        
        classifier = ReadabilityClassifier()
        result = await classifier.classify_async("The cat sat on the mat.")
        
        assert result.label == "easy"
        assert result.confidence == 0.8
        assert result.metadata["method"] == "transformers_pipeline"
        assert "grade_level" in result.metadata

    @pytest.mark.asyncio
    @patch('sifaka.classifiers.readability.importlib.import_module')
    async def test_classify_with_textstat_metrics(self, mock_import):
        """Test readability classification using textstat metrics."""
        # Mock transformers failure, textstat success
        def side_effect(module_name):
            if module_name == "transformers":
                raise ImportError("transformers not available")
            elif module_name == "textstat":
                mock_textstat = Mock()
                mock_textstat.flesch_reading_ease.return_value = 65.0
                mock_textstat.flesch_kincaid_grade.return_value = 8.5
                mock_textstat.automated_readability_index.return_value = 9.2
                mock_textstat.coleman_liau_index.return_value = 10.1
                mock_textstat.gunning_fog.return_value = 11.3
                mock_textstat.smog_index.return_value = 9.8
                mock_textstat.text_standard.return_value = "9th and 10th grade"
                return mock_textstat
            return Mock()
        
        mock_import.side_effect = side_effect
        
        classifier = ReadabilityClassifier()
        result = await classifier.classify_async("This is a moderately complex sentence.")
        
        assert result.label in ["elementary", "middle", "high", "college", "graduate"]
        assert result.confidence > 0
        assert result.metadata["method"] == "textstat_metrics"
        assert "flesch_reading_ease" in result.metadata
        assert "grade_level" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_with_simple_fallback(self):
        """Test readability classification using simple fallback."""
        # Create classifier with no external dependencies
        with patch('sifaka.classifiers.readability.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = ReadabilityClassifier()
        
        # Test simple text
        simple_text = "The cat sat on the mat."
        result = await classifier.classify_async(simple_text)
        
        assert result.label in classifier.grade_levels
        assert result.confidence > 0.5
        assert result.metadata["method"] == "simple_heuristics"
        assert "avg_word_length" in result.metadata
        assert "avg_sentence_length" in result.metadata

    @pytest.mark.asyncio
    async def test_classify_complex_text_fallback(self):
        """Test readability classification of complex text using fallback."""
        with patch('sifaka.classifiers.readability.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No dependencies")
            classifier = ReadabilityClassifier()
        
        # Test complex text
        complex_text = "The multifaceted implications of quantum mechanical phenomena necessitate comprehensive analysis."
        result = await classifier.classify_async(complex_text)
        
        assert result.label in classifier.grade_levels
        assert result.confidence > 0.5
        assert result.metadata["method"] == "simple_heuristics"
        # Complex text should have higher grade level
        assert result.label in ["high", "college", "graduate"]

    def test_determine_grade_level(self):
        """Test _determine_grade_level method."""
        classifier = ReadabilityClassifier()
        
        # Test different complexity levels
        assert classifier._determine_grade_level("easy", 0.9) == "elementary"
        assert classifier._determine_grade_level("medium", 0.7) == "middle"
        assert classifier._determine_grade_level("hard", 0.8) == "high"
        assert classifier._determine_grade_level("complex", 0.6) == "college"

    def test_get_classes(self):
        """Test get_classes method returns grade levels."""
        classifier = ReadabilityClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "elementary" in classes
        assert "middle" in classes
        assert "high" in classes
        assert "college" in classes
        assert "graduate" in classes

    @pytest.mark.asyncio
    async def test_timing_functionality(self):
        """Test that timing is properly recorded."""
        classifier = ReadabilityClassifier()
        result = await classifier.classify_async("Test message")
        
        assert result.processing_time_ms >= 0
        assert isinstance(result.processing_time_ms, float)


class TestCachedReadabilityClassifier:
    """Test CachedReadabilityClassifier implementation."""

    def test_cached_readability_classifier_initialization(self):
        """Test CachedReadabilityClassifier initialization."""
        classifier = CachedReadabilityClassifier(cache_size=64)
        
        assert classifier.name == "cached_readability"
        assert classifier.cache_size == 64
        assert hasattr(classifier, '_classifier')

    def test_cached_readability_classifier_caching(self):
        """Test that CachedReadabilityClassifier properly caches results."""
        classifier = CachedReadabilityClassifier()
        
        # First call
        result1 = classifier.classify("The cat sat on the mat.")
        
        # Second call with same text (should use cache)
        result2 = classifier.classify("The cat sat on the mat.")
        
        # Results should be identical
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_cached_readability_classifier_get_classes(self):
        """Test get_classes method for cached classifier."""
        classifier = CachedReadabilityClassifier()
        classes = classifier.get_classes()
        
        assert isinstance(classes, list)
        assert "elementary" in classes
        assert "college" in classes


class TestReadabilityClassifierFactory:
    """Test readability classifier factory function."""

    def test_create_readability_classifier_default(self):
        """Test creating readability classifier with default parameters."""
        classifier = create_readability_classifier()
        
        assert isinstance(classifier, ReadabilityClassifier)
        assert classifier.model_name == "microsoft/DialoGPT-medium"

    def test_create_readability_classifier_cached(self):
        """Test creating cached readability classifier."""
        classifier = create_readability_classifier(cached=True, cache_size=64)
        
        assert isinstance(classifier, CachedReadabilityClassifier)
        assert classifier.cache_size == 64

    def test_create_readability_classifier_custom_params(self):
        """Test creating readability classifier with custom parameters."""
        custom_grades = ["basic", "advanced"]
        classifier = create_readability_classifier(
            model_name="custom/model",
            grade_levels=custom_grades
        )
        
        assert isinstance(classifier, ReadabilityClassifier)
        assert classifier.model_name == "custom/model"
        assert classifier.grade_levels == custom_grades


class TestReadabilityClassifierEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test classification with very long text."""
        classifier = ReadabilityClassifier()
        long_text = "This is a test sentence. " * 1000
        
        result = await classifier.classify_async(long_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test classification with special characters."""
        classifier = ReadabilityClassifier()
        special_text = "Hello! @#$%^&*()_+ 123 ñáéíóú"
        
        result = await classifier.classify_async(special_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """Test classification with Unicode text."""
        classifier = ReadabilityClassifier()
        unicode_text = "Hello 世界 🌍 emoji test"
        
        result = await classifier.classify_async(unicode_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_single_word(self):
        """Test classification with single word."""
        classifier = ReadabilityClassifier()
        single_word = "Hello"
        
        result = await classifier.classify_async(single_word)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_numbers_only(self):
        """Test classification with numbers only."""
        classifier = ReadabilityClassifier()
        numbers_text = "123 456 789"
        
        result = await classifier.classify_async(numbers_text)
        
        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0
