"""Comprehensive unit tests for EmotionClassifier.

This module tests the emotion detection classifier including:
- Emotion detection using transformers pipeline
- Simple keyword-based fallback
- Caching functionality
- Error handling and edge cases
"""

from unittest.mock import Mock, patch

import pytest

from sifaka.classifiers.emotion import (
    EMOTION_MODELS,
    CachedEmotionClassifier,
    EmotionClassifier,
    create_emotion_classifier,
)


class TestEmotionClassifier:
    """Test EmotionClassifier implementation."""

    def test_emotion_classifier_initialization(self):
        """Test EmotionClassifier initialization with default parameters."""
        classifier = EmotionClassifier()

        assert classifier.name == "emotion_detection"
        assert "emotion" in classifier.description.lower()
        assert classifier.model_name == "j-hartmann/emotion-english-distilroberta-base"
        # With adaptive_threshold=True (default), threshold is min(0.3, 1.0/7) = 1/7 ≈ 0.143
        assert classifier.base_threshold == 0.3
        assert classifier.adaptive_threshold == True
        assert abs(classifier.threshold - (1.0 / 7)) < 0.001  # Adaptive threshold for 7 emotions

    @patch("sifaka.classifiers.emotion.importlib.import_module")
    def test_emotion_classifier_custom_parameters(self, mock_import):
        """Test EmotionClassifier initialization with custom parameters."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier(
            model_name="custom/emotion-model",
            threshold=0.5,
            adaptive_threshold=False,  # Disable adaptive threshold for this test
            name="custom_emotion",
            description="Custom emotion detector",
        )

        assert classifier.name == "custom_emotion"
        assert classifier.description == "Custom emotion detector"
        assert classifier.model_name == "custom/emotion-model"
        assert classifier.threshold == 0.5

    @patch("sifaka.classifiers.emotion.importlib.import_module")
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
            return_all_scores=True,
            device=-1,
            truncation=True,
            max_length=512,
        )
        assert classifier.pipeline is not None

    @patch("sifaka.classifiers.emotion.importlib.import_module")
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
        assert result.metadata["reason"] == "empty_text"

    @pytest.mark.asyncio
    async def test_classify_async_whitespace_text(self):
        """Test classification with whitespace-only text."""
        classifier = EmotionClassifier()

        result = await classifier.classify_async("   \n\t   ")

        assert result.label == "neutral"
        assert result.confidence > 0
        assert result.metadata["reason"] == "empty_text"

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_classify_with_pipeline(self, mock_import):
        """Test classification using transformers pipeline."""
        # Mock transformers
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "joy", "score": 0.85},
            {"label": "sadness", "score": 0.10},
            {"label": "anger", "score": 0.05},
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
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_classify_below_threshold(self, mock_import):
        """Test classification when all emotions are below threshold."""
        # Mock transformers with low confidence scores
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "joy", "score": 0.2},
            {"label": "sadness", "score": 0.15},
            {"label": "anger", "score": 0.1},
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier(threshold=0.3, adaptive_threshold=False)
        result = await classifier.classify_async("This is neutral text")

        assert result.label == "neutral"
        assert result.confidence > 0
        assert result.metadata["method"] == "transformers_pipeline"

    @pytest.mark.asyncio
    async def test_classify_with_fallback(self):
        """Test classification using simple fallback method."""
        # Create classifier with no transformers
        with patch("sifaka.classifiers.emotion.importlib.import_module") as mock_import:
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
        with patch("sifaka.classifiers.emotion.importlib.import_module") as mock_import:
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
        with patch("sifaka.classifiers.emotion.importlib.import_module") as mock_import:
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
        with patch("sifaka.classifiers.emotion.importlib.import_module") as mock_import:
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
        assert hasattr(classifier, "pipeline")

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
        assert classifier.base_threshold == 0.3
        # With adaptive_threshold=True (default), threshold is min(0.3, 1.0/7) = 1/7 ≈ 0.143
        assert abs(classifier.threshold - (1.0 / 7)) < 0.001

    def test_create_emotion_classifier_cached(self):
        """Test creating cached emotion classifier."""
        classifier = create_emotion_classifier(cached=True, cache_size=64)

        assert isinstance(classifier, CachedEmotionClassifier)
        assert classifier.cache_size == 64

    @patch("sifaka.classifiers.emotion.importlib.import_module")
    def test_create_emotion_classifier_custom_params(self, mock_import):
        """Test creating emotion classifier with custom parameters."""
        # Mock transformers module
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers

        classifier = create_emotion_classifier(
            model_name="custom/emotion-model",
            threshold=0.5,
            adaptive_threshold=False,  # Disable adaptive threshold for this test
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
    @patch("sifaka.classifiers.emotion.importlib.import_module")
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


class TestEmotionModelsConfiguration:
    """Test emotion models configuration and metadata."""

    def test_emotion_models_structure(self):
        """Test that EMOTION_MODELS has correct structure."""
        assert isinstance(EMOTION_MODELS, dict)
        assert len(EMOTION_MODELS) > 0

        for model_name, model_info in EMOTION_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(model_info, dict)
            assert "description" in model_info
            assert "emotions" in model_info
            assert isinstance(model_info["emotions"], list)
            assert len(model_info["emotions"]) > 0

    def test_default_model_exists(self):
        """Test that default model exists in EMOTION_MODELS."""
        default_model = "j-hartmann/emotion-english-distilroberta-base"
        assert default_model in EMOTION_MODELS

        model_info = EMOTION_MODELS[default_model]
        assert "anger" in model_info["emotions"]
        assert "joy" in model_info["emotions"]
        assert "sadness" in model_info["emotions"]

    def test_classifier_with_different_models(self):
        """Test classifier initialization with different predefined models."""
        for model_name in EMOTION_MODELS.keys():
            classifier = EmotionClassifier(model_name=model_name)
            assert classifier.model_name == model_name
            assert classifier.emotions == EMOTION_MODELS[model_name]["emotions"]

    def test_classifier_with_unknown_model(self):
        """Test classifier with model not in EMOTION_MODELS."""
        unknown_model = "unknown/emotion-model"
        classifier = EmotionClassifier(model_name=unknown_model)

        assert classifier.model_name == unknown_model
        # Should use default emotions list
        default_emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        assert classifier.emotions == default_emotions


class TestEmotionClassifierPipelineHandling:
    """Test pipeline handling and error scenarios."""

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_pipeline_different_output_formats(self, mock_import):
        """Test handling different pipeline output formats."""
        # Test format 1: List of dicts with score and label
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "joy", "score": 0.8},
            {"label": "sadness", "score": 0.2},
        ]
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier()
        result = await classifier.classify_async("Happy text")

        assert result.label == "joy"
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_pipeline_empty_results(self, mock_import):
        """Test handling empty pipeline results with fallback."""
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = []
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier()

        # Should fallback to keyword analysis instead of raising error
        result = await classifier.classify_async("Test text")
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "keyword_analysis"

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_pipeline_invalid_format(self, mock_import):
        """Test handling invalid pipeline result format with fallback."""
        mock_transformers = Mock()
        mock_pipeline = Mock()
        mock_pipeline.return_value = "invalid_format"
        mock_transformers.pipeline.return_value = mock_pipeline
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier()

        # Should fallback to keyword analysis instead of raising error
        result = await classifier.classify_async("Test text")
        assert result.label in classifier.get_classes()
        assert result.metadata["method"] == "keyword_analysis"

    @pytest.mark.asyncio
    @patch("sifaka.classifiers.emotion.importlib.import_module")
    async def test_pipeline_configuration(self, mock_import):
        """Test pipeline configuration parameters."""
        mock_transformers = Mock()
        mock_transformers.pipeline = Mock()
        mock_import.return_value = mock_transformers

        classifier = EmotionClassifier(model_name="custom/model")

        # Verify pipeline was called with correct parameters
        mock_transformers.pipeline.assert_called_with(
            "text-classification",
            model="custom/model",
            return_all_scores=True,
            device=-1,
            truncation=True,
            max_length=512,
        )

    @pytest.mark.asyncio
    async def test_create_empty_text_result(self):
        """Test create_empty_text_result method."""
        classifier = EmotionClassifier()

        result = classifier.create_empty_text_result("neutral")

        assert result.label == "neutral"
        assert (
            result.confidence == 0.95
        )  # EmotionClassifier returns high confidence for neutral empty text
        assert result.metadata["reason"] == "empty_text"
        assert result.metadata["input_length"] == 0

    @pytest.mark.asyncio
    async def test_create_classification_result(self):
        """Test create_classification_result method."""
        classifier = EmotionClassifier()

        result = classifier.create_classification_result(
            label="joy", confidence=0.85, metadata={"custom": "data"}
        )

        assert result.label == "joy"
        assert result.confidence == 0.85
        assert result.metadata["custom"] == "data"
        assert result.metadata["classifier_name"] == "emotion_detection"


class TestCachedEmotionClassifierAdvanced:
    """Advanced tests for CachedEmotionClassifier."""

    def test_cached_classifier_thread_safety(self):
        """Test cached classifier with concurrent access."""
        import concurrent.futures

        classifier = CachedEmotionClassifier()
        results = []

        def classify_text(text):
            return classifier.classify(f"Happy text {text}")

        # Run multiple classifications concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(classify_text, i) for i in range(10)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        assert len(results) == 10
        assert all(result.label in classifier.get_classes() for result in results)

    def test_cached_classifier_cache_info(self):
        """Test cache information retrieval."""
        classifier = CachedEmotionClassifier(cache_size=64)

        # Make some calls
        classifier.classify("Happy text")
        classifier.classify("Happy text")  # Cache hit
        classifier.classify("Sad text")  # Cache miss

        info = classifier.get_cache_info()
        assert info["hits"] >= 1
        assert info["misses"] >= 2
        assert info["maxsize"] == 64

    def test_cached_classifier_clear_cache(self):
        """Test cache clearing functionality."""
        classifier = CachedEmotionClassifier()

        # Make initial call
        result1 = classifier.classify("Test text")

        # Clear cache
        classifier.clear_cache()

        # Make same call again (should not use cache)
        result2 = classifier.classify("Test text")

        # Results should be the same but cache was cleared
        assert result1.label == result2.label

    @pytest.mark.asyncio
    async def test_cached_classifier_async_method(self):
        """Test cached classifier async method."""
        classifier = CachedEmotionClassifier()

        result = await classifier.classify_async("Happy async text")

        assert result.label in classifier.get_classes()
        assert 0.0 <= result.confidence <= 1.0
