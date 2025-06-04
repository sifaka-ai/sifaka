"""Emotion classification for detecting specific emotions in text.

This module provides a classifier for detecting emotions in text using pretrained models
from Hugging Face. Designed for the new PydanticAI-based Sifaka architecture.

Detects emotions like joy, sadness, anger, fear, surprise, disgust, and more.
"""

import importlib
import asyncio

from sifaka.classifiers.base import (
    BaseClassifier,
    CachedClassifier,
    ClassificationResult,
    TimingMixin,
)
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Popular pretrained emotion detection models
EMOTION_MODELS = {
    "j-hartmann/emotion-english-distilroberta-base": {
        "description": "DistilRoBERTa model for emotion classification",
        "emotions": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
    "cardiffnlp/twitter-roberta-base-emotion": {
        "description": "RoBERTa model trained on Twitter data for emotion detection",
        "emotions": [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "love",
            "optimism",
            "pessimism",
            "sadness",
            "surprise",
            "trust",
        ],
    },
    "SamLowe/roberta-base-go_emotions": {
        "description": "RoBERTa model for fine-grained emotion classification",
        "emotions": [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ],
    },
}


class EmotionClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting emotions in text using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    for accurate emotion detection.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for emotion detection
        pipeline: The Hugging Face pipeline (if available)
        emotions: List of emotions the model can detect
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        threshold: float = 0.3,
        name: str = "emotion_detection",
        description: str = "Detects emotions in text using pretrained models",
    ):
        """Initialize the emotion detection classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for emotion detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = EMOTION_MODELS.get(model_name, {})
        self.emotions = self.model_info.get(
            "emotions", ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        )
        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.emotions

    def _initialize_model(self) -> None:
        """Initialize the pretrained emotion detection model."""
        try:
            # Try to use transformers pipeline
            transformers = importlib.import_module("transformers")

            # Create a text classification pipeline
            self.pipeline = transformers.pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                device=-1,  # Use CPU by default
                truncation=True,
                max_length=512,
            )

            logger.debug(
                f"Initialized emotion classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                    "emotions": self.emotions,
                },
            )

        except (ImportError, Exception) as e:
            logger.warning(
                f"Transformers not available or model loading failed: {e}. "
                "Using simple fallback emotion detection. "
                "Install transformers for better accuracy: pip install transformers",
                extra={"classifier": self.name},
            )
            self.pipeline = None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for emotions asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with emotion prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("neutral")

        with self.time_operation("emotion_classification") as timer:
            try:
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                else:
                    result = await self._classify_with_fallback(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Emotion classification completed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "label": result.label,
                        "confidence": result.confidence,
                        "method": result.metadata.get("method", "unknown"),
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"Emotion classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for emotion: {str(e)}",
                    error_code="classification_error",
                    context={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check if transformers is properly installed",
                        "Verify input text is valid",
                        "Try with shorter text",
                    ],
                ) from e

    async def _classify_with_pipeline(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline."""

        def analyze():
            results = self.pipeline(text)
            return results

        try:
            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, analyze)

            # Process results - get the highest scoring emotion
            best_result = max(results, key=lambda x: x["score"])
            emotion = best_result["label"].lower()
            confidence = float(best_result["score"])

            # Get all emotions above threshold
            detected_emotions = [
                {"emotion": result["label"].lower(), "confidence": float(result["score"])}
                for result in results
                if float(result["score"]) >= self.threshold
            ]

            return self.create_classification_result(
                label=emotion,
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "threshold": self.threshold,
                    "input_length": len(text),
                    "all_emotions": results,
                    "detected_emotions": detected_emotions,
                    "primary_emotion": emotion,
                },
            )

        except Exception as e:
            # Fallback to simple analysis
            logger.warning(
                f"Pipeline emotion classification failed, using simple fallback: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_fallback(text)

    async def _classify_with_fallback(self, text: str) -> ClassificationResult:
        """Simple fallback emotion classification based on keywords."""

        def analyze():
            text_lower = text.lower()

            # Simple emotion keywords
            emotion_keywords = {
                "joy": [
                    "happy",
                    "joy",
                    "excited",
                    "wonderful",
                    "great",
                    "amazing",
                    "love",
                    "fantastic",
                ],
                "sadness": ["sad", "depressed", "unhappy", "miserable", "crying", "tears", "grief"],
                "anger": ["angry", "mad", "furious", "rage", "hate", "annoyed", "frustrated"],
                "fear": ["scared", "afraid", "terrified", "worried", "anxious", "nervous", "panic"],
                "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "wow"],
                "disgust": ["disgusted", "gross", "yuck", "revolting", "sick", "nasty"],
            }

            emotion_scores = {}
            for emotion, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    emotion_scores[emotion] = min(0.8, 0.4 + score * 0.1)

            if emotion_scores:
                primary_emotion = max(emotion_scores.keys(), key=lambda e: emotion_scores[e])
                confidence = emotion_scores[primary_emotion]
            else:
                primary_emotion = "neutral"
                confidence = 0.7

            return primary_emotion, confidence, emotion_scores

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        emotion, confidence, emotion_scores = await loop.run_in_executor(None, analyze)

        return self.create_classification_result(
            label=emotion,
            confidence=confidence,
            metadata={
                "method": "keyword_fallback",
                "emotion_scores": emotion_scores,
                "input_length": len(text),
                "warning": "Using simple fallback - install transformers for better accuracy",
            },
        )


class CachedEmotionClassifier(CachedClassifier):
    """Cached version of the emotion classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        threshold: float = 0.3,
        cache_size: int = 128,
        name: str = "cached_emotion",
        description: str = "Cached emotion classifier using pretrained models",
    ):
        """Initialize the cached emotion classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = EmotionClassifier(
            model_name=model_name,
            threshold=threshold,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform emotion classification without caching."""
        # Use a new event loop in a thread to avoid "asyncio.run() cannot be called from a running event loop"
        import asyncio
        import concurrent.futures

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._classifier.classify_async(text))
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self._classifier.get_classes()


# Factory function for easy creation
def create_emotion_classifier(
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    threshold: float = 0.3,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create an emotion classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for emotion detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured emotion classifier
    """
    if cached:
        return CachedEmotionClassifier(
            model_name=model_name,
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return EmotionClassifier(
            model_name=model_name,
            threshold=threshold,
        )
