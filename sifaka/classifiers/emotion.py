"""Emotion classification for detecting specific emotions in text.

This module provides a classifier for detecting emotions in text using pretrained models
from Hugging Face transformers.

Detects emotions like joy, sadness, anger, fear, surprise, disgust, and more.

Requires transformers library to be installed.
"""

import asyncio
import importlib

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
    for accurate emotion detection. Requires transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for emotion detection
        pipeline: The Hugging Face transformers pipeline
        emotions: List of emotions the model can detect
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        threshold: float = 0.3,
        adaptive_threshold: bool = True,
        name: str = "emotion_detection",
        description: str = "Detects emotions in text using pretrained models",
    ):
        """Initialize the emotion detection classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for emotion detection
            adaptive_threshold: Whether to adjust threshold based on number of emotions
            name: Name of the classifier
            description: Description of the classifier

        Raises:
            ImportError: If transformers library is not installed
            Exception: If model loading fails
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.base_threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.pipeline = None
        self.model_info = EMOTION_MODELS.get(model_name, {})
        self.emotions = self.model_info.get(
            "emotions", ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        )

        # Calculate adaptive threshold based on number of emotions
        if self.adaptive_threshold:
            # For models with many emotions, use a lower threshold
            # For models with few emotions, use the base threshold
            adaptive_value = 1.0 / len(self.emotions)
            self.threshold = min(self.base_threshold, adaptive_value)
        else:
            self.threshold = self.base_threshold

        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.emotions

    def create_empty_text_result(self, default_label: str = "neutral") -> ClassificationResult:
        """Create a result for empty or None text with meaningful confidence.

        For emotion classification, empty text is considered neutral with high confidence
        since the absence of emotional content is itself a strong indicator of neutrality.

        Args:
            default_label: Default label to use for empty text (should be "neutral")

        Returns:
            ClassificationResult indicating neutral emotion for empty text
        """
        return self.create_classification_result(
            label=default_label,
            confidence=0.95,  # High confidence that empty text is neutral
            metadata={
                "reason": "empty_text",
                "input_length": 0,
                "method": "empty_text_heuristic",
            },
        )

    def _initialize_model(self) -> None:
        """Initialize the pretrained emotion detection model."""
        try:
            # Import transformers - fail fast if not available
            transformers = importlib.import_module("transformers")

            # Create a text classification pipeline
            try:
                self.pipeline = transformers.pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=True,
                    device=-1,  # Use CPU by default
                    truncation=True,
                    max_length=512,
                )
            except (OSError, ValueError) as model_error:
                # Model not found or invalid - fallback to keyword analysis
                logger.warning(
                    f"Model '{self.model_name}' not found, using keyword-based fallback",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "error": str(model_error),
                        "fallback_method": "keyword_analysis",
                    },
                )
                self.pipeline = None
                return

            logger.debug(
                "Initialized emotion classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                    "emotions": self.emotions,
                },
            )
        except ImportError as e:
            # Transformers not available - use fallback method
            self.pipeline = None
            logger.warning(
                "Transformers not available, using keyword-based fallback",
                extra={
                    "classifier": self.name,
                    "error": str(e),
                    "fallback_method": "keyword_analysis",
                },
            )

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
                # Use pipeline if available, otherwise fallback to keyword analysis
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                else:
                    result = await self._classify_with_keywords(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    "Emotion classification completed",
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
                    "Emotion classification failed",
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

            # Process results - handle different pipeline output formats
            if not results:
                raise ValueError("Pipeline returned empty results")

            # Handle nested list format (real transformers often returns [[{...}]] instead of [{...}])
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                results = results[0]  # Unwrap the nested list

            # Check if results is a list of dictionaries or a different format
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                    # Standard format: list of dicts with score and label
                    best_result = max(results, key=lambda x: x["score"])
                    emotion = best_result["label"].lower()
                    confidence = float(best_result["score"])

                    # If the best result is below threshold, return neutral
                    if confidence < self.threshold:
                        emotion = "neutral"
                        # Set confidence to a reasonable value for neutral (1 - max_confidence)
                        # This represents confidence that the text is neutral rather than emotional
                        confidence = 1.0 - confidence

                else:
                    # Alternative format - try to extract from first result
                    if hasattr(results[0], "label") and hasattr(results[0], "score"):
                        emotion = results[0].label.lower()
                        confidence = float(results[0].score)

                        # If below threshold, return neutral
                        if confidence < self.threshold:
                            emotion = "neutral"
                            # Set confidence to a reasonable value for neutral (1 - max_confidence)
                            confidence = 1.0 - confidence

                    else:
                        raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
            else:
                raise ValueError(f"Unexpected pipeline results type: {type(results)}")

            # Get all emotions above threshold and calculate intensity scores
            detected_emotions = []
            max_score = max(float(result["score"]) for result in results) if results else 1.0

            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                    # Standard format: list of dicts with score and label
                    detected_emotions = [
                        {
                            "emotion": result["label"].lower(),
                            "confidence": float(result["score"]),
                            "intensity": float(result["score"]) / max_score,  # Relative intensity
                        }
                        for result in results
                        if float(result["score"]) >= self.threshold
                    ]
                else:
                    # Alternative format - only include the primary emotion if above threshold
                    if confidence >= self.threshold:
                        detected_emotions = [
                            {
                                "emotion": emotion,
                                "confidence": confidence,
                                "intensity": confidence / max_score,
                            }
                        ]

            return self.create_classification_result(
                label=emotion,
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "threshold": self.threshold,
                    "adaptive_threshold": self.adaptive_threshold,
                    "base_threshold": self.base_threshold,
                    "input_length": len(text),
                    "all_emotions": results,
                    "detected_emotions": detected_emotions,
                    "primary_emotion": emotion,
                    "emotion_intensity": confidence / max_score,
                    "max_emotion_score": max_score,
                },
            )

        except Exception as e:
            logger.error(
                f"Pipeline emotion classification failed: {e}",
                extra={"classifier": self.name},
                exc_info=True,
            )
            # Fallback to keyword analysis when pipeline fails
            logger.warning(
                "Pipeline failed, falling back to keyword analysis",
                extra={"classifier": self.name, "error": str(e)},
            )
            return await self._classify_with_keywords(text)

    async def _classify_with_keywords(self, text: str) -> ClassificationResult:
        """Classify using keyword-based fallback method when transformers is not available."""

        # Define emotion keywords for fallback classification
        emotion_keywords = {
            "joy": [
                "happy",
                "joyful",
                "excited",
                "delighted",
                "cheerful",
                "glad",
                "pleased",
                "thrilled",
                "elated",
                "ecstatic",
            ],
            "sadness": [
                "sad",
                "depressed",
                "unhappy",
                "miserable",
                "heartbroken",
                "devastated",
                "grief",
                "sorrow",
                "melancholy",
                "down",
            ],
            "anger": [
                "angry",
                "furious",
                "mad",
                "rage",
                "outraged",
                "irritated",
                "annoyed",
                "frustrated",
                "livid",
                "enraged",
            ],
            "fear": [
                "afraid",
                "scared",
                "terrified",
                "anxious",
                "worried",
                "nervous",
                "frightened",
                "panic",
                "dread",
                "alarmed",
            ],
            "surprise": [
                "surprised",
                "amazed",
                "astonished",
                "shocked",
                "stunned",
                "bewildered",
                "startled",
                "astounded",
            ],
            "disgust": [
                "disgusted",
                "revolted",
                "repulsed",
                "sickened",
                "nauseated",
                "appalled",
                "horrified",
            ],
        }

        text_lower = text.lower()
        emotion_scores = {}
        total_matches = 0

        # Count keyword matches for each emotion
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                emotion_scores[emotion] = matches
                total_matches += matches

        # Determine the dominant emotion
        if emotion_scores:
            best_emotion = max(emotion_scores.keys(), key=lambda e: emotion_scores[e])
            # Improved confidence calculation: base confidence + bonus for multiple matches
            base_confidence = 0.6  # Base confidence for any keyword match
            match_bonus = min(
                emotion_scores[best_emotion] * 0.1, 0.3
            )  # Up to 0.3 bonus for multiple matches
            confidence = base_confidence + match_bonus
            confidence = min(confidence, 1.0)  # Cap at 1.0

            # Apply threshold logic
            if confidence < self.threshold:
                best_emotion = "neutral"
                confidence = 1.0 - confidence  # Confidence in neutrality
        else:
            # No emotional keywords found
            best_emotion = "neutral"
            confidence = 0.8  # High confidence in neutrality when no emotional words
            total_matches = 0

        return self.create_classification_result(
            label=best_emotion,
            confidence=confidence,
            metadata={
                "method": "keyword_analysis",
                "keyword_matches": total_matches,
                "emotion_scores": emotion_scores,
                "threshold": self.threshold,
                "input_length": len(text),
                "fallback_reason": "transformers_not_available",
            },
        )

    async def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts for emotions asynchronously.

        This method processes multiple texts together for better efficiency
        compared to calling classify_async multiple times.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResult objects, one for each input text
        """
        if not texts:
            return []

        with self.time_operation("emotion_batch_classification") as timer:
            try:
                # Process all texts at once using the pipeline
                def analyze_batch():
                    return self.pipeline(texts)

                # Use asyncio to run in thread pool for CPU-bound work
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(None, analyze_batch)

                # Process each result
                results = []
                for i, (text, pipeline_result) in enumerate(zip(texts, batch_results)):
                    if not text or not text.strip():
                        result = self.create_empty_text_result("neutral")
                    else:
                        result = self._process_pipeline_result(text, pipeline_result)

                    results.append(result)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)

                # Update processing time for each result (distributed across batch)
                avg_time = processing_time / len(texts) if texts else 0.0
                for result in results:
                    result.processing_time_ms = avg_time

                logger.debug(
                    "Batch emotion classification completed",
                    extra={
                        "classifier": self.name,
                        "batch_size": len(texts),
                        "total_time_ms": processing_time,
                        "avg_time_per_text_ms": avg_time,
                    },
                )

                return results

            except Exception as e:
                logger.error(
                    "Batch emotion classification failed",
                    extra={
                        "classifier": self.name,
                        "batch_size": len(texts),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify batch for emotion: {str(e)}",
                    error_code="batch_classification_error",
                    context={
                        "classifier": self.name,
                        "batch_size": len(texts),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check if transformers is properly installed",
                        "Verify input texts are valid",
                        "Try with smaller batch size",
                        "Check system memory availability",
                    ],
                ) from e

    def _process_pipeline_result(self, text: str, results) -> ClassificationResult:
        """Process pipeline results for a single text (helper for batch processing)."""
        # This is the same logic as in _classify_with_pipeline but extracted for reuse
        if not results:
            raise ValueError("Pipeline returned empty results")

        # Check if results is a list of dictionaries or a different format
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                best_result = max(results, key=lambda x: x["score"])
                emotion = best_result["label"].lower()
                confidence = float(best_result["score"])

                # If the best result is below threshold, return neutral
                if confidence < self.threshold:
                    emotion = "neutral"
                    # Set confidence to a reasonable value for neutral (1 - max_confidence)
                    confidence = 1.0 - confidence

            else:
                # Alternative format - try to extract from first result
                if hasattr(results[0], "label") and hasattr(results[0], "score"):
                    emotion = results[0].label.lower()
                    confidence = float(results[0].score)

                    # If below threshold, return neutral
                    if confidence < self.threshold:
                        emotion = "neutral"
                        # Set confidence to a reasonable value for neutral (1 - max_confidence)
                        confidence = 1.0 - confidence

                else:
                    raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
        else:
            raise ValueError(f"Unexpected pipeline results type: {type(results)}")

        # Get all emotions above threshold and calculate intensity scores
        detected_emotions = []
        max_score = max(float(result["score"]) for result in results) if results else 1.0

        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                detected_emotions = [
                    {
                        "emotion": result["label"].lower(),
                        "confidence": float(result["score"]),
                        "intensity": float(result["score"]) / max_score,  # Relative intensity
                    }
                    for result in results
                    if float(result["score"]) >= self.threshold
                ]
            else:
                # Alternative format - only include the primary emotion if above threshold
                if confidence >= self.threshold:
                    detected_emotions = [
                        {
                            "emotion": emotion,
                            "confidence": confidence,
                            "intensity": confidence / max_score,
                        }
                    ]

        return self.create_classification_result(
            label=emotion,
            confidence=confidence,
            metadata={
                "method": "transformers_pipeline_batch",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "base_threshold": self.base_threshold,
                "input_length": len(text),
                "all_emotions": results,
                "detected_emotions": detected_emotions,
                "primary_emotion": emotion,
                "emotion_intensity": confidence / max_score,
                "max_emotion_score": max_score,
            },
        )


class CachedEmotionClassifier(CachedClassifier, TimingMixin):
    """Cached version of EmotionClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        threshold: float = 0.3,
        adaptive_threshold: bool = True,
        cache_size: int = 128,
        name: str = "cached_emotion",
        description: str = "Cached emotion classifier using pretrained models",
    ):
        """Initialize the cached emotion classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for emotion detection
            adaptive_threshold: Whether to adjust threshold based on number of emotions
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.base_threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.pipeline = None
        self.model_info = EMOTION_MODELS.get(model_name, {})
        self.emotions = self.model_info.get(
            "emotions", ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        )

        # Calculate adaptive threshold based on number of emotions
        if self.adaptive_threshold:
            # For models with many emotions, use a lower threshold
            # For models with few emotions, use the base threshold
            adaptive_value = 1.0 / len(self.emotions)
            self.threshold = min(self.base_threshold, adaptive_value)
        else:
            self.threshold = self.base_threshold

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained emotion detection model."""
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
            "Initialized cached emotion classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
                "emotions": self.emotions,
            },
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform emotion classification without caching."""
        if not text or not text.strip():
            return self.create_empty_text_result("neutral")

        try:
            return self._classify_with_pipeline_sync(text)
        except Exception as e:
            logger.error(
                "Cached emotion classification failed",
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

    def _classify_with_pipeline_sync(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline (synchronous)."""
        if self.pipeline is None:
            raise ValidationError(
                "Transformers pipeline is not available",
                error_code="dependency_missing",
                suggestions=["Install transformers: pip install transformers"],
            )

        # Run transformers analysis
        results = self.pipeline(text)

        # Process results - handle different pipeline output formats
        if not results:
            raise ValueError("Pipeline returned empty results")

        # Handle nested list format (some mocks return [[{...}]] instead of [{...}])
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]  # Unwrap the nested list

        # Check if results is a list of dictionaries or a different format
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                best_result = max(results, key=lambda x: x["score"])
                emotion = best_result["label"].lower()
                confidence = float(best_result["score"])

                # If the best result is below threshold, return neutral
                if confidence < self.threshold:
                    emotion = "neutral"
                    # Set confidence to a reasonable value for neutral (1 - max_confidence)
                    confidence = 1.0 - confidence

            else:
                # Alternative format - try to extract from first result
                if hasattr(results[0], "label") and hasattr(results[0], "score"):
                    emotion = results[0].label.lower()
                    confidence = float(results[0].score)

                    # If below threshold, return neutral
                    if confidence < self.threshold:
                        emotion = "neutral"
                        # Set confidence to a reasonable value for neutral (1 - max_confidence)
                        confidence = 1.0 - confidence

                else:
                    raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
        else:
            raise ValueError(f"Unexpected pipeline results type: {type(results)}")

        # Get all emotions above threshold and calculate intensity scores
        detected_emotions = []
        max_score = max(float(result["score"]) for result in results) if results else 1.0

        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                # Standard format: list of dicts with score and label
                detected_emotions = [
                    {
                        "emotion": result["label"].lower(),
                        "confidence": float(result["score"]),
                        "intensity": float(result["score"]) / max_score,  # Relative intensity
                    }
                    for result in results
                    if float(result["score"]) >= self.threshold
                ]
            else:
                # Alternative format - only include the primary emotion if above threshold
                if confidence >= self.threshold:
                    detected_emotions = [
                        {
                            "emotion": emotion,
                            "confidence": confidence,
                            "intensity": confidence / max_score,
                        }
                    ]

        return self.create_classification_result(
            label=emotion,
            confidence=confidence,
            metadata={
                "method": "transformers_pipeline",
                "model_name": self.model_name,
                "threshold": self.threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "base_threshold": self.base_threshold,
                "input_length": len(text),
                "all_emotions": results,
                "detected_emotions": detected_emotions,
                "primary_emotion": emotion,
                "emotion_intensity": confidence / max_score,
                "max_emotion_score": max_score,
                "cached": True,
            },
        )

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.emotions


# Factory function for easy creation
def create_emotion_classifier(
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    threshold: float = 0.3,
    adaptive_threshold: bool = True,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create an emotion classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for emotion detection
        adaptive_threshold: Whether to adjust threshold based on number of emotions
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured emotion classifier
    """
    if cached:
        return CachedEmotionClassifier(
            model_name=model_name,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold,
            cache_size=cache_size,
        )
    else:
        return EmotionClassifier(
            model_name=model_name,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold,
        )
