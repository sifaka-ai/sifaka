"""Intent classification for detecting the purpose/intent of text.

This module provides a classifier for detecting intent in text using pretrained models
from Hugging Face. Designed for the new PydanticAI-based Sifaka architecture.

Detects intents like question, request, statement, command, complaint, etc.
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

# Popular pretrained intent classification models
INTENT_MODELS = {
    "microsoft/DialoGPT-medium": {
        "description": "General conversational intent classification",
        "intents": ["question", "statement", "request", "greeting", "goodbye"],
    },
    "facebook/blenderbot-400M-distill": {
        "description": "Conversational AI model for intent detection",
        "intents": ["question", "statement", "request", "complaint", "compliment", "greeting"],
    },
    "microsoft/GODEL-v1_1-base-seq2seq": {
        "description": "Goal-oriented dialog model for intent classification",
        "intents": ["question", "request", "inform", "confirm", "deny", "greeting", "goodbye"],
    },
}


class IntentClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting intent in text using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    for accurate intent detection.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for intent detection
        pipeline: The Hugging Face pipeline (if available)
        intents: List of intents the model can detect
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        threshold: float = 0.4,
        name: str = "intent_detection",
        description: str = "Detects intent in text using pretrained models",
    ):
        """Initialize the intent detection classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for intent detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = INTENT_MODELS.get(model_name, {})
        self.intents = self.model_info.get(
            "intents", ["question", "statement", "request", "greeting"]
        )
        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.intents

    def _initialize_model(self) -> None:
        """Initialize the pretrained intent detection model."""
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
                f"Initialized intent classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                    "intents": self.intents,
                },
            )

        except (ImportError, Exception) as e:
            logger.warning(
                f"Transformers not available or model loading failed: {e}. "
                "Using simple fallback intent detection. "
                "Install transformers for better accuracy: pip install transformers",
                extra={"classifier": self.name},
            )
            self.pipeline = None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for intent asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with intent prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("statement")

        with self.time_operation("intent_classification") as timer:
            try:
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                else:
                    result = await self._classify_with_fallback(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Intent classification completed",
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
                    f"Intent classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for intent: {str(e)}",
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

            # Process results - get the highest scoring intent
            best_result = max(results, key=lambda x: x["score"])
            intent = best_result["label"].lower()
            confidence = float(best_result["score"])

            # Get all intents above threshold
            detected_intents = [
                {"intent": result["label"].lower(), "confidence": float(result["score"])}
                for result in results
                if float(result["score"]) >= self.threshold
            ]

            return self.create_classification_result(
                label=intent,
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "threshold": self.threshold,
                    "input_length": len(text),
                    "all_intents": results,
                    "detected_intents": detected_intents,
                    "primary_intent": intent,
                },
            )

        except Exception as e:
            # Fallback to simple analysis
            logger.warning(
                f"Pipeline intent classification failed, using simple fallback: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_fallback(text)

    async def _classify_with_fallback(self, text: str) -> ClassificationResult:
        """Simple fallback intent classification based on patterns."""

        def analyze():
            text_lower = text.lower().strip()

            # Simple intent patterns
            intent_patterns = {
                "question": [
                    lambda t: t.startswith(
                        (
                            "what",
                            "how",
                            "why",
                            "when",
                            "where",
                            "who",
                            "which",
                            "can",
                            "could",
                            "would",
                            "should",
                            "is",
                            "are",
                            "do",
                            "does",
                            "did",
                        )
                    ),
                    lambda t: "?" in t,
                ],
                "request": [
                    lambda t: t.startswith(
                        (
                            "please",
                            "can you",
                            "could you",
                            "would you",
                            "i need",
                            "i want",
                            "help me",
                        )
                    ),
                    lambda t: any(word in t for word in ["please", "help", "assist", "support"]),
                ],
                "greeting": [
                    lambda t: t.startswith(
                        ("hello", "hi", "hey", "good morning", "good afternoon", "good evening")
                    ),
                    lambda t: any(word in t for word in ["hello", "hi", "hey", "greetings"]),
                ],
                "goodbye": [
                    lambda t: t.startswith(("goodbye", "bye", "see you", "farewell", "take care")),
                    lambda t: any(word in t for word in ["goodbye", "bye", "farewell", "later"]),
                ],
                "complaint": [
                    lambda t: any(
                        word in t
                        for word in [
                            "problem",
                            "issue",
                            "wrong",
                            "error",
                            "broken",
                            "not working",
                            "disappointed",
                            "frustrated",
                        ]
                    ),
                ],
                "compliment": [
                    lambda t: any(
                        word in t
                        for word in [
                            "great",
                            "excellent",
                            "amazing",
                            "wonderful",
                            "fantastic",
                            "love",
                            "perfect",
                            "awesome",
                        ]
                    ),
                ],
            }

            intent_scores = {}
            for intent, patterns in intent_patterns.items():
                score = sum(1 for pattern in patterns if pattern(text_lower))
                if score > 0:
                    intent_scores[intent] = min(0.8, 0.5 + score * 0.15)

            if intent_scores:
                primary_intent = max(intent_scores.keys(), key=lambda i: intent_scores[i])
                confidence = intent_scores[primary_intent]
            else:
                primary_intent = "statement"
                confidence = 0.7

            return primary_intent, confidence, intent_scores

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        intent, confidence, intent_scores = await loop.run_in_executor(None, analyze)

        return self.create_classification_result(
            label=intent,
            confidence=confidence,
            metadata={
                "method": "pattern_fallback",
                "intent_scores": intent_scores,
                "input_length": len(text),
                "warning": "Using simple fallback - install transformers for better accuracy",
            },
        )


class CachedIntentClassifier(CachedClassifier):
    """Cached version of the intent classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        threshold: float = 0.4,
        cache_size: int = 128,
        name: str = "cached_intent",
        description: str = "Cached intent classifier using pretrained models",
    ):
        """Initialize the cached intent classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = IntentClassifier(
            model_name=model_name,
            threshold=threshold,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform intent classification without caching."""
        return asyncio.run(self._classifier.classify_async(text))


# Factory function for easy creation
def create_intent_classifier(
    model_name: str = "microsoft/DialoGPT-medium",
    threshold: float = 0.4,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create an intent classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for intent detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured intent classifier
    """
    if cached:
        return CachedIntentClassifier(
            model_name=model_name,
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return IntentClassifier(
            model_name=model_name,
            threshold=threshold,
        )
