"""Intent classification for detecting the purpose/intent of text.

This module provides a classifier for detecting intent in text using pretrained models
from Hugging Face transformers.

Detects intents like question, request, statement, command, complaint, etc.

Requires transformers library to be installed.
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
    "microsoft/DialoGPT-small": {
        "description": "Lightweight conversational model for intent classification",
        "intents": [
            "question",
            "statement",
            "request",
            "greeting",
            "goodbye",
            "complaint",
            "compliment",
        ],
        "labels": {"LABEL_0": "statement", "LABEL_1": "question", "LABEL_2": "request"},
    },
    "facebook/bart-large-mnli": {
        "description": "BART model fine-tuned for natural language inference",
        "intents": [
            "question",
            "statement",
            "request",
            "greeting",
            "goodbye",
            "complaint",
            "compliment",
        ],
    },
}


class IntentClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting intent in text using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    for accurate intent detection. Requires transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for intent detection
        pipeline: The Hugging Face transformers pipeline
        intents: List of intents the model can detect
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-small",
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

        Raises:
            ImportError: If transformers library is not installed
            Exception: If model loading fails
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = INTENT_MODELS.get(model_name, {})
        self.intents = self.model_info.get(
            "intents",
            ["question", "statement", "request", "greeting", "goodbye", "complaint", "compliment"],
        )
        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return self.intents

    def _initialize_model(self) -> None:
        """Initialize the pretrained intent detection model."""
        # Import transformers - fail fast if not available
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
                result = await self._classify_with_pipeline(text)

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

            # Process results - handle different pipeline output formats
            if not results:
                raise ValueError("Pipeline returned empty results")

            # Check if results is a list of dictionaries or a different format
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                    # Standard format: list of dicts with score and label
                    best_result = max(results, key=lambda x: x["score"])
                    intent = best_result["label"].lower()
                    confidence = float(best_result["score"])
                else:
                    # Alternative format - try to extract from first result
                    if hasattr(results[0], "label") and hasattr(results[0], "score"):
                        intent = results[0].label.lower()
                        confidence = float(results[0].score)
                    else:
                        raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
            else:
                raise ValueError(f"Unexpected pipeline results type: {type(results)}")

            # Get all intents above threshold - handle different formats
            detected_intents = []
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and "score" in results[0] and "label" in results[0]:
                    # Standard format: list of dicts with score and label
                    detected_intents = [
                        {"intent": result["label"].lower(), "confidence": float(result["score"])}
                        for result in results
                        if float(result["score"]) >= self.threshold
                    ]
                else:
                    # Alternative format - only include the primary intent if above threshold
                    if confidence >= self.threshold:
                        detected_intents = [{"intent": intent, "confidence": confidence}]

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
            logger.error(
                f"Pipeline intent classification failed: {e}",
                extra={"classifier": self.name},
                exc_info=True,
            )
            raise


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
