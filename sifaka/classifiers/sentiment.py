"""Sentiment classifier using Hugging Face transformers.

This module provides classifiers for sentiment analysis using pretrained models
from Hugging Face transformers.
Requires transformers library to be installed.
"""

from typing import List
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


# Popular pretrained sentiment models
SENTIMENT_MODELS = {
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "description": "RoBERTa trained on Twitter data",
        "labels": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "description": "Multilingual BERT for sentiment",
        "labels": {
            "1 star": "negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "positive",
        },
    },
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "description": "DistilBERT fine-tuned on SST-2",
        "labels": {"NEGATIVE": "negative", "POSITIVE": "positive"},
    },
    "j-hartmann/emotion-english-distilroberta-base": {
        "description": "DistilRoBERTa for emotion classification",
        "labels": {
            "joy": "positive",
            "optimism": "positive",
            "love": "positive",
            "sadness": "negative",
            "anger": "negative",
            "fear": "negative",
            "pessimism": "negative",
        },
    },
}


class SentimentClassifier(BaseClassifier, TimingMixin):
    """Classifier for sentiment analysis using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers.
    Requires the transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        positive_threshold: Threshold above which sentiment is positive
        negative_threshold: Threshold below which sentiment is negative
        pipeline: The Hugging Face pipeline
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        name: str = "pretrained_sentiment",
        description: str = "Analyzes sentiment using pretrained models",
    ):
        """Initialize the pretrained sentiment classifier.

        Args:
            model_name: Name of the pretrained model to use
            positive_threshold: Threshold above which sentiment is positive
            negative_threshold: Threshold below which sentiment is negative
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.pipeline = None
        self.model_info = SENTIMENT_MODELS.get(model_name, {})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained sentiment analysis model."""
        try:
            import transformers

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
                f"Initialized sentiment classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                },
            )

        except ImportError as e:
            logger.error(
                "transformers library is required for sentiment classification",
                extra={"classifier": self.name},
            )
            raise ValidationError(
                "transformers library is required for sentiment classification. "
                "Install it with: pip install transformers",
                error_code="missing_dependency",
                context={
                    "classifier": self.name,
                    "required_library": "transformers",
                },
                suggestions=[
                    "Install transformers: pip install transformers",
                    "Or use uv: uv pip install transformers",
                ],
            ) from e

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for sentiment asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with sentiment prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("neutral")

        with self.time_operation("sentiment_classification") as timer:
            try:
                result = await self._classify_with_pipeline(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Sentiment classification completed",
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
                    f"Sentiment classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for sentiment: {str(e)}",
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
                        "Check internet connection for model download",
                    ],
                ) from e

    async def _classify_with_pipeline(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline."""

        # Run pipeline analysis in a thread to avoid blocking
        def analyze():
            results = self.pipeline(text)
            return results

        # Use asyncio to run in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, analyze)

        # Process results based on model type
        label_mapping = self.model_info.get("labels", {})

        # Find the best prediction
        best_result = max(results, key=lambda x: x["score"])
        raw_label = best_result["label"]
        confidence = best_result["score"]

        # Map to standard sentiment labels
        if raw_label in label_mapping:
            final_label = label_mapping[raw_label]
        else:
            # Try to infer from label name
            raw_lower = raw_label.lower()
            if (
                "pos" in raw_lower
                or "joy" in raw_lower
                or "love" in raw_lower
                or "optimism" in raw_lower
            ):
                final_label = "positive"
            elif (
                "neg" in raw_lower
                or "sad" in raw_lower
                or "anger" in raw_lower
                or "fear" in raw_lower
            ):
                final_label = "negative"
            else:
                final_label = "neutral"

        return self.create_classification_result(
            label=final_label,
            confidence=confidence,
            metadata={
                "method": "transformers_pipeline",
                "model_name": self.model_name,
                "raw_label": raw_label,
                "positive_threshold": self.positive_threshold,
                "negative_threshold": self.negative_threshold,
                "input_length": len(text),
                "all_scores": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["negative", "neutral", "positive"]


class CachedSentimentClassifier(CachedClassifier):
    """Cached version of the sentiment classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        cache_size: int = 128,
        name: str = "cached_sentiment",
        description: str = "Cached sentiment classifier using pretrained models",
    ):
        """Initialize the cached sentiment classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = SentimentClassifier(
            model_name=model_name,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform sentiment classification without caching."""
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an event loop, so we need to create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._classifier.classify_async(text))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self._classifier.classify_async(text))

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return self._classifier.get_classes()


# Factory function for easy creation
def create_sentiment_classifier(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    positive_threshold: float = 0.1,
    negative_threshold: float = -0.1,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a sentiment classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        positive_threshold: Threshold above which sentiment is positive
        negative_threshold: Threshold below which sentiment is negative
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured pretrained sentiment classifier
    """
    if cached:
        return CachedSentimentClassifier(
            model_name=model_name,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            cache_size=cache_size,
        )
    else:
        return SentimentClassifier(
            model_name=model_name,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
