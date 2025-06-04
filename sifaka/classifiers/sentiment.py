"""Sentiment classifier using Hugging Face transformers.

This module provides classifiers for sentiment analysis using pretrained models
from Hugging Face transformers with fallback to TextBlob/lexicon-based analysis.
Designed for the new PydanticAI-based Sifaka architecture.
"""

import importlib
from typing import List, Set
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

# Simple sentiment lexicons for fallback
POSITIVE_WORDS: Set[str] = {
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "awesome",
    "love",
    "like",
    "enjoy",
    "happy",
    "pleased",
    "satisfied",
    "delighted",
    "perfect",
    "brilliant",
    "outstanding",
    "superb",
    "magnificent",
    "beautiful",
    "best",
    "better",
    "positive",
    "nice",
    "fine",
    "glad",
    "thrilled",
    "excited",
    "incredible",
    "marvelous",
    "splendid",
    "terrific",
    "fabulous",
    "remarkable",
}

NEGATIVE_WORDS: Set[str] = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "disgusting",
    "hate",
    "dislike",
    "angry",
    "sad",
    "disappointed",
    "frustrated",
    "annoyed",
    "upset",
    "mad",
    "worst",
    "worse",
    "negative",
    "poor",
    "ugly",
    "stupid",
    "boring",
    "wrong",
    "fail",
    "failed",
    "broken",
    "useless",
    "worthless",
    "pathetic",
    "dreadful",
    "appalling",
    "atrocious",
    "abysmal",
    "deplorable",
    "miserable",
    "wretched",
}

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

    This classifier uses pretrained models from Hugging Face transformers
    with fallback to TextBlob or lexicon-based sentiment analysis.

    Attributes:
        model_name: Name of the pretrained model to use
        positive_threshold: Threshold above which sentiment is positive
        negative_threshold: Threshold below which sentiment is negative
        pipeline: The Hugging Face pipeline (if available)
        tokenizer: The tokenizer (if available)
        model: The model (if available)
        textblob: The TextBlob library instance (if available)
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
        self.tokenizer = None
        self.model = None
        self.textblob = None
        self.model_info = SENTIMENT_MODELS.get(model_name, {})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained sentiment analysis model."""
        try:
            # Try to use transformers pipeline first (easiest)
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
                f"Initialized sentiment classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                },
            )

        except ImportError:
            logger.warning(
                "transformers not available. Trying TextBlob...", extra={"classifier": self.name}
            )

            try:
                # Try TextBlob as fallback
                self.textblob = importlib.import_module("textblob")

                logger.debug(
                    f"Initialized sentiment classifier with TextBlob",
                    extra={
                        "classifier": self.name,
                        "method": "textblob",
                    },
                )

            except ImportError:
                logger.warning(
                    "Neither transformers nor TextBlob available. "
                    "Using lexicon-based sentiment analysis. "
                    "Install transformers or TextBlob for better accuracy: "
                    "pip install transformers or pip install textblob",
                    extra={"classifier": self.name},
                )
                self.pipeline = None
                self.tokenizer = None
                self.model = None
                self.textblob = None

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
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                elif self.textblob is not None:
                    result = await self._classify_with_textblob(text)
                else:
                    result = await self._classify_with_lexicon(text)

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
        try:
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

        except Exception as e:
            # Fallback to TextBlob or lexicon
            logger.warning(
                f"Pipeline sentiment classification failed, using fallback: {e}",
                extra={"classifier": self.name},
            )
            if self.textblob is not None:
                return await self._classify_with_textblob(text)
            else:
                return await self._classify_with_lexicon(text)

    async def _classify_with_textblob(self, text: str) -> ClassificationResult:
        """Classify using TextBlob sentiment analysis."""
        try:
            # Run TextBlob analysis in a thread to avoid blocking
            def analyze():
                blob = self.textblob.TextBlob(text)
                return blob.sentiment.polarity, blob.sentiment.subjectivity

            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            polarity, subjectivity = await loop.run_in_executor(None, analyze)

            # Determine label based on thresholds
            if polarity > self.positive_threshold:
                label = "positive"
                confidence = min(0.5 + (polarity * 0.5), 0.95)
            elif polarity < self.negative_threshold:
                label = "negative"
                confidence = min(0.5 + (abs(polarity) * 0.5), 0.95)
            else:
                label = "neutral"
                confidence = 0.7 - abs(polarity)  # More neutral = higher confidence

            return self.create_classification_result(
                label=label,
                confidence=confidence,
                metadata={
                    "method": "textblob_fallback",
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "positive_threshold": self.positive_threshold,
                    "negative_threshold": self.negative_threshold,
                    "input_length": len(text),
                },
            )

        except Exception as e:
            # Final fallback to lexicon
            logger.warning(
                f"TextBlob sentiment classification failed, using lexicon: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_lexicon(text)

    async def _classify_with_lexicon(self, text: str) -> ClassificationResult:
        """Classify using simple lexicon-based approach."""

        def analyze():
            text_lower = text.lower()
            words = text_lower.split()

            # Count positive and negative words
            positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
            negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)

            # Calculate sentiment score
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                sentiment_score = 0.0
            else:
                sentiment_score = (positive_count - negative_count) / len(words)

            return positive_count, negative_count, sentiment_score, len(words)

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        positive_count, negative_count, sentiment_score, word_count = await loop.run_in_executor(
            None, analyze
        )

        # Determine label and confidence
        if sentiment_score > 0.05:  # More lenient threshold for lexicon
            label = "positive"
            confidence = min(0.6 + (sentiment_score * 2), 0.85)
        elif sentiment_score < -0.05:
            label = "negative"
            confidence = min(0.6 + (abs(sentiment_score) * 2), 0.85)
        else:
            label = "neutral"
            confidence = 0.7

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "lexicon_based_fallback",
                "sentiment_score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": word_count,
                "positive_threshold": self.positive_threshold,
                "negative_threshold": self.negative_threshold,
                "input_length": len(text),
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
