"""Toxicity classifier using Hugging Face transformers.

This module provides classifiers for detecting toxic, harmful, or abusive
language using pretrained models from Hugging Face transformers with fallback
to rule-based detection. Designed for the new PydanticAI-based Sifaka architecture.
"""

import importlib
from typing import List, Optional, Set, Dict, Any
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

# Toxicity indicators for rule-based fallback
TOXIC_WORDS: Set[str] = {
    "hate",
    "kill",
    "die",
    "death",
    "murder",
    "violence",
    "hurt",
    "pain",
    "stupid",
    "idiot",
    "moron",
    "dumb",
    "worthless",
    "useless",
    "trash",
    "disgusting",
    "ugly",
    "loser",
    "failure",
    "pathetic",
    "weak",
}

SEVERE_TOXIC_WORDS: Set[str] = {
    "kill yourself",
    "kys",
    "suicide",
    "die in",
    "hope you die",
    "cancer",
    "suffer",
    "torture",
    "abuse",
    "violence upon",
}

THREAT_WORDS: Set[str] = {
    "i will kill",
    "i'll kill",
    "gonna kill",
    "going to hurt",
    "i will hurt",
    "i'll hurt",
    "gonna hurt",
    "beat you up",
    "kick your ass",
    "destroy you",
    "ruin your life",
}

# Popular pretrained toxicity models
TOXICITY_MODELS = {
    "unitary/toxic-bert-base": {
        "description": "BERT-based toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
    },
    "martin-ha/toxic-comment-model": {
        "description": "DistilBERT toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
    },
    "unitary/unbiased-toxic-roberta": {
        "description": "RoBERTa-based unbiased toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
    },
}


class ToxicityClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting toxic language using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    with fallback to rule-based toxicity detection. It identifies various
    forms of toxic language including hate speech, threats, and abuse.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for toxicity detection
        pipeline: The Hugging Face pipeline (if available)
        tokenizer: The tokenizer (if available)
        model: The model (if available)
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert-base",
        threshold: float = 0.7,
        name: str = "pretrained_toxicity",
        description: str = "Detects toxic language using pretrained models",
    ):
        """Initialize the pretrained toxicity classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for toxicity detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.model_info = TOXICITY_MODELS.get(model_name, {})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained toxicity detection model."""
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
                f"Initialized toxicity classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                },
            )

        except ImportError:
            logger.warning(
                "transformers not available. Trying manual model loading...",
                extra={"classifier": self.name},
            )

            try:
                # Try manual model loading
                transformers = importlib.import_module("transformers")

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )

                logger.debug(
                    f"Initialized toxicity classifier with manual model loading",
                    extra={
                        "classifier": self.name,
                        "model_name": self.model_name,
                        "method": "transformers_manual",
                    },
                )

            except (ImportError, Exception) as e:
                logger.warning(
                    f"Transformers not available or model loading failed: {e}. "
                    "Using rule-based toxicity detection. "
                    "Install transformers for better accuracy: pip install transformers",
                    extra={"classifier": self.name},
                )
                self.pipeline = None
                self.tokenizer = None
                self.model = None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for toxicity asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with toxicity prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("non_toxic")

        with self.time_operation("toxicity_classification") as timer:
            try:
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                elif self.model is not None and self.tokenizer is not None:
                    result = await self._classify_with_manual_model(text)
                else:
                    result = await self._classify_with_rules(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Toxicity classification completed",
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
                    f"Toxicity classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for toxicity: {str(e)}",
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

            # Process results
            label_mapping = self.model_info.get("labels", {})

            # Find the toxic/non-toxic scores
            toxic_score = 0.0
            non_toxic_score = 0.0

            for result in results:
                label = result["label"]
                score = result["score"]

                if label in ["TOXIC", "toxic", "1"]:
                    toxic_score = score
                elif label in ["NON_TOXIC", "non_toxic", "0"]:
                    non_toxic_score = score

            # Determine final label and confidence
            if toxic_score > non_toxic_score and toxic_score >= self.threshold:
                final_label = "toxic"
                confidence = toxic_score
            else:
                final_label = "non_toxic"
                confidence = non_toxic_score

            return self.create_classification_result(
                label=final_label,
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "toxic_score": toxic_score,
                    "non_toxic_score": non_toxic_score,
                    "threshold": self.threshold,
                    "input_length": len(text),
                    "all_scores": results,
                },
            )

        except Exception as e:
            # Fallback to rule-based analysis
            logger.warning(
                f"Pipeline toxicity classification failed, using rule-based analysis: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_rules(text)

    async def _classify_with_manual_model(self, text: str) -> ClassificationResult:
        """Classify using manual model and tokenizer."""
        try:
            # Run manual model analysis in a thread to avoid blocking
            def analyze():
                import torch

                # Tokenize input
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                return predictions.numpy()[0]

            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(None, analyze)

            # Assuming binary classification: [non_toxic, toxic]
            non_toxic_score = float(predictions[0])
            toxic_score = float(predictions[1])

            # Determine final label and confidence
            if toxic_score >= self.threshold:
                final_label = "toxic"
                confidence = toxic_score
            else:
                final_label = "non_toxic"
                confidence = non_toxic_score

            return self.create_classification_result(
                label=final_label,
                confidence=confidence,
                metadata={
                    "method": "transformers_manual",
                    "model_name": self.model_name,
                    "toxic_score": toxic_score,
                    "non_toxic_score": non_toxic_score,
                    "threshold": self.threshold,
                    "input_length": len(text),
                },
            )

        except Exception as e:
            # Fallback to rule-based analysis
            logger.warning(
                f"Manual model toxicity classification failed, using rule-based analysis: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_rules(text)

    async def _classify_with_rules(self, text: str) -> ClassificationResult:
        """Classify using rule-based approach as fallback."""

        def analyze():
            text_lower = text.lower()

            # Count different types of toxic content
            toxic_count = sum(1 for word in TOXIC_WORDS if word in text_lower)
            severe_count = sum(1 for phrase in SEVERE_TOXIC_WORDS if phrase in text_lower)
            threat_count = sum(1 for phrase in THREAT_WORDS if phrase in text_lower)

            # Calculate toxicity scores
            words = text_lower.split()
            word_count = len(words)

            toxic_score = toxic_count / max(1, word_count) * 2
            severe_score = severe_count * 0.5
            threat_score = threat_count * 0.7

            total_score = toxic_score + severe_score + threat_score

            return toxic_count, severe_count, threat_count, total_score, word_count

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        toxic_count, severe_count, threat_count, total_score, word_count = (
            await loop.run_in_executor(None, analyze)
        )

        # Determine label and confidence
        if total_score > 0.3:  # Lower threshold for rule-based
            label = "toxic"
            confidence = min(0.5 + total_score * 0.4, 0.85)
        else:
            label = "non_toxic"
            confidence = max(0.7, 0.95 - total_score)

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "rule_based_fallback",
                "toxic_words": toxic_count,
                "severe_words": severe_count,
                "threat_words": threat_count,
                "toxicity_score": total_score,
                "total_words": word_count,
                "threshold": self.threshold,
                "input_length": len(text),
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["non_toxic", "toxic"]


class CachedToxicityClassifier(CachedClassifier):
    """Cached version of the toxicity classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert-base",
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "cached_toxicity",
        description: str = "Cached toxicity classifier using pretrained models",
    ):
        """Initialize the cached toxicity classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = ToxicityClassifier(
            model_name=model_name,
            threshold=threshold,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform toxicity classification without caching."""
        return asyncio.run(self._classifier.classify_async(text))


# Factory function for easy creation
def create_toxicity_classifier(
    model_name: str = "unitary/toxic-bert-base",
    threshold: float = 0.7,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a toxicity classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for toxicity detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured pretrained toxicity classifier
    """
    if cached:
        return CachedToxicityClassifier(
            model_name=model_name,
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return ToxicityClassifier(
            model_name=model_name,
            threshold=threshold,
        )
