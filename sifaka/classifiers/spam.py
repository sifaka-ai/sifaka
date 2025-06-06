"""Spam classifier for detecting spam content in text.

This module provides a classifier for detecting spam content using
pretrained models from Hugging Face transformers.
"""

from typing import List, Dict, Any
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

# Available pretrained models for spam detection
SPAM_MODELS: Dict[str, Dict[str, Any]] = {
    "mrm8488/bert-tiny-finetuned-sms-spam-detection": {
        "description": "BERT-Tiny fine-tuned on SMS spam dataset",
        "labels": {"LABEL_0": "ham", "LABEL_1": "spam"},
        "accuracy": 0.98,
        "size": "tiny",
    },
    "ggrizzly/roBERTa-spam-detection": {
        "description": "RoBERTa-based spam detection model",
        "labels": {"LABEL_0": "ham", "LABEL_1": "spam"},
        "accuracy": 0.95,
        "size": "base",
    },
    "AntiSpamInstitute/spam-detector-bert-MoE-v2.2": {
        "description": "BERT with Mixture of Experts for spam detection",
        "labels": {"LABEL_0": "ham", "LABEL_1": "spam"},
        "accuracy": 0.97,
        "size": "large",
    },
}


class SpamClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting spam content in text.

    This classifier uses pretrained models from Hugging Face transformers
    to identify spam content. Requires the transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for spam detection
        pipeline: The Hugging Face pipeline for classification
        model_info: Information about the selected model
    """

    def __init__(
        self,
        model_name: str = "mrm8488/bert-tiny-finetuned-sms-spam-detection",
        threshold: float = 0.7,
        name: str = "spam",
        description: str = "Detects spam content using pretrained models",
    ):
        """Initialize the spam classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for spam detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = SPAM_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained spam detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for spam classification",
                error_code="dependency_missing",
                suggestions=["Install transformers: pip install transformers"],
            ) from e

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
            f"Initialized spam classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for spam asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with spam prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("ham")

        with self.time_operation("spam_classification") as timer:
            try:
                result = await self._classify_with_transformers(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Spam classification completed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "label": result.label,
                        "confidence": result.confidence,
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"Spam classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for spam: {str(e)}",
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
                        "Check if the model is available on Hugging Face",
                    ],
                ) from e

    async def _classify_with_transformers(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline."""
        if self.pipeline is None:
            raise ValidationError(
                "Transformers pipeline is not available",
                error_code="dependency_missing",
                suggestions=["Install transformers: pip install transformers"],
            )

        # Run transformers analysis in a thread to avoid blocking
        def analyze():
            results = self.pipeline(text)
            return results

        # Use asyncio to run in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, analyze)

        # Process results - transformers returns list of dicts with label and score
        label_mapping = self.model_info.get("labels", {"LABEL_0": "ham", "LABEL_1": "spam"})

        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        raw_label = best_result["label"]
        confidence = float(best_result["score"])

        # Map model label to our standard labels
        label = label_mapping.get(raw_label, raw_label.lower())

        # Calculate probabilities for both classes
        spam_prob = 0.0
        ham_prob = 0.0
        for result in results:
            mapped_label = label_mapping.get(result["label"], result["label"].lower())
            if mapped_label == "spam":
                spam_prob = float(result["score"])
            elif mapped_label == "ham":
                ham_prob = float(result["score"])

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "spam_probability": spam_prob,
                "ham_probability": ham_prob,
                "threshold": self.threshold,
                "input_length": len(text),
                "raw_results": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["ham", "spam"]


class CachedSpamClassifier(CachedClassifier, TimingMixin):
    """Cached version of SpamClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        model_name: str = "mrm8488/bert-tiny-finetuned-sms-spam-detection",
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "cached_spam",
        description: str = "Detects spam content with LRU caching",
    ):
        """Initialize the cached spam classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for spam detection
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = SPAM_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained spam detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for spam classification",
                error_code="dependency_missing",
                suggestions=["Install transformers: pip install transformers"],
            ) from e

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
            f"Initialized cached spam classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform spam classification without caching."""
        try:
            return self._classify_with_transformers_sync(text)
        except Exception as e:
            logger.error(
                f"Cached spam classification failed",
                extra={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to classify text for spam: {str(e)}",
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
                    "Check if the model is available on Hugging Face",
                ],
            ) from e

    def _classify_with_transformers_sync(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline (synchronous)."""
        if self.pipeline is None:
            raise ValidationError(
                "Transformers pipeline is not available",
                error_code="dependency_missing",
                suggestions=["Install transformers: pip install transformers"],
            )

        # Run transformers analysis
        results = self.pipeline(text)

        # Process results - transformers returns list of dicts with label and score
        label_mapping = self.model_info.get("labels", {"LABEL_0": "ham", "LABEL_1": "spam"})

        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        raw_label = best_result["label"]
        confidence = float(best_result["score"])

        # Map model label to our standard labels
        label = label_mapping.get(raw_label, raw_label.lower())

        # Calculate probabilities for both classes
        spam_prob = 0.0
        ham_prob = 0.0
        for result in results:
            mapped_label = label_mapping.get(result["label"], result["label"].lower())
            if mapped_label == "spam":
                spam_prob = float(result["score"])
            elif mapped_label == "ham":
                ham_prob = float(result["score"])

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "spam_probability": spam_prob,
                "ham_probability": ham_prob,
                "threshold": self.threshold,
                "input_length": len(text),
                "cached": True,
                "raw_results": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["ham", "spam"]


# Factory functions for easy creation
def create_spam_classifier(
    model_name: str = "mrm8488/bert-tiny-finetuned-sms-spam-detection",
    threshold: float = 0.7,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a spam classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for spam detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured spam classifier
    """
    if cached:
        return CachedSpamClassifier(
            model_name=model_name,
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return SpamClassifier(
            model_name=model_name,
            threshold=threshold,
        )
