"""Toxicity classifier using Hugging Face transformers.

This module provides classifiers for detecting toxic, harmful, or abusive
language using pretrained models from Hugging Face transformers.
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

# Available pretrained toxicity models
TOXICITY_MODELS: Dict[str, Dict[str, Any]] = {
    "unitary/toxic-bert-base": {
        "description": "BERT-based toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
        "accuracy": 0.92,
        "size": "base",
    },
    "martin-ha/toxic-comment-model": {
        "description": "DistilBERT toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
        "accuracy": 0.89,
        "size": "small",
    },
    "unitary/unbiased-toxic-roberta": {
        "description": "RoBERTa-based unbiased toxicity classifier",
        "labels": {"TOXIC": "toxic", "NON_TOXIC": "non_toxic"},
        "accuracy": 0.94,
        "size": "base",
    },
}


class ToxicityClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting toxic language using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    to identify various forms of toxic language including hate speech,
    threats, and abuse. Requires the transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for toxicity detection
        pipeline: The Hugging Face pipeline for classification
        model_info: Information about the selected model
    """

    def __init__(
        self,
        model_name: str = "martin-ha/toxic-comment-model",
        threshold: float = 0.7,
        name: str = "toxicity",
        description: str = "Detects toxic language using pretrained models",
    ):
        """Initialize the toxicity classifier.

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
        self.model_info = TOXICITY_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained toxicity detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for toxicity classification",
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
            f"Initialized toxicity classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

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
                result = await self._classify_with_transformers(text)

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

        # Run pipeline analysis in a thread to avoid blocking
        def analyze():
            results = self.pipeline(text)
            return results

        # Use asyncio to run in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, analyze)

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
                "model_name": self.model_name,
                "toxic_score": toxic_score,
                "non_toxic_score": non_toxic_score,
                "threshold": self.threshold,
                "input_length": len(text),
                "raw_results": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["non_toxic", "toxic"]


class CachedToxicityClassifier(CachedClassifier, TimingMixin):
    """Cached version of ToxicityClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        model_name: str = "martin-ha/toxic-comment-model",
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "cached_toxicity",
        description: str = "Detects toxic content with LRU caching",
    ):
        """Initialize the cached toxicity classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for toxicity detection
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = TOXICITY_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained toxicity detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for toxicity classification",
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
            f"Initialized cached toxicity classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform toxicity classification without caching."""
        try:
            return self._classify_with_transformers_sync(text)
        except Exception as e:
            logger.error(
                f"Cached toxicity classification failed",
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
                "model_name": self.model_name,
                "toxic_score": toxic_score,
                "non_toxic_score": non_toxic_score,
                "threshold": self.threshold,
                "input_length": len(text),
                "cached": True,
                "raw_results": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["non_toxic", "toxic"]


# Factory function for easy creation
def create_toxicity_classifier(
    model_name: str = "martin-ha/toxic-comment-model",
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
