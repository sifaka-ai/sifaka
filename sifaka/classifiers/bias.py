"""Bias detection classifier for identifying various types of bias in text.

This module provides a classifier for detecting bias in text using
pretrained models from Hugging Face transformers.

Requires transformers library to be installed.
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

# Available pretrained bias detection models
BIAS_MODELS: Dict[str, Dict[str, Any]] = {
    "d4data/bias-detection-model": {
        "description": "English sequence classification model trained on MBAD Dataset",
        "labels": {"LABEL_0": "unbiased", "LABEL_1": "biased"},
        "dataset": "MBAD",
        "size": "base",
    },
    "valurank/distilroberta-bias": {
        "description": "DistilRoBERTa fine-tuned for bias detection",
        "labels": {"LABEL_0": "unbiased", "LABEL_1": "biased"},
        "dataset": "custom",
        "size": "small",
    },
    "D1V1DE/bias-detection": {
        "description": "BERT-based bias detection model",
        "labels": {"LABEL_0": "unbiased", "LABEL_1": "biased"},
        "dataset": "custom",
        "size": "base",
    },
}


class BiasClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting bias in text using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    to identify various types of bias in text. Requires the transformers
    library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for bias detection
        pipeline: The Hugging Face pipeline for classification
        model_info: Information about the selected model
    """

    def __init__(
        self,
        model_name: str = "d4data/bias-detection-model",
        threshold: float = 0.7,
        name: str = "bias",
        description: str = "Detects bias in text using pretrained models",
    ):
        """Initialize the bias detection classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for bias detection
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = BIAS_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["unbiased", "biased"]

    def _initialize_model(self) -> None:
        """Initialize the pretrained bias detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for bias classification",
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
            f"Initialized bias classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for bias asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with bias prediction
        """
        if not text or not text.strip():
            return self.create_empty_text_result("unbiased")

        with self.time_operation("bias_classification") as timer:
            try:
                result = await self._classify_with_transformers(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Bias classification completed",
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
                    f"Bias classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for bias: {str(e)}",
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
        label_mapping = self.model_info.get("labels", {"LABEL_0": "unbiased", "LABEL_1": "biased"})

        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        raw_label = best_result["label"]
        confidence = float(best_result["score"])

        # Map model label to our standard labels
        label = label_mapping.get(raw_label, raw_label.lower())

        # Calculate probabilities for both classes
        biased_prob = 0.0
        unbiased_prob = 0.0
        for result in results:
            mapped_label = label_mapping.get(result["label"], result["label"].lower())
            if mapped_label == "biased":
                biased_prob = float(result["score"])
            elif mapped_label == "unbiased":
                unbiased_prob = float(result["score"])

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "biased_probability": biased_prob,
                "unbiased_probability": unbiased_prob,
                "threshold": self.threshold,
                "input_length": len(text),
                "raw_results": results,
            },
        )


class CachedBiasClassifier(CachedClassifier, TimingMixin):
    """Cached version of BiasClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        model_name: str = "d4data/bias-detection-model",
        threshold: float = 0.7,
        cache_size: int = 128,
        name: str = "cached_bias",
        description: str = "Detects bias content with LRU caching",
    ):
        """Initialize the cached bias classifier.

        Args:
            model_name: Name of the pretrained model to use
            threshold: Confidence threshold for bias detection
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.model_info = BIAS_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained bias detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for bias classification",
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
            f"Initialized cached bias classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform bias classification without caching."""
        try:
            return self._classify_with_transformers_sync(text)
        except Exception as e:
            logger.error(
                f"Cached bias classification failed",
                extra={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise ValidationError(
                f"Failed to classify text for bias: {str(e)}",
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
        label_mapping = self.model_info.get("labels", {"LABEL_0": "unbiased", "LABEL_1": "biased"})

        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        raw_label = best_result["label"]
        confidence = float(best_result["score"])

        # Map model label to our standard labels
        label = label_mapping.get(raw_label, raw_label.lower())

        # Calculate probabilities for both classes
        biased_prob = 0.0
        unbiased_prob = 0.0
        for result in results:
            mapped_label = label_mapping.get(result["label"], result["label"].lower())
            if mapped_label == "biased":
                biased_prob = float(result["score"])
            elif mapped_label == "unbiased":
                unbiased_prob = float(result["score"])

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "biased_probability": biased_prob,
                "unbiased_probability": unbiased_prob,
                "threshold": self.threshold,
                "input_length": len(text),
                "cached": True,
                "raw_results": results,
            },
        )

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return ["unbiased", "biased"]


# Factory function for easy creation
def create_bias_classifier(
    model_name: str = "d4data/bias-detection-model",
    threshold: float = 0.7,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a bias classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for bias detection
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured bias classifier
    """
    if cached:
        return CachedBiasClassifier(
            model_name=model_name,
            threshold=threshold,
            cache_size=cache_size,
        )
    else:
        return BiasClassifier(
            model_name=model_name,
            threshold=threshold,
        )
