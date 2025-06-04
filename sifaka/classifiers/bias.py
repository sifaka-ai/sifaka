"""Bias detection classifier for identifying various types of bias in text.

This module provides a classifier for detecting bias in text using the detoxify library
and pretrained models. Designed for the new PydanticAI-based Sifaka architecture.

Uses detoxify library for accurate bias and toxicity detection without hardcoded patterns.
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

# Popular pretrained bias detection models
BIAS_MODELS = {
    "unitary/unbiased-toxic-roberta": {
        "description": "RoBERTa model for unbiased toxicity and bias detection",
        "labels": {"BIASED": "biased", "UNBIASED": "unbiased"},
    },
    "martin-ha/bias-detection-model": {
        "description": "BERT-based general bias detection model",
        "labels": {"BIAS": "biased", "NO_BIAS": "unbiased"},
    },
}


class BiasClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting bias in text using detoxify and pretrained models.

    This classifier uses the detoxify library for comprehensive bias detection
    with fallback to pretrained transformers models.

    Attributes:
        model_name: Name of the pretrained model to use
        threshold: Confidence threshold for bias detection
        detoxify_model: The detoxify model instance (if available)
        pipeline: The Hugging Face pipeline (if available)
    """

    def __init__(
        self,
        model_name: str = "unitary/unbiased-toxic-roberta",
        threshold: float = 0.7,
        detoxify_model: str = "unbiased",
        name: str = "bias_detection",
        description: str = "Detects various types of bias in text using detoxify and pretrained models",
    ):
        """Initialize the bias detection classifier.

        Args:
            model_name: Name of the pretrained model to use as fallback
            threshold: Confidence threshold for bias detection
            detoxify_model: Detoxify model variant ("original", "unbiased", "multilingual")
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.threshold = threshold
        self.detoxify_model_name = detoxify_model
        self.detoxify_model = None
        self.pipeline = None
        self.model_info = BIAS_MODELS.get(model_name, {})
        self._initialize_models()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return ["biased", "unbiased"]

    def _initialize_models(self) -> None:
        """Initialize the detoxify model and fallback transformers model."""
        # Try to initialize detoxify first (preferred)
        try:
            detoxify = importlib.import_module("detoxify")
            self.detoxify_model = detoxify.Detoxify(self.detoxify_model_name)

            logger.debug(
                f"Initialized bias classifier with detoxify",
                extra={
                    "classifier": self.name,
                    "detoxify_model": self.detoxify_model_name,
                    "method": "detoxify",
                },
            )

        except (ImportError, Exception) as e:
            logger.warning(
                f"Detoxify not available: {e}. "
                "Install with: pip install detoxify. "
                "Falling back to transformers.",
                extra={"classifier": self.name},
            )
            self.detoxify_model = None

        # Try to initialize transformers as fallback
        if self.detoxify_model is None:
            try:
                transformers = importlib.import_module("transformers")
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
                    },
                )

            except (ImportError, Exception) as e:
                logger.warning(
                    f"Transformers not available or model loading failed: {e}. "
                    "Using simple fallback detection. "
                    "Install transformers for better accuracy: pip install transformers",
                    extra={"classifier": self.name},
                )
                self.pipeline = None

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
                if self.detoxify_model is not None:
                    result = await self._classify_with_detoxify(text)
                elif self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                else:
                    result = await self._classify_with_fallback(text)

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
                        "method": result.metadata.get("method", "unknown"),
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
                        "Check if detoxify or transformers is properly installed",
                        "Verify input text is valid",
                        "Try with shorter text",
                    ],
                ) from e

    async def _classify_with_detoxify(self, text: str) -> ClassificationResult:
        """Classify using detoxify library."""

        def analyze():
            results = self.detoxify_model.predict(text)
            return results

        # Use asyncio to run in thread pool for CPU-bound work
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, analyze)

        # Detoxify returns scores for different categories
        # We'll use the highest score to determine overall bias
        max_score = 0.0
        max_category = "unbiased"

        bias_categories = []
        for category, score in results.items():
            if score > self.threshold:
                bias_categories.append(category)
            if score > max_score:
                max_score = score
                max_category = category

        # Determine if biased based on any category exceeding threshold
        is_biased = len(bias_categories) > 0
        label = "biased" if is_biased else "unbiased"
        confidence = max_score if is_biased else (1.0 - max_score)

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "detoxify",
                "detoxify_model": self.detoxify_model_name,
                "threshold": self.threshold,
                "input_length": len(text),
                "all_scores": results,
                "bias_categories_detected": bias_categories,
                "highest_score_category": max_category,
            },
        )

    async def _classify_with_pipeline(self, text: str) -> ClassificationResult:
        """Classify using transformers pipeline."""

        def analyze():
            results = self.pipeline(text)
            return results

        try:
            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, analyze)

            # Process results
            best_result = max(results, key=lambda x: x["score"])
            raw_label = best_result["label"]
            confidence = float(best_result["score"])

            # Map model labels to our standard labels
            label_mapping = self.model_info.get("labels", {})
            final_label = label_mapping.get(raw_label, raw_label.lower())

            # Determine if biased based on threshold
            is_biased = final_label == "biased" and confidence >= self.threshold

            return self.create_classification_result(
                label="biased" if is_biased else "unbiased",
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "raw_label": raw_label,
                    "threshold": self.threshold,
                    "input_length": len(text),
                    "all_scores": results,
                },
            )

        except Exception as e:
            # Fallback to simple analysis
            logger.warning(
                f"Pipeline bias classification failed, using simple fallback: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_fallback(text)

    async def _classify_with_fallback(self, text: str) -> ClassificationResult:
        """Simple fallback classification based on basic heuristics."""

        def analyze():
            # Very simple bias indicators
            text_lower = text.lower()

            # Look for obvious bias keywords
            bias_keywords = [
                "all women",
                "all men",
                "typical woman",
                "typical man",
                "all blacks",
                "all whites",
                "those people",
                "you people",
                "all muslims",
                "all christians",
                "all jews",
                "all liberals",
                "all conservatives",
            ]

            bias_count = sum(1 for keyword in bias_keywords if keyword in text_lower)

            # Simple scoring
            if bias_count > 0:
                confidence = min(0.8, 0.5 + bias_count * 0.1)
                label = "biased"
            else:
                confidence = 0.7
                label = "unbiased"

            return label, confidence, bias_count

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        label, confidence, bias_count = await loop.run_in_executor(None, analyze)

        return self.create_classification_result(
            label=label,
            confidence=confidence,
            metadata={
                "method": "simple_fallback",
                "bias_indicators_found": bias_count,
                "input_length": len(text),
                "warning": "Using simple fallback - install detoxify for better accuracy",
            },
        )


class CachedBiasClassifier(CachedClassifier):
    """Cached version of the bias classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "unitary/unbiased-toxic-roberta",
        threshold: float = 0.7,
        detoxify_model: str = "unbiased",
        cache_size: int = 128,
        name: str = "cached_bias",
        description: str = "Cached bias classifier using detoxify and pretrained models",
    ):
        """Initialize the cached bias classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = BiasClassifier(
            model_name=model_name,
            threshold=threshold,
            detoxify_model=detoxify_model,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform bias classification without caching."""
        return asyncio.run(self._classifier.classify_async(text))


# Factory function for easy creation
def create_bias_classifier(
    model_name: str = "unitary/unbiased-toxic-roberta",
    threshold: float = 0.7,
    detoxify_model: str = "unbiased",
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a bias classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use as fallback
        threshold: Confidence threshold for bias detection
        detoxify_model: Detoxify model variant ("original", "unbiased", "multilingual")
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured bias classifier
    """
    if cached:
        return CachedBiasClassifier(
            model_name=model_name,
            threshold=threshold,
            detoxify_model=detoxify_model,
            cache_size=cache_size,
        )
    else:
        return BiasClassifier(
            model_name=model_name,
            threshold=threshold,
            detoxify_model=detoxify_model,
        )
