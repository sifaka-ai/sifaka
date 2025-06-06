"""Readability classifier for assessing text complexity and reading level.

This module provides a classifier for determining text readability using pretrained models
from Hugging Face transformers.

Requires transformers library to be installed.
"""

import importlib
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

# Popular pretrained readability models
READABILITY_MODELS = {
    "textattack/roberta-base-CoLA": {
        "description": "RoBERTa model for linguistic acceptability (complexity)",
        "labels": {"LABEL_0": "complex", "LABEL_1": "simple"},
    },
    "microsoft/DialoGPT-medium": {
        "description": "General text complexity assessment model",
        "labels": {"SIMPLE": "simple", "COMPLEX": "complex"},
    },
}


class ReadabilityClassifier(BaseClassifier, TimingMixin):
    """Classifier for assessing text readability using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    for accurate readability assessment. Requires transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        grade_levels: List of grade level categories
        pipeline: The Hugging Face transformers pipeline
        tokenizer: The tokenizer (if available)
        model: The model (if available)
    """

    def __init__(
        self,
        model_name: str = "textattack/roberta-base-CoLA",
        grade_levels: List[str] = None,
        name: str = "readability",
        description: str = "Assesses text readability and complexity using pretrained models",
    ):
        """Initialize the readability classifier.

        Args:
            model_name: Name of the pretrained model to use
            grade_levels: List of grade level categories
            name: Name of the classifier
            description: Description of the classifier

        Raises:
            ImportError: If transformers library is not installed
            Exception: If model loading fails
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.grade_levels = grade_levels or ["elementary", "middle", "high", "college", "graduate"]
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.model_info = READABILITY_MODELS.get(model_name, {})
        self._initialize_model()

    def get_classes(self) -> list[str]:
        """Get the list of possible class labels."""
        return ["simple", "moderate", "complex"]

    def _initialize_model(self) -> None:
        """Initialize the pretrained readability model."""
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
            f"Initialized readability classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
                "grade_levels": self.grade_levels,
            },
        )

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for readability asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with readability assessment
        """
        if not text or not text.strip():
            return self.create_empty_text_result("unknown")

        with self.time_operation("readability_classification") as timer:
            try:
                result = await self._classify_with_pipeline(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Readability classification completed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "label": result.label,
                        "confidence": result.confidence,
                        "method": result.metadata.get("method", "unknown"),
                        "grade_level": result.metadata.get("grade_level", "unknown"),
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"Readability classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                raise ValidationError(
                    f"Failed to classify text for readability: {str(e)}",
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
                    raw_label = best_result["label"]
                    confidence = float(best_result["score"])
                else:
                    # Alternative format - try to extract from first result
                    if hasattr(results[0], "label") and hasattr(results[0], "score"):
                        raw_label = results[0].label
                        confidence = float(results[0].score)
                    else:
                        raise ValueError(f"Unexpected pipeline result format: {type(results[0])}")
            else:
                raise ValueError(f"Unexpected pipeline results type: {type(results)}")

            # Map model labels to our standard labels
            label_mapping = self.model_info.get("labels", {})
            final_label = label_mapping.get(raw_label, raw_label.lower())

            # Determine grade level based on complexity
            grade_level = self._determine_grade_level(final_label, confidence)

            return self.create_classification_result(
                label=final_label,
                confidence=confidence,
                metadata={
                    "method": "transformers_pipeline",
                    "model_name": self.model_name,
                    "raw_label": raw_label,
                    "grade_level": grade_level,
                    "input_length": len(text),
                    "all_scores": results,
                },
            )

        except Exception as e:
            logger.error(
                f"Pipeline readability classification failed: {e}",
                extra={"classifier": self.name},
                exc_info=True,
            )
            raise

    def _determine_grade_level(self, complexity: str, confidence: float) -> str:
        """Determine grade level based on complexity assessment."""
        if complexity in ["simple", "easy"]:
            return "elementary" if confidence > 0.8 else "middle"
        elif complexity in ["medium", "moderate"]:
            return "middle" if confidence > 0.7 else "high"
        elif complexity in ["complex", "hard", "difficult"]:
            return "college" if confidence > 0.8 else "high"
        else:
            return "middle"  # Default fallback


class CachedReadabilityClassifier(CachedClassifier):
    """Cached version of the readability classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "textattack/roberta-base-CoLA",
        grade_levels: List[str] = None,
        cache_size: int = 128,
        name: str = "cached_readability",
        description: str = "Cached readability classifier using pretrained models",
    ):
        """Initialize the cached readability classifier."""
        super().__init__(name=name, description=description, cache_size=cache_size)
        self._classifier = ReadabilityClassifier(
            model_name=model_name,
            grade_levels=grade_levels,
            name=f"base_{name}",
            description=f"Base classifier for {description}",
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform readability classification without caching."""
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
def create_readability_classifier(
    model_name: str = "textattack/roberta-base-CoLA",
    grade_levels: List[str] = None,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a readability classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        grade_levels: List of grade level categories
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured readability classifier
    """
    if cached:
        return CachedReadabilityClassifier(
            model_name=model_name,
            grade_levels=grade_levels,
            cache_size=cache_size,
        )
    else:
        return ReadabilityClassifier(
            model_name=model_name,
            grade_levels=grade_levels,
        )
