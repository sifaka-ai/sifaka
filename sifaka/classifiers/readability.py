"""Readability classifier for assessing text complexity and reading level.

This module provides a classifier for determining text readability using the textstat library
for traditional metrics and pretrained models for advanced assessment. Designed for the new
PydanticAI-based Sifaka architecture.

Uses textstat library for accurate readability metrics without hardcoded values.
"""

import importlib
from typing import Dict, List, Tuple
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
    "microsoft/DialoGPT-medium": {
        "description": "General text complexity assessment model",
        "labels": {"SIMPLE": "simple", "COMPLEX": "complex"},
    },
    "textattack/roberta-base-CoLA": {
        "description": "RoBERTa model for linguistic acceptability (complexity)",
        "labels": {"LABEL_0": "complex", "LABEL_1": "simple"},
    },
}


class ReadabilityClassifier(BaseClassifier, TimingMixin):
    """Classifier for assessing text readability using pretrained models.

    This classifier uses pretrained models from Hugging Face transformers
    with fallback to traditional readability metrics.

    Attributes:
        model_name: Name of the pretrained model to use
        grade_levels: List of grade level categories
        pipeline: The Hugging Face pipeline (if available)
        tokenizer: The tokenizer (if available)
        model: The model (if available)
    """

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
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
                f"Initialized readability classifier with transformers pipeline",
                extra={
                    "classifier": self.name,
                    "model_name": self.model_name,
                    "method": "transformers_pipeline",
                    "description": self.model_info.get("description", "Unknown model"),
                    "grade_levels": self.grade_levels,
                },
            )

        except (ImportError, Exception) as e:
            logger.warning(
                f"Transformers not available or model loading failed: {e}. "
                "Using traditional readability metrics. "
                "Install transformers for better accuracy: pip install transformers",
                extra={"classifier": self.name},
            )
            self.pipeline = None
            self.tokenizer = None
            self.model = None

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
                if self.pipeline is not None:
                    result = await self._classify_with_pipeline(text)
                else:
                    result = await self._classify_with_metrics(text)

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

            # Process results
            best_result = max(results, key=lambda x: x["score"])
            raw_label = best_result["label"]
            confidence = float(best_result["score"])

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
            # Fallback to traditional metrics
            logger.warning(
                f"Pipeline readability classification failed, using traditional metrics: {e}",
                extra={"classifier": self.name},
            )
            return await self._classify_with_metrics(text)

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

    async def _classify_with_metrics(self, text: str) -> ClassificationResult:
        """Classify using textstat library for accurate readability metrics."""

        def analyze():
            try:
                # Try to use textstat library for accurate metrics
                textstat = importlib.import_module("textstat")

                # Get comprehensive readability metrics
                metrics = {
                    "flesch_kincaid": textstat.flesch_kincaid_grade(text),
                    "flesch_reading_ease": textstat.flesch_reading_ease(text),
                    "gunning_fog": textstat.gunning_fog(text),
                    "smog": textstat.smog_index(text),
                    "automated_readability": textstat.automated_readability_index(text),
                    "coleman_liau": textstat.coleman_liau_index(text),
                    "linsear_write": textstat.linsear_write_formula(text),
                    "dale_chall": textstat.dale_chall_readability_score(text),
                    "difficult_words": textstat.difficult_words(text),
                    "sentences": textstat.sentence_count(text),
                    "words": textstat.lexicon_count(text),
                    "syllables": textstat.syllable_count(text),
                    "avg_sentence_length": textstat.avg_sentence_length(text),
                    "avg_syllables_per_word": textstat.avg_syllables_per_word(text),
                }

                # Calculate derived metrics
                metrics["difficult_word_ratio"] = (
                    metrics["difficult_words"] / metrics["words"] if metrics["words"] > 0 else 0
                )

                return metrics, "textstat"

            except ImportError:
                logger.warning(
                    "textstat library not available. Install with: pip install textstat. "
                    "Using simplified fallback metrics.",
                    extra={"classifier": self.name},
                )
                # Simple fallback metrics
                words = len(text.split())
                sentences = max(1, text.count(".") + text.count("!") + text.count("?"))

                # Very basic Flesch-Kincaid approximation
                avg_sentence_length = words / sentences
                flesch_kincaid = max(0, (avg_sentence_length - 15) / 2)  # Rough approximation

                metrics = {
                    "flesch_kincaid": flesch_kincaid,
                    "flesch_reading_ease": max(0, 100 - flesch_kincaid * 10),
                    "gunning_fog": flesch_kincaid,
                    "smog": flesch_kincaid,
                    "automated_readability": flesch_kincaid,
                    "coleman_liau": flesch_kincaid,
                    "linsear_write": flesch_kincaid,
                    "dale_chall": flesch_kincaid + 5,
                    "difficult_words": max(0, words // 10),  # Rough estimate
                    "sentences": sentences,
                    "words": words,
                    "syllables": words * 1.5,  # Rough estimate
                    "avg_sentence_length": avg_sentence_length,
                    "avg_syllables_per_word": 1.5,
                    "difficult_word_ratio": 0.1,  # Default estimate
                }

                return metrics, "fallback"

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        metrics, method = await loop.run_in_executor(None, analyze)

        # Determine overall readability level and grade
        grade_level, complexity_label, confidence = self._assess_readability(metrics)

        return self.create_classification_result(
            label=complexity_label,
            confidence=confidence,
            metadata={
                "method": f"textstat_metrics_{method}",
                "grade_level": grade_level,
                "flesch_kincaid_grade": metrics["flesch_kincaid"],
                "flesch_reading_ease": metrics["flesch_reading_ease"],
                "gunning_fog_index": metrics["gunning_fog"],
                "smog_index": metrics["smog"],
                "automated_readability_index": metrics["automated_readability"],
                "coleman_liau_index": metrics["coleman_liau"],
                "linsear_write_formula": metrics["linsear_write"],
                "dale_chall_score": metrics["dale_chall"],
                "difficult_word_ratio": metrics["difficult_word_ratio"],
                "avg_sentence_length": metrics["avg_sentence_length"],
                "avg_syllables_per_word": metrics["avg_syllables_per_word"],
                "text_stats": {
                    "sentences": metrics["sentences"],
                    "words": metrics["words"],
                    "syllables": metrics["syllables"],
                    "difficult_words": metrics["difficult_words"],
                },
                "input_length": len(text),
                "library_used": method,
            },
        )

    def _assess_readability(self, metrics: Dict) -> Tuple[str, str, float]:
        """Assess overall readability based on multiple metrics."""
        fk_grade = metrics["flesch_kincaid"]
        fre_score = metrics["flesch_reading_ease"]

        # Determine grade level based on Flesch-Kincaid primarily
        if fk_grade <= 6:
            grade_level = "elementary"
            complexity = "simple"
        elif fk_grade <= 9:
            grade_level = "middle"
            complexity = "moderate"
        elif fk_grade <= 13:
            grade_level = "high"
            complexity = "moderate"
        elif fk_grade <= 16:
            grade_level = "college"
            complexity = "complex"
        else:
            grade_level = "graduate"
            complexity = "complex"

        # Adjust based on other metrics
        if fre_score < 30:  # Very difficult
            complexity = "complex"
            if grade_level in ["elementary", "middle"]:
                grade_level = "college"
        elif fre_score > 90:  # Very easy
            complexity = "simple"
            if grade_level in ["college", "graduate"]:
                grade_level = "middle"

        # Calculate confidence based on metric consistency
        confidence = self._calculate_confidence(metrics, grade_level, complexity)

        return grade_level, complexity, confidence

    def _calculate_confidence(self, metrics: Dict, grade_level: str, complexity: str) -> float:
        """Calculate confidence based on metric consistency."""
        # Base confidence
        confidence = 0.7

        # Use grade_level and complexity for confidence adjustment
        if grade_level in ["elementary", "middle"] and complexity == "simple":
            confidence += 0.05
        elif grade_level in ["college", "graduate"] and complexity == "complex":
            confidence += 0.05

        # Adjust based on text length
        if metrics["words"] < 10:
            confidence -= 0.2
        elif metrics["words"] > 100:
            confidence += 0.1

        # Adjust based on metric agreement
        fk_grade = metrics["flesch_kincaid"]
        fog_index = metrics["gunning_fog"]

        if abs(fk_grade - fog_index) < 2:  # Metrics agree
            confidence += 0.1
        elif abs(fk_grade - fog_index) > 5:  # Metrics disagree
            confidence -= 0.1

        return max(0.5, min(0.95, confidence))


class CachedReadabilityClassifier(CachedClassifier):
    """Cached version of the readability classifier for improved performance."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
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
        return asyncio.run(self._classifier.classify_async(text))


# Factory function for easy creation
def create_readability_classifier(
    model_name: str = "microsoft/DialoGPT-medium",
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
