"""Language classifier for detecting the language of text.

This module provides a classifier for detecting the language of text using
the langdetect library with fallback to simple heuristics. Designed for
the new PydanticAI-based Sifaka architecture.
"""

import importlib
from typing import Any, Dict, List, Optional
import asyncio
import time

from sifaka.classifiers.base import (
    BaseClassifier,
    CachedClassifier,
    ClassificationResult,
    TimingMixin,
)
from sifaka.utils.errors import ValidationError
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Language code to name mapping (subset)
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
    "he": "Hebrew",
}

# Simple language detection patterns for fallback
LANGUAGE_PATTERNS = {
    "en": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
    "es": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"],
    "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
    "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
    "it": ["il", "di", "che", "e", "la", "per", "in", "un", "è", "con"],
}


class LanguageClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting the language of text.

    This classifier uses the langdetect library when available,
    with fallback to simple pattern-based detection. It can detect
    over 50 languages and provides confidence scores.

    Attributes:
        min_confidence: Minimum confidence threshold for detection
        fallback_lang: Language to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        seed: Random seed for consistent results
        detector: The language detection library instance
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.5,
        seed: Optional[int] = None,
        name: str = "language",
        description: str = "Detects the language of text",
    ):
        """Initialize the language classifier.

        Args:
            min_confidence: Minimum confidence threshold for detection
            fallback_lang: Language to use when confidence is too low
            fallback_confidence: Confidence to assign to fallback language
            seed: Random seed for consistent results
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.min_confidence = min_confidence
        self.fallback_lang = fallback_lang
        self.fallback_confidence = fallback_confidence
        self.seed = seed
        self.detector: Optional[Any] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the language detector."""
        try:
            # Try to use langdetect
            langdetect = importlib.import_module("langdetect")

            # Set seed for consistent results if provided
            if self.seed is not None:
                langdetect.DetectorFactory.seed = self.seed

            self.detector = langdetect
            logger.debug(
                "Initialized language classifier with langdetect", extra={"classifier": self.name}
            )

        except ImportError:
            logger.warning(
                "langdetect not available. Using pattern-based detection. "
                "Install langdetect for better accuracy: pip install langdetect",
                extra={"classifier": self.name},
            )
            self.detector = None

    async def classify_async(self, text: str) -> ClassificationResult:
        """Classify text for language asynchronously.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with language prediction
        """
        if not text or not text.strip():
            return self.create_classification_result(
                label=self.fallback_lang,
                confidence=self.fallback_confidence,
                metadata={
                    "reason": "empty_text",
                    "input_length": 0,
                    "language_name": self.get_language_name(self.fallback_lang),
                },
            )

        with self.time_operation("language_classification") as timer:
            try:
                if self.detector is not None:
                    result = await self._classify_with_library(text)
                else:
                    result = await self._classify_with_patterns(text)

                # Get processing time from timer context
                processing_time = getattr(timer, "duration_ms", 0.0)
                result.processing_time_ms = processing_time

                logger.debug(
                    f"Language classification completed",
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
                    f"Language classification failed",
                    extra={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )
                # Return fallback result instead of raising error
                return self.create_classification_result(
                    label=self.fallback_lang,
                    confidence=self.fallback_confidence,
                    metadata={
                        "reason": "detection_error",
                        "error": str(e),
                        "input_length": len(text),
                        "language_name": self.get_language_name(self.fallback_lang),
                    },
                )

    async def _classify_with_library(self, text: str) -> ClassificationResult:
        """Classify using langdetect library."""
        if self.detector is None:
            raise ValidationError(
                "Language detector is not available",
                error_code="dependency_missing",
                suggestions=["Install langdetect: pip install langdetect"],
            )

        try:
            # Run langdetect analysis in a thread to avoid blocking
            def analyze():
                lang_probs = self.detector.detect_langs(text)

                # Find best language and probability
                best_lang = None
                best_prob = 0.0

                for lang_prob in lang_probs:
                    lang_code = getattr(lang_prob, "lang", None)
                    prob = float(getattr(lang_prob, "prob", 0.0))

                    if lang_code and prob > best_prob:
                        best_lang = lang_code
                        best_prob = prob

                return best_lang, best_prob, lang_probs

            # Use asyncio to run in thread pool for CPU-bound work
            loop = asyncio.get_event_loop()
            best_lang, best_prob, lang_probs = await loop.run_in_executor(None, analyze)

            # Check if confidence meets threshold
            if best_lang and best_prob >= self.min_confidence:
                detected_lang = best_lang
                confidence = best_prob
            else:
                detected_lang = self.fallback_lang
                confidence = self.fallback_confidence

            return self.create_classification_result(
                label=detected_lang,
                confidence=confidence,
                metadata={
                    "method": "langdetect",
                    "language_name": self.get_language_name(detected_lang),
                    "input_length": len(text),
                    "min_confidence": self.min_confidence,
                    "all_langs": [
                        {
                            "lang": getattr(lp, "lang", None),
                            "prob": float(getattr(lp, "prob", 0.0)),
                            "name": self.get_language_name(getattr(lp, "lang", "")),
                        }
                        for lp in lang_probs
                    ],
                },
            )

        except Exception as e:
            # Fallback to pattern-based detection
            logger.warning(
                f"langdetect failed, using pattern detection: {e}", extra={"classifier": self.name}
            )
            return await self._classify_with_patterns(text)

    async def _classify_with_patterns(self, text: str) -> ClassificationResult:
        """Classify using simple pattern-based approach."""

        def analyze():
            text_lower = text.lower()
            words = text_lower.split()

            # Count pattern matches for each language
            lang_scores = {}
            for lang, patterns in LANGUAGE_PATTERNS.items():
                score = sum(1 for word in words if word in patterns)
                if score > 0:
                    lang_scores[lang] = score / len(words)

            return lang_scores, len(words)

        # Run analysis in thread pool for consistency
        loop = asyncio.get_event_loop()
        lang_scores, word_count = await loop.run_in_executor(None, analyze)

        # Find best match
        if lang_scores:
            best_lang = max(lang_scores, key=lambda x: lang_scores[x])
            confidence = min(lang_scores[best_lang] * 2, 0.8)  # Conservative confidence
        else:
            best_lang = self.fallback_lang
            confidence = self.fallback_confidence

        return self.create_classification_result(
            label=best_lang,
            confidence=confidence,
            metadata={
                "method": "pattern_based",
                "language_name": self.get_language_name(best_lang),
                "input_length": len(text),
                "word_count": word_count,
                "pattern_scores": lang_scores,
                "min_confidence": self.min_confidence,
            },
        )

    def get_language_name(self, lang_code: str) -> str:
        """Get the full name of a language from its code.

        Args:
            lang_code: The language code (e.g., 'en', 'es')

        Returns:
            The full language name
        """
        return LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return list(LANGUAGE_NAMES.keys())


class CachedLanguageClassifier(CachedClassifier, TimingMixin):
    """Cached version of LanguageClassifier with LRU caching for improved performance."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.5,
        seed: Optional[int] = None,
        cache_size: int = 128,
        name: str = "cached_language",
        description: str = "Detects language with LRU caching",
    ):
        """Initialize the cached language classifier.

        Args:
            min_confidence: Minimum confidence threshold for detection
            fallback_lang: Language to use when confidence is too low
            fallback_confidence: Confidence to assign to fallback language
            seed: Random seed for consistent results
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.min_confidence = min_confidence
        self.fallback_lang = fallback_lang
        self.fallback_confidence = fallback_confidence
        self.seed = seed
        self.detector: Optional[Any] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the language detector."""
        try:
            # Try to use langdetect
            langdetect = importlib.import_module("langdetect")

            # Set seed for consistent results if provided
            if self.seed is not None:
                langdetect.DetectorFactory.seed = self.seed

            self.detector = langdetect
            logger.debug(
                "Initialized cached language classifier with langdetect",
                extra={"classifier": self.name},
            )

        except ImportError:
            logger.warning(
                "langdetect not available. CachedLanguageClassifier will use pattern-based detection.",
                extra={"classifier": self.name},
            )
            self.detector = None

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform language classification without caching."""
        try:
            if self.detector is not None:
                return self._classify_with_library_sync(text)
            else:
                return self._classify_with_patterns_sync(text)
        except Exception as e:
            logger.error(
                f"Cached language classification failed",
                extra={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            # Return fallback result instead of raising error
            return self.create_classification_result(
                label=self.fallback_lang,
                confidence=self.fallback_confidence,
                metadata={
                    "reason": "detection_error",
                    "error": str(e),
                    "input_length": len(text),
                    "language_name": self.get_language_name(self.fallback_lang),
                    "cached": True,
                },
            )

    def _classify_with_library_sync(self, text: str) -> ClassificationResult:
        """Classify using langdetect library (synchronous)."""
        if self.detector is None:
            raise ValidationError(
                "Language detector is not available",
                error_code="dependency_missing",
                suggestions=["Install langdetect: pip install langdetect"],
            )

        try:
            lang_probs = self.detector.detect_langs(text)

            # Find best language and probability
            best_lang = None
            best_prob = 0.0

            for lang_prob in lang_probs:
                lang_code = getattr(lang_prob, "lang", None)
                prob = float(getattr(lang_prob, "prob", 0.0))

                if lang_code and prob > best_prob:
                    best_lang = lang_code
                    best_prob = prob

            # Check if confidence meets threshold
            if best_lang and best_prob >= self.min_confidence:
                detected_lang = best_lang
                confidence = best_prob
            else:
                detected_lang = self.fallback_lang
                confidence = self.fallback_confidence

            return self.create_classification_result(
                label=detected_lang,
                confidence=confidence,
                metadata={
                    "method": "langdetect",
                    "language_name": self.get_language_name(detected_lang),
                    "input_length": len(text),
                    "min_confidence": self.min_confidence,
                    "cached": True,
                    "all_langs": [
                        {
                            "lang": getattr(lp, "lang", None),
                            "prob": float(getattr(lp, "prob", 0.0)),
                            "name": self.get_language_name(getattr(lp, "lang", "")),
                        }
                        for lp in lang_probs
                    ],
                },
            )

        except Exception as e:
            # Fallback to pattern-based detection
            logger.warning(
                f"langdetect failed, using pattern detection: {e}", extra={"classifier": self.name}
            )
            return self._classify_with_patterns_sync(text)

    def _classify_with_patterns_sync(self, text: str) -> ClassificationResult:
        """Classify using simple pattern-based approach (synchronous)."""
        text_lower = text.lower()
        words = text_lower.split()

        # Count pattern matches for each language
        lang_scores = {}
        for lang, patterns in LANGUAGE_PATTERNS.items():
            score = sum(1 for word in words if word in patterns)
            if score > 0:
                lang_scores[lang] = score / len(words)

        # Find best match
        if lang_scores:
            best_lang = max(lang_scores, key=lambda x: lang_scores[x])
            confidence = min(lang_scores[best_lang] * 2, 0.8)  # Conservative confidence
        else:
            best_lang = self.fallback_lang
            confidence = self.fallback_confidence

        return self.create_classification_result(
            label=best_lang,
            confidence=confidence,
            metadata={
                "method": "pattern_based",
                "language_name": self.get_language_name(best_lang),
                "input_length": len(text),
                "word_count": len(words),
                "pattern_scores": lang_scores,
                "min_confidence": self.min_confidence,
                "cached": True,
            },
        )

    def get_language_name(self, lang_code: str) -> str:
        """Get the full name of a language from its code.

        Args:
            lang_code: The language code (e.g., 'en', 'es')

        Returns:
            The full language name
        """
        return LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    def get_classes(self) -> List[str]:
        """Get the list of possible class labels."""
        return list(LANGUAGE_NAMES.keys())


# Factory functions for easy creation
def create_language_classifier(
    min_confidence: float = 0.7,
    fallback_lang: str = "en",
    fallback_confidence: float = 0.5,
    seed: Optional[int] = None,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a language classifier with the specified parameters.

    Args:
        min_confidence: Minimum confidence threshold for detection
        fallback_lang: Language to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        seed: Random seed for consistent results
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured language classifier
    """
    if cached:
        return CachedLanguageClassifier(
            min_confidence=min_confidence,
            fallback_lang=fallback_lang,
            fallback_confidence=fallback_confidence,
            seed=seed,
            cache_size=cache_size,
        )
    else:
        return LanguageClassifier(
            min_confidence=min_confidence,
            fallback_lang=fallback_lang,
            fallback_confidence=fallback_confidence,
            seed=seed,
        )
