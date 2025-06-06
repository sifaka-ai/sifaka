"""Language classifier for detecting the language of text.

This module provides a classifier for detecting the language of text using
pretrained models from Hugging Face transformers.
"""

from typing import Dict, List, Any
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

# Available pretrained language detection models
LANGUAGE_MODELS: Dict[str, Dict[str, Any]] = {
    "papluca/xlm-roberta-base-language-detection": {
        "description": "XLM-RoBERTa fine-tuned for language detection",
        "languages": 20,
        "accuracy": 0.996,
        "size": "base",
    },
    "jb2k/bert-base-multilingual-cased-language-detection": {
        "description": "BERT multilingual cased for language detection",
        "languages": 100,
        "accuracy": 0.99,
        "size": "base",
    },
}

# Language code to name mapping (comprehensive)
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    # Additional languages for BERT multilingual model
    "af": "Afrikaans",
    "az": "Azerbaijani",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "ka": "Georgian",
    "kk": "Kazakh",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Myanmar",
    "nb": "Norwegian",
    "ne": "Nepali",
    "no": "Norwegian",
    "pa": "Punjabi",
    "ro": "Romanian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "tl": "Filipino",
    "uk": "Ukrainian",
    "uz": "Uzbek",
    "yo": "Yoruba",
}


class LanguageClassifier(BaseClassifier, TimingMixin):
    """Classifier for detecting the language of text.

    This classifier uses pretrained models from Hugging Face transformers
    for accurate language detection. It can detect many languages and
    provides confidence scores. Requires the transformers library to be installed.

    Attributes:
        model_name: Name of the pretrained model to use
        min_confidence: Minimum confidence threshold for detection
        fallback_lang: Language to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        pipeline: The Hugging Face pipeline for classification
        model_info: Information about the selected model
    """

    def __init__(
        self,
        model_name: str = "papluca/xlm-roberta-base-language-detection",
        min_confidence: float = 0.7,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.5,
        name: str = "language",
        description: str = "Detects the language of text using pretrained models",
    ):
        """Initialize the language classifier.

        Args:
            model_name: Name of the pretrained model to use
            min_confidence: Minimum confidence threshold for detection
            fallback_lang: Language to use when confidence is too low
            fallback_confidence: Confidence to assign to fallback language
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description)
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.fallback_lang = fallback_lang
        self.fallback_confidence = fallback_confidence
        self.pipeline = None
        self.model_info = LANGUAGE_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained language detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for language classification",
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
            f"Initialized language classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

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
                result = await self._classify_with_transformers(text)

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
                raise ValidationError(
                    f"Failed to classify text for language: {str(e)}",
                    error_code="classification_error",
                    context={
                        "classifier": self.name,
                        "text_length": len(text),
                        "error_type": type(e).__name__,
                    },
                    suggestions=[
                        "Check if transformers is properly installed",
                        "Verify input text is valid",
                        "Try with longer text for better accuracy",
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
        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        detected_lang = best_result["label"]
        best_confidence = float(best_result["score"])

        # Check if confidence meets threshold
        if best_confidence >= self.min_confidence:
            final_lang = detected_lang
            confidence = best_confidence
        else:
            final_lang = self.fallback_lang
            confidence = self.fallback_confidence

        return self.create_classification_result(
            label=final_lang,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "language_name": self.get_language_name(final_lang),
                "input_length": len(text),
                "min_confidence": self.min_confidence,
                "detected_language": detected_lang,
                "detected_confidence": best_confidence,
                "all_languages": [
                    {
                        "lang": result["label"],
                        "score": float(result["score"]),
                        "name": self.get_language_name(result["label"]),
                    }
                    for result in results
                ],
                "raw_results": results,
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
        model_name: str = "papluca/xlm-roberta-base-language-detection",
        min_confidence: float = 0.7,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.5,
        cache_size: int = 128,
        name: str = "cached_language",
        description: str = "Detects language with LRU caching",
    ):
        """Initialize the cached language classifier.

        Args:
            model_name: Name of the pretrained model to use
            min_confidence: Minimum confidence threshold for detection
            fallback_lang: Language to use when confidence is too low
            fallback_confidence: Confidence to assign to fallback language
            cache_size: Maximum number of results to cache
            name: Name of the classifier
            description: Description of the classifier
        """
        super().__init__(name=name, description=description, cache_size=cache_size)
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.fallback_lang = fallback_lang
        self.fallback_confidence = fallback_confidence
        self.pipeline = None
        self.model_info = LANGUAGE_MODELS.get(model_name, {"description": "Unknown model"})
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained language detection model."""
        try:
            import transformers
        except ImportError as e:
            raise ValidationError(
                "transformers is required for language classification",
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
            f"Initialized cached language classifier with transformers pipeline",
            extra={
                "classifier": self.name,
                "model_name": self.model_name,
                "method": "transformers_pipeline",
                "description": self.model_info.get("description", "Unknown model"),
            },
        )

    def _classify_uncached(self, text: str) -> ClassificationResult:
        """Perform language classification without caching."""
        try:
            return self._classify_with_transformers_sync(text)
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
            raise ValidationError(
                f"Failed to classify text for language: {str(e)}",
                error_code="classification_error",
                context={
                    "classifier": self.name,
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                },
                suggestions=[
                    "Check if transformers is properly installed",
                    "Verify input text is valid",
                    "Try with longer text for better accuracy",
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
        # Find the result with highest score
        best_result = max(results, key=lambda x: x["score"])
        detected_lang = best_result["label"]
        best_confidence = float(best_result["score"])

        # Check if confidence meets threshold
        if best_confidence >= self.min_confidence:
            final_lang = detected_lang
            confidence = best_confidence
        else:
            final_lang = self.fallback_lang
            confidence = self.fallback_confidence

        return self.create_classification_result(
            label=final_lang,
            confidence=confidence,
            metadata={
                "model_name": self.model_name,
                "language_name": self.get_language_name(final_lang),
                "input_length": len(text),
                "min_confidence": self.min_confidence,
                "detected_language": detected_lang,
                "detected_confidence": best_confidence,
                "cached": True,
                "all_languages": [
                    {
                        "lang": result["label"],
                        "score": float(result["score"]),
                        "name": self.get_language_name(result["label"]),
                    }
                    for result in results
                ],
                "raw_results": results,
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


# Factory function for easy creation
def create_language_classifier(
    model_name: str = "papluca/xlm-roberta-base-language-detection",
    min_confidence: float = 0.7,
    fallback_lang: str = "en",
    fallback_confidence: float = 0.5,
    cached: bool = False,
    cache_size: int = 128,
) -> BaseClassifier:
    """Create a language classifier with the specified parameters.

    Args:
        model_name: Name of the pretrained model to use
        min_confidence: Minimum confidence threshold for detection
        fallback_lang: Language to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        cached: Whether to use caching
        cache_size: Cache size if using cached version

    Returns:
        Configured language classifier
    """
    if cached:
        return CachedLanguageClassifier(
            model_name=model_name,
            min_confidence=min_confidence,
            fallback_lang=fallback_lang,
            fallback_confidence=fallback_confidence,
            cache_size=cache_size,
        )
    else:
        return LanguageClassifier(
            model_name=model_name,
            min_confidence=min_confidence,
            fallback_lang=fallback_lang,
            fallback_confidence=fallback_confidence,
        )
