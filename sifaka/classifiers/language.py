"""
Language classifier for Sifaka.

This module provides a classifier that detects the language of text using the
langdetect library, which is a port of Google's language-detection library.
It supports over 50 languages and provides detailed confidence scores.
"""

import importlib
from typing import List, Any, Optional, Sequence, Protocol, runtime_checkable

from sifaka.classifiers import ClassificationResult


@runtime_checkable
class LanguageDetector(Protocol):
    """
    Protocol for language detection engines.

    This protocol defines the interface that any language detector must implement
    to be compatible with the LanguageClassifier. It requires methods for detecting
    the most likely language and for retrieving a list of possible languages with
    confidence scores.
    """

    def detect_langs(self, text: str) -> Sequence[Any]:
        """
        Detect possible languages in text with confidence scores.

        Args:
            text: The text to detect languages in

        Returns:
            A sequence of language probability objects with lang and prob attributes
        """
        ...

    def detect(self, text: str) -> str:
        """
        Detect the most likely language in text.

        Args:
            text: The text to detect the language of

        Returns:
            The language code of the most likely language
        """
        ...


class LanguageClassifier:
    """
    A language classifier using langdetect.

    This classifier detects the language of text using the langdetect library,
    which is a port of Google's language-detection library. It supports over 50
    languages and provides detailed confidence scores and language metadata.

    Attributes:
        min_confidence: Minimum confidence threshold for language detection.
        fallback_lang: Fallback language code to use when confidence is below threshold.
        fallback_confidence: Confidence to assign to fallback language.
        seed: Random seed for consistent results.
        name: The name of the classifier.
        description: The description of the classifier.
    """

    # Language names mapping
    LANGUAGE_NAMES = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "ca": "Catalan",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "et": "Estonian",
        "fa": "Persian",
        "fi": "Finnish",
        "fr": "French",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "kn": "Kannada",
        "ko": "Korean",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mr": "Marathi",
        "ne": "Nepali",
        "nl": "Dutch",
        "no": "Norwegian",
        "pa": "Punjabi",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "unknown": "Unknown",
    }

    def __init__(
        self,
        min_confidence: float = 0.1,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.0,
        seed: int = 0,
        detector: Optional[LanguageDetector] = None,
        name: str = "language_classifier",
        description: str = "Detects the language of text",
    ):
        """
        Initialize the language classifier.

        Args:
            min_confidence: Minimum confidence threshold for language detection.
            fallback_lang: Fallback language code to use when confidence is below threshold.
            fallback_confidence: Confidence to assign to fallback language.
            seed: Random seed for consistent results.
            detector: Optional custom language detector implementation.
            name: The name of the classifier.
            description: The description of the classifier.
        """
        self._name = name
        self._description = description
        self._min_confidence = min_confidence
        self._fallback_lang = fallback_lang
        self._fallback_confidence = fallback_confidence
        self._seed = seed
        self._detector = detector
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the classifier name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the classifier description."""
        return self._description

    def _load_langdetect(self) -> LanguageDetector:
        """
        Load the langdetect library and create a detector.

        Returns:
            A language detector instance.

        Raises:
            ImportError: If langdetect is not installed.
            RuntimeError: If detector initialization fails.
        """
        try:
            langdetect = importlib.import_module("langdetect")
            langdetect.DetectorFactory.seed = self._seed

            class LangDetectWrapper:
                def __init__(self, detect_langs_func: Any, detect_func: Any) -> None:
                    self.detect_langs_func = detect_langs_func
                    self.detect_func = detect_func

                def detect_langs(self, text: str) -> Sequence[Any]:
                    result = self.detect_langs_func(text)
                    if not isinstance(result, Sequence):
                        raise TypeError(f"Expected Sequence, got {type(result)}")
                    return result

                def detect(self, text: str) -> str:
                    result = self.detect_func(text)
                    if not isinstance(result, str):
                        raise TypeError(f"Expected str, got {type(result)}")
                    return result

            detector = LangDetectWrapper(langdetect.detect_langs, langdetect.detect)
            return detector
        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install langdetect"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load langdetect: {e}")

    def _initialize(self) -> None:
        """Initialize the language detector if needed."""
        if not self._initialized:
            if self._detector is None:
                self._detector = self._load_langdetect()
            self._initialized = True

    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from language code.

        Args:
            lang_code: The language code.

        Returns:
            The full language name.
        """
        return self.LANGUAGE_NAMES.get(lang_code, lang_code)

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the language of text.

        Args:
            text: The text to classify.

        Returns:
            A ClassificationResult with the language label and confidence score.
        """
        # Initialize detector if needed
        self._initialize()

        # Handle empty text
        if not text or not text.strip():
            return ClassificationResult(
                label=self._fallback_lang,
                confidence=self._fallback_confidence,
                metadata={
                    "input_length": 0,
                    "language_name": self.get_language_name(self._fallback_lang),
                    "reason": "empty_text",
                },
            )

        try:
            # Detect language
            lang_probs = self._detector.detect_langs(text)

            # Find best language and probability
            best_lang = None
            best_prob = 0.0

            for lang_prob in lang_probs:
                lang_code = getattr(lang_prob, "lang", None)
                prob = float(getattr(lang_prob, "prob", 0.0))

                if lang_code and prob > best_prob:
                    best_lang = lang_code
                    best_prob = prob

            # Check if confidence is too low
            if best_lang is None or best_prob < self._min_confidence:
                return ClassificationResult(
                    label=self._fallback_lang,
                    confidence=self._fallback_confidence,
                    metadata={
                        "detected_lang": best_lang,
                        "detected_prob": best_prob,
                        "language_name": self.get_language_name(self._fallback_lang),
                        "reason": "low_confidence" if best_lang else "no_language_detected",
                        "input_length": len(text),
                    },
                )

            # Return result with best language
            return ClassificationResult(
                label=best_lang,
                confidence=best_prob,
                metadata={
                    "language_name": self.get_language_name(best_lang),
                    "input_length": len(text),
                    "all_langs": [
                        {
                            "lang": getattr(lang_prob, "lang", None),
                            "prob": float(getattr(lang_prob, "prob", 0.0)),
                            "name": self.get_language_name(getattr(lang_prob, "lang", "")),
                        }
                        for lang_prob in lang_probs
                    ],
                },
            )

        except Exception as e:
            # Handle errors
            return ClassificationResult(
                label=self._fallback_lang,
                confidence=self._fallback_confidence,
                metadata={
                    "error": str(e),
                    "language_name": self.get_language_name(self._fallback_lang),
                    "reason": "detection_error",
                    "input_length": len(text),
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: The list of texts to classify.

        Returns:
            A list of ClassificationResults.
        """
        return [self.classify(text) for text in texts]

    @classmethod
    def create_with_custom_detector(
        cls,
        detector: LanguageDetector,
        name: str = "custom_language_classifier",
        description: str = "Custom language detector",
        min_confidence: float = 0.1,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.0,
    ) -> "LanguageClassifier":
        """
        Create a classifier with a custom detector.

        Args:
            detector: Custom language detector implementation.
            name: Name of the classifier.
            description: Description of the classifier.
            min_confidence: Minimum confidence threshold for language detection.
            fallback_lang: Fallback language code to use when confidence is below threshold.
            fallback_confidence: Confidence to assign to fallback language.

        Returns:
            A LanguageClassifier instance.
        """
        return cls(
            min_confidence=min_confidence,
            fallback_lang=fallback_lang,
            fallback_confidence=fallback_confidence,
            detector=detector,
            name=name,
            description=description,
        )
