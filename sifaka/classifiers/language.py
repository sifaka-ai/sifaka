"""
Language classifier using langdetect.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, ClassVar
import importlib
import logging

from sifaka.classifiers.base import Classifier, ClassificationResult
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Only import type hints during type checking
if TYPE_CHECKING:
    from langdetect import detect_langs
    from langdetect.language import Language


class LanguageClassifier(Classifier):
    """
    A lightweight language classifier using langdetect.

    This classifier detects the language of text using the langdetect library,
    which is a port of Google's language-detection library.

    Requires the 'language' extra to be installed:
    pip install sifaka[language]

    Attributes:
        min_confidence: Minimum confidence threshold
    """

    # ISO 639-1 language codes and their names
    LANGUAGE_NAMES: ClassVar[Dict[str, str]] = {
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
    }

    min_confidence: float = 0.1

    def __init__(
        self,
        name: str = "language_classifier",
        description: str = "Detects text language",
        min_confidence: float = 0.1,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the language classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            min_confidence: Minimum confidence threshold
            config: Additional configuration
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            labels=list(self.LANGUAGE_NAMES.keys()),
            cost=1,  # Low cost for statistical analysis
            **kwargs,
        )
        self.min_confidence = min_confidence
        self._detect_langs = None

    def _load_langdetect(self) -> None:
        """Load the language detector."""
        try:
            langdetect = importlib.import_module("langdetect")
            self._detect_langs = langdetect.detect_langs

            # Set seed for consistent results
            langdetect.DetectorFactory.seed = 0

        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load language detector: {e}")

    def warm_up(self) -> None:
        """Initialize the language detector if needed."""
        if self._detect_langs is None:
            self._load_langdetect()

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text language.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with language detection results
        """
        self.warm_up()
        try:
            # Get language probabilities
            langs = self._detect_langs(text)

            if not langs:
                return ClassificationResult(
                    label="en",  # Default to English
                    confidence=0.0,
                    metadata={"error": "No language detected"},
                )

            # Get most likely language
            top_lang = langs[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob

            # Get all detected languages with probabilities
            all_langs = {
                lang.lang: {
                    "probability": lang.prob,
                    "name": self.LANGUAGE_NAMES.get(lang.lang, "Unknown"),
                }
                for lang in langs
                if lang.prob >= self.min_confidence
            }

            return ClassificationResult(
                label=lang_code,
                confidence=confidence,
                metadata={
                    "language_name": self.LANGUAGE_NAMES.get(lang_code, "Unknown"),
                    "all_languages": all_langs,
                },
            )
        except Exception as e:
            logger.error("Failed to detect language: %s", e)
            return ClassificationResult(
                label="en",  # Default to English
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        return [self.classify(text) for text in texts]

    def get_language_name(self, lang_code: str) -> str:
        """
        Get the full name of a language from its code.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Full language name or 'Unknown'
        """
        return self.LANGUAGE_NAMES.get(lang_code, "Unknown")
