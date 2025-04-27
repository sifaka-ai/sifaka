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
    from langdetect import detect_langs, detect
    from langdetect.language import Language
    from langdetect import LangDetectException


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
        self._detect = None

    def _load_langdetect(self) -> None:
        """Load the language detector."""
        try:
            langdetect = importlib.import_module("langdetect")
            self._detect_langs = langdetect.detect_langs
            self._detect = langdetect.detect

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
        Classify the language of the input text.

        Args:
            text: The text to classify.

        Returns:
            ClassificationResult with detected language and confidence.
        """
        metadata = {
            "language_name": "Unknown",  # Default language name
            "language_code": "en",  # Default language code
            "all_languages": {
                "en": {"probability": 1.0, "name": "English"}
            },  # Default language info
        }

        # Handle invalid input types
        if not isinstance(text, str):
            metadata.update(
                {
                    "error_type": "invalid_input",
                    "error": f"Invalid input type: {type(text).__name__}",
                }
            )
            return ClassificationResult(
                label="en",
                confidence=0.0,
                metadata=metadata,
            )

        # Handle empty or whitespace-only strings
        if not text or text.isspace():
            metadata.update(
                {
                    "error_type": "empty_input",
                    "error": "No language detected - empty or whitespace input",
                }
            )
            return ClassificationResult(
                label="en",
                confidence=0.0,
                metadata=metadata,
            )

        try:
            # Ensure langdetect is loaded
            if self._detect is None:
                self.warm_up()

            # Detect language
            lang = self._detect(text)
            confidence = 1.0  # langdetect doesn't provide confidence scores

            # Map language code to full name if available
            language_name = self.LANGUAGE_NAMES.get(lang, lang)
            metadata.update(
                {
                    "language_code": lang,
                    "language_name": language_name,
                    "all_languages": {lang: {"probability": 1.0, "name": language_name}},
                }
            )

            return ClassificationResult(label=lang, confidence=confidence, metadata=metadata)

        except Exception as e:
            logger.error("Failed to detect language: %s", e)
            metadata.update(
                {
                    "error_type": "detection_error",
                    "error": f"Failed to detect language: {str(e)}",
                }
            )
            return ClassificationResult(
                label="en",
                confidence=0.0,
                metadata=metadata,
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
