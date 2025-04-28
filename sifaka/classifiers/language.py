"""
Language classifier using langdetect.
"""

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    BaseClassifier,
    ClassificationResult,
    ClassifierConfig,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for language detection engines."""

    @abstractmethod
    def detect_langs(self, text: str) -> Sequence[Any]: ...
    @abstractmethod
    def detect(self, text: str) -> str: ...

@dataclass(frozen=True)
class LanguageConfig:
    """Configuration for language detection."""

    min_confidence: float = 0.1
    seed: int = 0  # Seed for consistent results
    fallback_lang: str = "en"
    fallback_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

@dataclass(frozen=True)
class LanguageInfo:
    """Information about a supported language."""

    code: str
    name: str
    native_name: Optional[str] = None
    script: Optional[str] = None

class LanguageClassifier(BaseClassifier):
    """
    A lightweight language classifier using langdetect.

    This classifier detects the language of text using the langdetect library,
    which is a port of Google's language-detection library.

    Requires the 'language' extra to be installed:
    pip install sifaka[language]
    """

    # Class-level constants
    LANGUAGE_NAMES: Final[Dict[str, str]] = {
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

    DEFAULT_COST: Final[int] = 1  # Low cost for statistical analysis

    def __init__(
        self,
        name: str = "language_classifier",
        description: str = "Detects text language",
        lang_config: Optional[LanguageConfig] = None,
        detector: Optional[LanguageDetector] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the language classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            lang_config: Language detection configuration
            detector: Custom language detector implementation
            **kwargs: Additional configuration parameters
        """
        config = ClassifierConfig(
            labels=list(self.LANGUAGE_NAMES.keys()), cost=self.DEFAULT_COST, **kwargs
        )
        super().__init__(name=name, description=description, config=config)

        self._lang_config = lang_config or LanguageConfig()
        self._detector = detector
        self._initialized = False

    def _validate_detector(self, detector: Any) -> TypeGuard[LanguageDetector]:
        """Validate that a detector implements the required protocol."""
        if not isinstance(detector, LanguageDetector):
            raise ValueError(
                f"Detector must implement LanguageDetector protocol, got {type(detector)}"
            )
        return True

    def _load_langdetect(self) -> LanguageDetector:
        """Load the language detector."""
        try:
            langdetect = importlib.import_module("langdetect")
            # Set seed for consistent results
            langdetect.DetectorFactory.seed = self._lang_config.seed

            # Create a wrapper class that implements our protocol
            class LangDetectWrapper:
                def __init__(self, detect_langs, detect):
                    self.detect_langs = detect_langs
                    self.detect = detect

            detector = LangDetectWrapper(langdetect.detect_langs, langdetect.detect)
            self._validate_detector(detector)
            return detector
        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load language detector: {e}")

    def warm_up(self) -> None:
        """Initialize the language detector if needed."""
        if not self._initialized:
            self._detector = self._detector or self._load_langdetect()
            self._initialized = True

    def get_language_name(self, lang_code: str) -> str:
        """Get the full name of a language from its code."""
        return self.LANGUAGE_NAMES.get(lang_code, "Unknown")

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement language classification logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detected language and confidence
        """
        self.warm_up()

        # Initialize default metadata
        metadata = {
            "language_name": self.get_language_name(self._lang_config.fallback_lang),
            "language_code": self._lang_config.fallback_lang,
            "all_languages": {
                self._lang_config.fallback_lang: {
                    "probability": self._lang_config.fallback_confidence,
                    "name": self.get_language_name(self._lang_config.fallback_lang),
                }
            },
        }

        try:
            # Get language probabilities
            lang_probabilities = self._detector.detect_langs(text)

            # Convert to our format and sort by probability
            detected_langs = []
            for lang in lang_probabilities:
                code = str(lang).split(":")[0]
                prob = float(str(lang).split(":")[1])
                detected_langs.append((code, prob))

            detected_langs.sort(key=lambda x: x[1], reverse=True)

            if not detected_langs:
                return ClassificationResult(
                    label=self._lang_config.fallback_lang,
                    confidence=self._lang_config.fallback_confidence,
                    metadata={**metadata, "reason": "no_languages_detected"},
                )

            # Get the most likely language
            top_lang_code, confidence = detected_langs[0]

            # Update metadata with all detected languages
            metadata.update(
                {
                    "language_code": top_lang_code,
                    "language_name": self.get_language_name(top_lang_code),
                    "all_languages": {
                        code: {"probability": prob, "name": self.get_language_name(code)}
                        for code, prob in detected_langs
                    },
                }
            )

            return ClassificationResult(
                label=top_lang_code, confidence=confidence, metadata=metadata
            )

        except Exception as e:
            logger.error("Failed to detect language: %s", e)
            return ClassificationResult(
                label=self._lang_config.fallback_lang,
                confidence=self._lang_config.fallback_confidence,
                metadata={**metadata, "error": str(e), "reason": "detection_error"},
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)
        return [self._classify_impl(text) for text in texts]

    @classmethod
    def create_with_custom_detector(
        cls,
        detector: LanguageDetector,
        name: str = "custom_language_classifier",
        description: str = "Custom language detector",
        lang_config: Optional[LanguageConfig] = None,
        **kwargs,
    ) -> "LanguageClassifier":
        """
        Factory method to create a classifier with a custom detector.

        Args:
            detector: Custom language detector implementation
            name: Name of the classifier
            description: Description of the classifier
            lang_config: Custom language configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured LanguageClassifier instance
        """
        instance = cls(
            name=name, description=description, lang_config=lang_config, detector=detector, **kwargs
        )
        instance._validate_detector(detector)
        return instance
