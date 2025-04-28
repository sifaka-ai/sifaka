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
    """
    Configuration for language detection.

    Note: This class is provided for backward compatibility.
    The preferred way to configure language detection is to use
    ClassifierConfig with params:

    ```python
    config = ClassifierConfig(
        labels=["en", "fr", "de", ...],  # language codes
        cost=1,
        params={
            "min_confidence": 0.1,
            "seed": 0,
            "fallback_lang": "en",
            "fallback_confidence": 0.0,
        }
    )
    ```
    """

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
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the language classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            lang_config: Language detection configuration (for backward compatibility)
            detector: Custom language detector implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Store detector for later use
        self._detector = detector
        self._initialized = False

        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Add language config to params if provided
            if lang_config is not None:
                params["min_confidence"] = lang_config.min_confidence
                params["seed"] = lang_config.seed
                params["fallback_lang"] = lang_config.fallback_lang
                params["fallback_confidence"] = lang_config.fallback_confidence

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=list(self.LANGUAGE_NAMES.keys()),
                cost=self.DEFAULT_COST,
                params=params,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

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
            seed = self.config.params.get("seed", 0)
            langdetect.DetectorFactory.seed = seed

            # Create a wrapper that implements the LanguageDetector protocol
            class LangDetectWrapper:
                def __init__(self, detect_langs, detect):
                    self.detect_langs_func = detect_langs
                    self.detect_func = detect

                def detect_langs(self, text: str) -> Sequence[Any]:
                    return self.detect_langs_func(text)

                def detect(self, text: str) -> str:
                    return self.detect_func(text)

            # Create wrapper with langdetect functions
            detector = LangDetectWrapper(langdetect.detect_langs, langdetect.detect)

            self._validate_detector(detector)
            return detector

        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load langdetect: {e}")

    def warm_up(self) -> None:
        """Initialize the language detector if needed."""
        if not self._initialized:
            self._detector = self._detector or self._load_langdetect()
            self._initialized = True

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from language code."""
        return self.LANGUAGE_NAMES.get(lang_code, lang_code)

    def _classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement language detection logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detected language
        """
        self.warm_up()

        # Get configuration from params
        min_confidence = self.config.params.get("min_confidence", 0.1)
        fallback_lang = self.config.params.get("fallback_lang", "en")
        fallback_confidence = self.config.params.get("fallback_confidence", 0.0)

        try:
            # Get language probabilities
            lang_probs = self._detector.detect_langs(text)

            # Find the most likely language
            best_lang = None
            best_prob = 0.0

            for lang_prob in lang_probs:
                lang_code = getattr(lang_prob, "lang", None)
                prob = float(getattr(lang_prob, "prob", 0.0))

                if lang_code and prob > best_prob:
                    best_lang = lang_code
                    best_prob = prob

            # If confidence is too low, use fallback language
            if best_lang is None or best_prob < min_confidence:
                return ClassificationResult(
                    label=fallback_lang,
                    confidence=fallback_confidence,
                    metadata={
                        "detected_lang": best_lang,
                        "detected_prob": best_prob,
                        "language_name": self.get_language_name(fallback_lang),
                        "reason": "low_confidence" if best_lang else "no_language_detected",
                    },
                )

            # Return the detected language
            return ClassificationResult(
                label=best_lang,
                confidence=best_prob,
                metadata={
                    "language_name": self.get_language_name(best_lang),
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
            logger.error("Failed to detect language: %s", e)
            return ClassificationResult(
                label=fallback_lang,
                confidence=fallback_confidence,
                metadata={
                    "error": str(e),
                    "language_name": self.get_language_name(fallback_lang),
                    "reason": "detection_error",
                },
            )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts using individual calls.

        Unfortunately, langdetect doesn't have a native batch interface,
        so we need to fall back to individual calls.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        self.validate_batch_input(texts)
        return [self.classify(text) for text in texts]

    @classmethod
    def create_with_custom_detector(
        cls,
        detector: LanguageDetector,
        name: str = "custom_language_classifier",
        description: str = "Custom language detector",
        lang_config: Optional[LanguageConfig] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "LanguageClassifier":
        """
        Factory method to create a classifier with a custom detector.

        Args:
            detector: Custom language detector implementation
            name: Name of the classifier
            description: Description of the classifier
            lang_config: Custom language configuration (for backward compatibility)
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters

        Returns:
            Configured LanguageClassifier instance
        """
        # Validate detector first
        if not isinstance(detector, LanguageDetector):
            raise ValueError(
                f"Detector must implement LanguageDetector protocol, got {type(detector)}"
            )

        # If lang_config is provided but config is not, create config from lang_config
        if lang_config is not None and config is None:
            # Extract params from lang_config
            params = {
                "min_confidence": lang_config.min_confidence,
                "seed": lang_config.seed,
                "fallback_lang": lang_config.fallback_lang,
                "fallback_confidence": lang_config.fallback_confidence,
            }

            # Create config with params
            config = ClassifierConfig(
                labels=list(cls.LANGUAGE_NAMES.keys()),
                cost=cls.DEFAULT_COST,
                params=params,
            )

        # Create instance with validated detector
        instance = cls(
            name=name,
            description=description,
            lang_config=lang_config,
            detector=detector,
            config=config,
            **kwargs,
        )

        return instance
