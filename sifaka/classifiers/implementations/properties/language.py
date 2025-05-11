"""
Language detection classifiers for Sifaka.

This module provides classifiers for detecting the language of text content.

## Architecture

LanguageClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl() implements language detection
4. **State Management**: Uses StateManager for internal state
"""

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from typing_extensions import TypeGuard
from pydantic import PrivateAttr

from sifaka.classifiers.classifier import Classifier
from sifaka.classifiers.result import ClassificationResult
from sifaka.classifiers.config import ClassifierConfig, standardize_classifier_config
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_classifier_state

logger = get_logger(__name__)


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for language detection engines."""

    @abstractmethod
    def detect_langs(self, text: str) -> Sequence[Any]: ...
    @abstractmethod
    def detect(self, text: str) -> str: ...


@dataclass(frozen=True)
class LanguageInfo:
    """Information about a supported language."""

    code: str
    name: str
    native_name: Optional[str] = None
    script: Optional[str] = None


class LanguageClassifier(Classifier):
    """
    A lightweight language classifier using langdetect.

    This classifier detects the language of text using the langdetect library,
    which is a port of Google's language-detection library.

    Requires the 'language' extra to be installed:
    pip install sifaka[language]
    """

    # Class-level constants
    LANGUAGE_NAMES: Dict[str, str] = {
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

    DEFAULT_COST: int = 1  # Low cost for statistical analysis

    def __init__(
        self,
        name: str = "language_classifier",
        description: str = "Detects text language",
        detector: Optional[LanguageDetector] = None,
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the language classifier.

        Args:
            name: The name of the classifier
            description: Description of the classifier
            detector: Custom language detector implementation
            config: Optional classifier configuration
            **kwargs: Additional configuration parameters
        """
        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

            # Create config with remaining kwargs
            config = ClassifierConfig(
                labels=list(self.LANGUAGE_NAMES.keys()),
                cost=self.DEFAULT_COST,
                params=params,
                **kwargs,
            )

        # Initialize base class
        super().__init__(name=name, description=description, config=config)

        # Initialize state - handled by BaseClassifier in model_post_init

        # Store detector in state if provided
        if detector is not None and self._validate_detector(detector):
            cache = self._state_manager.get("cache", {})
            cache["detector"] = detector
            self._state_manager.update("cache", cache)

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
            # Check if detector is already in state
            if self._state_manager.get("cache", {}).get("detector"):
                return self._state_manager.get("cache")["detector"]

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

            # Validate and store in state
            if self._validate_detector(detector):
                cache = self._state_manager.get("cache", {})
                cache["detector"] = detector
                self._state_manager.update("cache", cache)
                return detector

        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load langdetect: {e}")

    def initialize(self) -> None:
        """Initialize the language detector."""
        # Load detector
        detector = self._load_langdetect()

        # Store in state
        cache = self._state_manager.get("cache", {})
        cache["detector"] = detector
        self._state_manager.update("cache", cache)

    def warm_up(self) -> None:
        """Initialize the language detector if needed."""
        # Use the base class warm_up method which calls initialize()
        super().warm_up()

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from language code."""
        return self.LANGUAGE_NAMES.get(lang_code, lang_code)

    def _classify_impl_uncached(self, text: str) -> ClassificationResult:
        """
        Implement language detection logic.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detected language
        """
        # Ensure resources are initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Get configuration from params
        min_confidence = self.config.params.get("min_confidence", 0.1)
        fallback_lang = self.config.params.get("fallback_lang", "en")
        fallback_confidence = self.config.params.get("fallback_confidence", 0.0)

        try:
            # Get detector from state
            detector = self._state_manager.get("cache", {}).get("detector")
            if not detector:
                raise RuntimeError("Language detector not initialized")

            # Get language probabilities
            lang_probs = detector.detect_langs(text)

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
            result = ClassificationResult(
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

            # Track statistics
            stats = self._state_manager.get("statistics", {})
            stats[best_lang] = stats.get(best_lang, 0) + 1
            self._state_manager.update("statistics", stats)

            return result

        except Exception as e:
            logger.error("Failed to detect language: %s", e)

            # Track errors in state
            error_info = {"error": str(e), "type": type(e).__name__}
            errors = self._state_manager.get("errors", [])
            errors.append(error_info)
            self._state_manager.update("errors", errors)

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

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics.

        This method provides access to statistics collected during classifier operation,
        including classification counts by label, error counts, and cache information.

        Returns:
            Dictionary containing statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()

        # Add language-specific statistics
        stats.update(
            {
                # Configuration
                "seed": self.config.params.get("seed", 0),
            }
        )

        # Add language counts
        detected_languages = set(self._state_manager.get("statistics", {}).keys())
        stats["detected_languages"] = [
            {"code": lang, "name": self.get_language_name(lang)} for lang in detected_languages
        ]

        return stats

    def clear_cache(self) -> None:
        """
        Clear any cached data in the classifier.

        This method clears both the result cache and resets statistics in the state
        but preserves the detector and initialization status.
        """
        # Call parent clear_cache to reset basic statistics
        super().clear_cache()

        # Keep the detector in cache
        cache = self._state_manager.get("cache", {})
        preserved_cache = {k: v for k, v in cache.items() if k == "detector"}
        self._state_manager.update("cache", preserved_cache)

    @classmethod
    def create_with_custom_detector(
        cls,
        detector: LanguageDetector,
        name: str = "custom_language_classifier",
        description: str = "Custom language detector",
        config: Optional[ClassifierConfig] = None,
        **kwargs,
    ) -> "LanguageClassifier":
        """
        Factory method to create a classifier with a custom detector.

        Args:
            detector: Custom language detector implementation
            name: Name of the classifier
            description: Description of the classifier
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

        # Create config if not provided
        if config is None:
            # Extract params from kwargs if present
            params = kwargs.pop("params", {})

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
            detector=detector,
            config=config,
            **kwargs,
        )

        # Initialize state
        instance._state_manager.update("cache", {"detector": detector})
        instance._state_manager.update("initialized", True)

        return instance


def create_language_classifier(
    name: str = "language_classifier",
    description: str = "Detects text language",
    min_confidence: float = 0.1,
    fallback_lang: str = "en",
    fallback_confidence: float = 0.0,
    seed: int = 0,
    cache_size: int = 100,
    cost: float = 1,  # Default cost for language classifier
    config: Optional[Union[Dict[str, Any], ClassifierConfig]] = None,
    **kwargs: Any,
) -> LanguageClassifier:
    """
    Create a language classifier.

    This factory function creates a LanguageClassifier with the specified
    configuration options.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        min_confidence: Minimum confidence threshold for language detection
        fallback_lang: Language code to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        seed: Random seed for consistent results
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        A LanguageClassifier instance

    Examples:
        ```python
        from sifaka.classifiers.implementations.properties.language import create_language_classifier

        # Create a language classifier with default settings
        classifier = create_language_classifier()

        # Create a language classifier with custom settings
        classifier = create_language_classifier(
            name="custom_language_classifier",
            description="Custom language detector with specific settings",
            min_confidence=0.2,
            fallback_lang="fr",
            cache_size=200
        )

        # Classify text
        result = classifier.classify("Hello, world!")
        print(f"Language: {result.label}, Name: {result.metadata['language_name']}")
        print(f"Confidence: {result.confidence:.2f}")
        ```
    """
    # Use standardize_classifier_config to handle different config formats
    classifier_config = standardize_classifier_config(
        config=config,
        labels=list(LanguageClassifier.LANGUAGE_NAMES.keys()),
        cost=cost,
        cache_size=cache_size,
        params={
            "min_confidence": min_confidence,
            "fallback_lang": fallback_lang,
            "fallback_confidence": fallback_confidence,
            "seed": seed,
        },
        **kwargs,
    )

    return LanguageClassifier(
        name=name,
        description=description,
        config=classifier_config,
    )
