"""
Language detection classifiers for Sifaka.

This module provides classifiers for detecting the language of text content.

## Architecture

LanguageClassifier follows the composition over inheritance pattern:
1. **Classifier**: Provides the public API and handles caching
2. **Implementation**: Contains the core classification logic
3. **Factory Function**: Creates a classifier with the language implementation

## Lifecycle

1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract parameters from config.params
   - Set up default values

2. **Warm-up**: Load language detection resources
   - Load langdetect library when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Apply language detection
   - Convert probabilities to standardized format
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map language codes to labels
   - Convert probabilities to confidence values
   - Include detailed language information in metadata
"""

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from typing_extensions import TypeGuard

from sifaka.classifiers.base import (
    ClassificationResult,
    ClassifierConfig,
    Classifier,
    ClassifierImplementation,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.state import ClassifierState

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


class LanguageClassifierImplementation:
    """
    Implementation of language classification logic using langdetect.

    This implementation uses the langdetect library to detect the language of text.
    It provides a fast, local alternative to API-based language detection and
    can identify a wide range of languages with reasonable accuracy.

    ## Architecture

    LanguageClassifierImplementation follows the composition pattern:
    1. **Core Logic**: classify_impl() implements language detection
    2. **State Management**: Uses ClassifierState for internal state
    3. **Resource Management**: Loads and manages langdetect library

    ## Lifecycle

    1. **Initialization**: Set up configuration and parameters
       - Initialize with config
       - Extract parameters from config.params
       - Set up default values

    2. **Warm-up**: Load language detection resources
       - Load langdetect library when needed
       - Initialize only once
       - Handle initialization errors gracefully

    3. **Classification**: Process input text
       - Apply language detection
       - Convert probabilities to standardized format
       - Handle empty text and edge cases

    4. **Result Creation**: Return standardized results
       - Map language codes to labels
       - Convert probabilities to confidence values
       - Include detailed language information in metadata
    """

    # Class-level constants
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
    DEFAULT_COST: ClassVar[int] = 1  # Low cost for statistical analysis

    def __init__(self, config: ClassifierConfig) -> None:
        """
        Initialize the language classifier implementation.

        Args:
            config: Configuration for the classifier
        """
        self.config = config
        self._state = ClassifierState()
        self._state.initialized = False
        self._state.cache = {}

    def _validate_detector(self, detector: Any) -> TypeGuard[LanguageDetector]:
        """
        Validate that a detector implements the required protocol.

        Args:
            detector: The detector to validate

        Returns:
            True if the detector is valid

        Raises:
            ValueError: If the detector doesn't implement the LanguageDetector protocol
        """
        if not isinstance(detector, LanguageDetector):
            raise ValueError(
                f"Detector must implement LanguageDetector protocol, got {type(detector)}"
            )
        return True

    def _load_langdetect(self) -> LanguageDetector:
        """
        Load the language detector.

        Returns:
            Initialized language detector

        Raises:
            ImportError: If the langdetect package is not installed
            RuntimeError: If langdetect initialization fails
        """
        try:
            # Check if detector is already in state
            if "detector" in self._state.cache:
                return self._state.cache["detector"]

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
                self._state.cache["detector"] = detector
                return detector

        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. "
                "Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load langdetect: {e}")

    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from language code.

        Args:
            lang_code: The language code to look up

        Returns:
            The full language name or the original code if not found
        """
        return self.LANGUAGE_NAMES.get(lang_code, lang_code)

    def warm_up_impl(self) -> None:
        """
        Initialize the language detector if needed.

        This method loads the langdetect library and initializes the detector.
        It is called automatically when needed but can also be called
        explicitly to pre-initialize resources.

        Raises:
            RuntimeError: If detector initialization fails
        """
        if not self._state.initialized:
            try:
                # Load detector
                detector = self._load_langdetect()

                # Store in state
                self._state.cache["detector"] = detector

                # Mark as initialized
                self._state.initialized = True
            except Exception as e:
                logger.error("Failed to initialize language detector: %s", e)
                self._state.error = f"Failed to initialize language detector: {e}"
                raise RuntimeError(f"Failed to initialize language detector: {e}") from e

    def classify_impl(self, text: str) -> ClassificationResult:
        """
        Implement language detection logic.

        This method contains the core language detection logic using the langdetect library.

        Args:
            text: The text to classify

        Returns:
            ClassificationResult with detected language
        """
        # Ensure resources are initialized
        if not self._state.initialized:
            self.warm_up_impl()

        # Handle empty or whitespace-only text
        if not text.strip():
            return ClassificationResult(
                label=None,
                confidence=0.0,
                metadata={
                    "reason": "empty_input",
                },
            )

        # Get configuration from params
        min_confidence = self.config.params.get("min_confidence", 0.1)
        fallback_lang = self.config.params.get("fallback_lang", "en")
        fallback_confidence = self.config.params.get("fallback_confidence", 0.0)

        try:
            # Get detector from state
            detector = self._state.cache.get("detector")
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

    def batch_classify_impl(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Implement batch language detection logic.

        Unfortunately, langdetect doesn't have a native batch interface,
        so we need to fall back to individual calls.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResults
        """
        # Ensure resources are initialized
        if not self._state.initialized:
            self.warm_up_impl()

        # Process each text individually
        return [self.classify_impl(text) for text in texts]


def create_language_classifier_with_custom_detector(
    detector: LanguageDetector,
    name: str = "custom_language_classifier",
    description: str = "Custom language detector",
    min_confidence: float = 0.1,
    fallback_lang: str = "en",
    fallback_confidence: float = 0.0,
    cache_size: int = 100,
    cost: float = 1,
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Factory function to create a classifier with a custom detector.

    This function provides a simpler interface for creating a language classifier
    with a custom detector implementation, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    Args:
        detector: Custom language detector implementation
        name: Name of the classifier
        description: Description of the classifier
        min_confidence: Minimum confidence threshold for language detection
        fallback_lang: Language code to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        cache_size: Size of the classification cache (0 to disable)
        cost: Computational cost of this classifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured Classifier instance with LanguageClassifierImplementation

    Raises:
        ValueError: If the detector doesn't implement the LanguageDetector protocol

    Examples:
        ```python
        from sifaka.classifiers.language import create_language_classifier_with_custom_detector

        # Create a custom detector that implements the LanguageDetector protocol
        class MyDetector:
            def detect_langs(self, text: str) -> Sequence[Any]:
                # Custom implementation
                return [...]

            def detect(self, text: str) -> str:
                # Custom implementation
                return "en"

        # Create a classifier with the custom detector
        classifier = create_language_classifier_with_custom_detector(
            detector=MyDetector(),
            name="my_language_classifier",
            description="My custom language detector",
            min_confidence=0.2,
            fallback_lang="en"
        )
        ```
    """
    # Validate detector first
    if not isinstance(detector, LanguageDetector):
        raise ValueError(f"Detector must implement LanguageDetector protocol, got {type(detector)}")

    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "min_confidence": min_confidence,
            "fallback_lang": fallback_lang,
            "fallback_confidence": fallback_confidence,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=list(LanguageClassifierImplementation.LANGUAGE_NAMES.keys()),
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = LanguageClassifierImplementation(config)

    # Set the detector directly in the implementation's state
    implementation._state.cache["detector"] = detector
    implementation._state.initialized = True

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )


def create_language_classifier(
    name: str = "language_classifier",
    description: str = "Detects text language",
    min_confidence: float = 0.1,
    fallback_lang: str = "en",
    fallback_confidence: float = 0.0,
    seed: int = 0,
    cache_size: int = 100,
    cost: float = 1,  # Default cost for language classifier
    **kwargs: Any,
) -> Classifier[str, str]:
    """
    Create a language classifier.

    This factory function creates a language classifier with the specified
    configuration options. It follows the composition over inheritance pattern,
    creating a Classifier with a LanguageClassifierImplementation.

    Args:
        name: Name of the classifier
        description: Description of the classifier
        min_confidence: Minimum confidence threshold for language detection
        fallback_lang: Language code to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        seed: Random seed for consistent results
        cache_size: Size of the cache for memoization
        cost: Cost of running the classifier
        **kwargs: Additional configuration parameters

    Returns:
        A Classifier instance with LanguageClassifierImplementation

    Examples:
        ```python
        from sifaka.classifiers.language import create_language_classifier

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
    # Prepare params
    params = kwargs.pop("params", {})
    params.update(
        {
            "min_confidence": min_confidence,
            "fallback_lang": fallback_lang,
            "fallback_confidence": fallback_confidence,
            "seed": seed,
        }
    )

    # Create config
    config = ClassifierConfig(
        labels=list(LanguageClassifierImplementation.LANGUAGE_NAMES.keys()),
        cache_size=cache_size,
        min_confidence=min_confidence,
        cost=cost,
        params=params,
    )

    # Create implementation
    implementation = LanguageClassifierImplementation(config)

    # Create and return classifier
    return Classifier(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
