"""
Language detection classifiers for Sifaka.

This module provides classifiers for detecting the language of text content using
the langdetect library, which is a port of Google's language-detection library.

## Overview
The LanguageClassifier is a specialized classifier that identifies the language of
text content with high accuracy. It supports over 50 languages and provides detailed
confidence scores and language metadata. The classifier is designed to be lightweight,
fast, and easy to use, making it ideal for content analysis and preprocessing tasks.

## Architecture
LanguageClassifier follows the standard Sifaka classifier architecture:
1. **Public API**: classify() and batch_classify() methods (inherited)
2. **Caching Layer**: _classify_impl() handles caching (inherited)
3. **Core Logic**: _classify_impl_uncached() implements language detection
4. **State Management**: Uses StateManager for internal state
5. **Fallback Handling**: Configurable fallback language when confidence is low
6. **Detector Loading**: On-demand loading of the language detection engine

## Lifecycle
1. **Initialization**: Set up configuration and parameters
   - Initialize with name, description, and config
   - Extract parameters from config.params
   - Set up default values

2. **Warm-up**: Load language detector resources
   - Load langdetect when needed
   - Initialize only once
   - Handle initialization errors gracefully

3. **Classification**: Process input text
   - Validate input text
   - Apply language detection
   - Convert scores to standardized format
   - Handle empty text and edge cases

4. **Result Creation**: Return standardized results
   - Map language codes to full language names
   - Include confidence scores
   - Provide detailed language probabilities in metadata

## Usage Examples
```python
from sifaka.classifiers.implementations.properties.language import create_language_classifier

# Create a language classifier with default settings
classifier = create_language_classifier()

# Classify text
result = classifier.classify("Hello, world!")
print(f"Language: {result.label}, Name: {result.metadata['language_name']}")
print(f"Confidence: {result.confidence:.2f}")

# Create a classifier with custom settings
custom_classifier = create_language_classifier(
    min_confidence=0.2,  # Higher threshold for language detection
    fallback_lang="fr",  # Use French as fallback language
    cache_size=100       # Enable caching
)

# Batch classify multiple texts
texts = [
    "Hello, world!",
    "Bonjour le monde!",
    "Hola mundo!"
]
results = custom_classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Language: {result.label}, Name: {result.metadata['language_name']}")
    print(f"Confidence: {result.confidence:.2f}")
```

## Error Handling
The classifier provides robust error handling:
- ImportError: When langdetect is not installed
- RuntimeError: When detector initialization fails
- Graceful handling of empty or invalid inputs
- Fallback to a configurable language when confidence is low

## Configuration
Key configuration options include:
- min_confidence: Threshold for language detection (default: 0.1)
- fallback_lang: Language code to use when confidence is too low (default: "en")
- fallback_confidence: Confidence to assign to fallback language (default: 0.0)
- seed: Random seed for consistent results (default: 0)
- cache_size: Size of the classification cache (0 to disable)
"""

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable
from typing_extensions import TypeGuard
from sifaka.classifiers.classifier import Classifier
from sifaka.core.results import ClassificationResult
from sifaka.utils.config.classifiers import ClassifierConfig, standardize_classifier_config
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class LanguageDetector(Protocol):
    """
    Protocol for language detection engines.

    This protocol defines the interface that any language detector must implement
    to be compatible with the LanguageClassifier. It requires methods for detecting
    the most likely language and for retrieving a list of possible languages with
    confidence scores.

    ## Architecture
    The protocol follows a standard interface pattern:
    - Uses Python's typing.Protocol for structural subtyping
    - Is runtime checkable for dynamic type verification
    - Defines two required methods with clear input/output contracts
    - Enables pluggable language detection implementations

    ## Implementation Requirements
    1. Implement detect_langs() method that accepts a string and returns a sequence
       of language probability objects
    2. Implement detect() method that accepts a string and returns the most likely
       language code
    3. The language probability objects should have 'lang' and 'prob' attributes
    4. The 'lang' attribute should contain the language code (e.g., "en", "fr")
    5. The 'prob' attribute should contain a confidence score between 0.0 and 1.0

    ## Examples
    ```python
    from sifaka.classifiers.implementations.properties.language import LanguageDetector

    class CustomDetector:
        def detect_langs(self, text: str) -> List[Any]:
            # Simple implementation based on keywords
            langs = []

            # Create language probability objects
            class LangProb:
                def __init__(self, lang: Any, prob: Any) -> None:
                    self.lang = lang
                    self.prob = prob

            # Check for English keywords
            if "the" in text or "is" in text or "and" in text:
                langs.append(LangProb("en", 0.8))

            # Check for French keywords
            if "le" in text or "la" in text or "et" in text:
                langs.append(LangProb("fr", 0.7))

            # Sort by probability
            langs.sort(key=lambda x: x.prob, reverse=True)
            return langs

        def detect(self, text: str) -> str:
            langs = self.detect_langs(text)
            return langs[0].lang if langs else "en"

    # Verify protocol compliance
    detector = CustomDetector()
    assert isinstance(detector, LanguageDetector)
    ```
    """

    @abstractmethod
    def detect_langs(self, text: str) -> Sequence[Any]: ...

    @abstractmethod
    def detect(self, text: str) -> str: ...


@dataclass(frozen=True)
class LanguageInfo:
    """
    Information about a supported language.

    This immutable data class encapsulates information about a language,
    including its code, name, native name, and script. It provides a
    standardized way to represent language information throughout the
    classifier system.

    ## Architecture
    The class uses Python's dataclass with frozen=True for:
    - Immutability to prevent accidental modification
    - Automatic implementation of __init__, __repr__, and __eq__
    - Type hints for all fields
    - Default values for optional fields

    ## Examples
    ```python
    # Create language information for English
    english = LanguageInfo(
        code="en",
        name="English",
        native_name="English",
        script="Latin"
    )

    # Create language information for Japanese
    japanese = LanguageInfo(
        code="ja",
        name="Japanese",
        native_name="日本語",
        script="Kanji/Hiragana/Katakana"
    )

    # Access language information
    print(f"Language: {english.name} ({english.code})")
    print(f"Native name: {japanese.native_name}")
    print(f"Script: {japanese.script}")
    ```

    Attributes:
        code: ISO language code (e.g., "en", "fr", "ja")
        name: English name of the language
        native_name: Name of the language in the language itself
        script: Writing system used by the language
    """

    code: str
    name: str
    native_name: Optional[str] = None
    script: Optional[str] = None


class LanguageClassifier(Classifier):
    """
    A lightweight language classifier using langdetect.

    This classifier detects the language of text using the langdetect library,
    which is a port of Google's language-detection library. It supports over 50
    languages and provides detailed confidence scores and language metadata.

    ## Architecture
    LanguageClassifier follows a component-based architecture:
    - Extends the base Classifier class for consistent interface
    - Uses langdetect for language detection
    - Implements configurable thresholds for language confidence
    - Provides detailed language information in result metadata
    - Uses StateManager for efficient state tracking and caching
    - Supports both synchronous and batch classification

    ## Lifecycle
    1. **Initialization**: Set up configuration and parameters
       - Initialize with name, description, and config
       - Extract parameters from config.params
       - Set up default values and constants

    2. **Warm-up**: Load language detector resources
       - Load langdetect when needed (lazy initialization)
       - Initialize only once and cache for reuse
       - Handle initialization errors gracefully with clear messages

    3. **Classification**: Process input text
       - Validate input text and handle edge cases
       - Apply language detection algorithms
       - Convert scores to standardized format
       - Apply thresholds to determine language confidence
       - Map language codes to human-readable names

    4. **Result Creation**: Return standardized results
       - Map language codes to appropriate labels
       - Convert scores to confidence values
       - Include detailed language information in metadata
       - Track statistics for monitoring and debugging

    ## Examples
    ```python
    from sifaka.classifiers.implementations.properties.language import create_language_classifier

    # Create a language classifier with default settings
    classifier = create_language_classifier()

    # Classify text
    result = classifier.classify("Hello, world!")
    print(f"Language: {result.label}, Name: {result.metadata['language_name']}")
    print(f"Confidence: {result.confidence:.2f}")

    # Access all detected languages
    for lang_info in result.metadata.get('all_langs', []):
        print(f"{lang_info['name']}: {lang_info['prob']:.2f}")
    ```

    ## Configuration Options
    - min_confidence: Threshold for language detection (default: 0.1)
    - fallback_lang: Language code to use when confidence is too low (default: "en")
    - fallback_confidence: Confidence to assign to fallback language (default: 0.0)
    - seed: Random seed for consistent results (default: 0)
    - cache_size: Size of the classification cache (0 to disable)

    Requires the 'language' extra to be installed:
    pip install sifaka[language]
    """

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
    DEFAULT_COST: int = 1

    def __init__(
        self,
        name: str = "language_classifier",
        description: str = "Detects text language",
        detector: Optional[LanguageDetector] = None,
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
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
        if config is None:
            params = kwargs.pop("params", {})
            config = ClassifierConfig[str](
                params=params,
                **kwargs,
            )
        super().__init__(implementation=self, name=name, description=description, config=config)
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
            if self._state_manager.get("cache", {}).get("detector"):
                detector = self._state_manager.get("cache")["detector"]
                if isinstance(detector, LanguageDetector):
                    return detector

            langdetect = importlib.import_module("langdetect")
            seed = self.config.params.get("seed", 0)
            langdetect.DetectorFactory.seed = seed

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
            if self._validate_detector(detector):
                cache = self._state_manager.get("cache", {})
                cache["detector"] = detector
                self._state_manager.update("cache", cache)
                return detector

            raise ValueError("Failed to validate LangDetectWrapper as a valid LanguageDetector")
        except ImportError:
            raise ImportError(
                "langdetect package is required for LanguageClassifier. Install it with: pip install sifaka[language]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load langdetect: {e}")

    def initialize(self) -> None:
        """Initialize the language detector."""
        detector = self._load_langdetect()
        cache = self._state_manager.get("cache", {})
        cache["detector"] = detector
        self._state_manager.update("cache", cache)

    def warm_up(self) -> None:
        """Initialize the language detector if needed."""
        if not self._state_manager.get("initialized", False):
            self.initialize()

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
        if not self._state_manager.get("initialized", False):
            self.warm_up()
        min_confidence = self.config.params.get("min_confidence", 0.1)
        fallback_lang = self.config.params.get("fallback_lang", "en")
        fallback_confidence = self.config.params.get("fallback_confidence", 0.0)

        try:
            detector = self._state_manager.get("cache", {}).get("detector")
            if not detector:
                raise RuntimeError("Language detector not initialized")
            lang_probs = detector.detect_langs(text)
            best_lang = None
            best_prob = 0.0
            for lang_prob in lang_probs:
                lang_code = getattr(lang_prob, "lang", None)
                prob = float(getattr(lang_prob, "prob", 0.0))
                if lang_code and prob > best_prob:
                    best_lang = lang_code
                    best_prob = prob
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
                    passed=False,
                    message=(
                        "Language detection failed: confidence too low"
                        if best_lang
                        else "No language detected"
                    ),
                )
            result: ClassificationResult = ClassificationResult(
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
                passed=True,
                message=f"Detected language: {self.get_language_name(best_lang)} with confidence {best_prob:.2f}",
            )
            stats = self._state_manager.get("statistics", {})
            stats[best_lang] = stats.get(best_lang, 0) + 1
            self._state_manager.update("statistics", stats)
            return result
        except Exception as e:
            logger.error("Failed to detect language: %s", e)
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
                passed=False,
                message=f"Language detection error: {str(e)}",
            )

    def validate_batch_input(self, texts: List[str]) -> None:
        """
        Validate batch input.

        Args:
            texts: List of texts to validate

        Raises:
            ValueError: If texts is not a list or contains non-string elements
        """
        if not isinstance(texts, list):
            raise ValueError(f"Expected list of texts, got {type(texts)}")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in texts must be strings")

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
        stats: Dict[str, Any] = super().get_statistics()
        stats.update({"seed": self.config.params.get("seed", 0)})
        statistics = self._state_manager.get("statistics", {})
        if not isinstance(statistics, dict):
            statistics = {}
        detected_languages = set(statistics.keys())
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
        super().clear_cache()
        cache = self._state_manager.get("cache", {})
        preserved_cache = {k: v for k, v in cache.items() if k == "detector"}
        self._state_manager.update("cache", preserved_cache)

    @classmethod
    def create_with_custom_detector(
        cls,
        detector: LanguageDetector,
        name: str = "custom_language_classifier",
        description: str = "Custom language detector",
        config: Optional[ClassifierConfig[str]] = None,
        **kwargs: Any,
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
        if not isinstance(detector, LanguageDetector):
            raise ValueError(
                f"Detector must implement LanguageDetector protocol, got {type(detector)}"
            )
        if config is None:
            params = kwargs.pop("params", {})
            config = ClassifierConfig[str](params=params, **kwargs)
        instance = cls(name=name, description=description, detector=detector, config=config)
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
    config: Optional[Union[Dict[str, Any], ClassifierConfig[str]]] = None,
    **kwargs: Any,
) -> LanguageClassifier:
    """
    Factory function to create a language classifier.

    This function provides a simpler interface for creating a language classifier
    with the specified parameters, handling the creation of the ClassifierConfig
    object and setting up the classifier with the appropriate parameters.

    ## Architecture
    The factory function follows a standardized pattern:
    1. Extract and prepare parameters for configuration
    2. Create a configuration dictionary with standardized structure
    3. Pass the configuration to the classifier constructor
    4. Return the fully configured classifier instance

    ## Examples
    ```python
    from sifaka.classifiers.implementations.properties.language import create_language_classifier

    # Create with default settings
    classifier = create_language_classifier()

    # Create with custom confidence threshold
    sensitive_classifier = create_language_classifier(
        min_confidence=0.2,  # Higher threshold for language detection
        fallback_lang="fr",  # Use French as fallback language
        cache_size=100       # Enable caching
    )

    # Create with custom name and description
    custom_classifier = create_language_classifier(
        name="custom_language_classifier",
        description="Custom language detector with specific settings",
        seed=42  # Set random seed for reproducible results
    )
    ```

    Args:
        name: Name of the classifier for identification and logging
        description: Human-readable description of the classifier's purpose
        min_confidence: Minimum confidence threshold for language detection
        fallback_lang: Language code to use when confidence is too low
        fallback_confidence: Confidence to assign to fallback language
        seed: Random seed for consistent results
        cache_size: Size of the classification cache (0 to disable caching)
        cost: Computational cost metric for resource allocation decisions
        config: Optional classifier configuration (dict or ClassifierConfig)
        **kwargs: Additional configuration parameters to pass to the classifier

    Returns:
        Configured LanguageClassifier instance ready for immediate use

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
    classifier_config = standardize_classifier_config(
        config=config,
        cache_size=cache_size,
        params={
            "min_confidence": min_confidence,
            "fallback_lang": fallback_lang,
            "fallback_confidence": fallback_confidence,
            "seed": seed,
        },
        **kwargs,
    )
    return LanguageClassifier(name=name, description=description, config=classifier_config)
