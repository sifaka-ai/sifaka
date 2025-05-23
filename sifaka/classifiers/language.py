"""Language classifier for detecting the language of text.

This module provides a classifier for detecting the language of text using
the langdetect library with fallback to simple heuristics.
"""

import importlib
from typing import Any, Dict, List, Optional

from sifaka.classifiers.base import ClassificationResult, ClassifierError, TextClassifier
from sifaka.utils.logging import get_logger
from sifaka.validators.classifier import ClassifierValidator

# Configure logger
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


class LanguageClassifier(TextClassifier):
    """Classifier for detecting the language of text.
    
    This classifier uses the langdetect library when available,
    with fallback to simple pattern-based detection. It can detect
    over 50 languages and provides confidence scores.
    
    Attributes:
        min_confidence: Minimum confidence threshold for detection
        fallback_lang: Language to use when confidence is too low
        detector: The language detection library instance
    """
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        fallback_lang: str = "en",
        fallback_confidence: float = 0.5,
        seed: Optional[int] = None,
        name: str = "LanguageClassifier",
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
            logger.debug("Initialized language classifier with langdetect")
            
        except ImportError:
            logger.warning(
                "langdetect not available. Using pattern-based detection. "
                "Install langdetect for better accuracy: pip install langdetect"
            )
            self.detector = None
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text for language.
        
        Args:
            text: The text to classify
            
        Returns:
            ClassificationResult with language prediction
            
        Raises:
            ClassifierError: If classification fails
        """
        if not text or not text.strip():
            return ClassificationResult(
                label=self.fallback_lang,
                confidence=self.fallback_confidence,
                metadata={
                    "reason": "empty_text",
                    "input_length": 0,
                    "language_name": self.get_language_name(self.fallback_lang),
                }
            )
        
        try:
            if self.detector is not None:
                return self._classify_with_library(text)
            else:
                return self._classify_with_patterns(text)
                
        except Exception as e:
            logger.error(f"Language classification failed: {e}")
            # Return fallback result instead of raising error
            return ClassificationResult(
                label=self.fallback_lang,
                confidence=self.fallback_confidence,
                metadata={
                    "reason": "detection_error",
                    "error": str(e),
                    "input_length": len(text),
                    "language_name": self.get_language_name(self.fallback_lang),
                }
            )
    
    def _classify_with_library(self, text: str) -> ClassificationResult:
        """Classify using langdetect library."""
        try:
            # Detect language with probabilities
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
            
            return ClassificationResult(
                label=detected_lang,
                confidence=confidence,
                metadata={
                    "method": "langdetect",
                    "language_name": self.get_language_name(detected_lang),
                    "input_length": len(text),
                    "all_langs": [
                        {
                            "lang": getattr(lp, "lang", None),
                            "prob": float(getattr(lp, "prob", 0.0)),
                            "name": self.get_language_name(getattr(lp, "lang", "")),
                        }
                        for lp in lang_probs
                    ],
                }
            )
            
        except Exception as e:
            # Fallback to pattern-based detection
            logger.warning(f"langdetect failed, using pattern detection: {e}")
            return self._classify_with_patterns(text)
    
    def _classify_with_patterns(self, text: str) -> ClassificationResult:
        """Classify using simple pattern-based approach."""
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
            best_lang = max(lang_scores, key=lang_scores.get)
            confidence = min(lang_scores[best_lang] * 2, 0.8)  # Conservative confidence
        else:
            best_lang = self.fallback_lang
            confidence = self.fallback_confidence
        
        return ClassificationResult(
            label=best_lang,
            confidence=confidence,
            metadata={
                "method": "pattern_based",
                "language_name": self.get_language_name(best_lang),
                "input_length": len(text),
                "pattern_scores": lang_scores,
            }
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


def create_language_validator(
    required_languages: Optional[List[str]] = None,
    min_confidence: float = 0.7,
    name: str = "LanguageValidator"
) -> ClassifierValidator:
    """Create a validator that checks text language.
    
    Args:
        required_languages: List of acceptable language codes (e.g., ['en', 'es'])
        min_confidence: Minimum confidence for language detection
        name: Name of the validator
        
    Returns:
        A ClassifierValidator configured for language validation
    """
    classifier = LanguageClassifier(min_confidence=min_confidence)
    
    return ClassifierValidator(
        classifier=classifier,
        threshold=min_confidence,
        valid_labels=required_languages,  # Only these languages are valid
        name=name
    )
