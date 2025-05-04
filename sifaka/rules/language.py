"""
Language validation rules for Sifaka.

This module provides rules for validating the language of text, ensuring
that text is in the expected language(s).

.. warning::
    This module is deprecated and will be removed in a future version.
    Use :mod:`sifaka.rules.content.language` instead.
"""

import importlib
import warnings
from typing import List, Optional, Any

from sifaka.rules.base import Rule, RuleConfig, RuleResult

# Issue deprecation warning
warnings.warn(
    "The module 'sifaka.rules.language' is deprecated and will be removed in a future version. "
    "Use 'sifaka.rules.content.language' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def create_language_rule(
    allowed_languages: Optional[List[str]] = None,
    threshold: float = 0.7,
    name: str = "language_rule",
    description: str = "Validates that text is in the allowed language(s)",
    config: Optional[RuleConfig] = None,
    **kwargs: Any,
) -> Rule:
    """
    .. deprecated:: 0.1.0
        Use :func:`sifaka.rules.content.language.create_language_rule` instead.
    """
    warnings.warn(
        "The function 'create_language_rule' in 'sifaka.rules.language' is deprecated. "
        "Use 'sifaka.rules.content.language.create_language_rule' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """
    Create a rule that validates text language.

    This factory function creates a rule that ensures text is in one of the
    allowed languages. It uses langdetect to detect the language of the text
    and validates that it matches one of the allowed languages.

    Args:
        allowed_languages: List of language codes that are allowed (e.g., ["en", "fr"])
        threshold: Confidence threshold for language detection
        name: Name of the rule
        description: Description of the rule
        config: Optional rule configuration
        **kwargs: Additional parameters for the rule

    Returns:
        A rule that validates text language

    Examples:
        ```python
        from sifaka.rules.language import create_language_rule

        # Create a rule that only allows English
        rule = create_language_rule(allowed_languages=["en"])

        # Create a rule that allows English or French
        rule = create_language_rule(
            allowed_languages=["en", "fr"],
            threshold=0.6,
            name="english_or_french_rule"
        )

        # Validate text
        result = rule.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'language' extra to be installed:
    pip install sifaka[language]
    """
    # Default to English if no languages specified
    if allowed_languages is None:
        allowed_languages = ["en"]

    # Import the FunctionRule class
    from sifaka.rules.base import FunctionRule

    # Create a language validation function
    def validate_language(text: str) -> RuleResult:
        """Validate that text is in one of the allowed languages."""
        # Handle empty text
        if not text or not text.strip():
            return RuleResult(
                passed=False,
                message="Empty text cannot be validated for language",
                metadata={"reason": "empty_text"},
            )

        try:
            # Import langdetect
            try:
                langdetect = importlib.import_module("langdetect")
                langdetect.DetectorFactory.seed = 0  # For consistent results
            except ImportError:
                return RuleResult(
                    passed=False,
                    message="langdetect package is required for language validation. Install with: pip install langdetect",
                    metadata={"reason": "missing_dependency"},
                )

            # Detect language
            try:
                lang_probs = langdetect.detect_langs(text)

                # Find the most likely language
                best_lang = None
                best_prob = 0.0

                for lang_prob in lang_probs:
                    lang_code = getattr(lang_prob, "lang", None)
                    prob = float(getattr(lang_prob, "prob", 0.0))

                    if lang_code and prob > best_prob:
                        best_lang = lang_code
                        best_prob = prob

                # Check if language is allowed and confidence is high enough
                if best_lang in allowed_languages and best_prob >= threshold:
                    return RuleResult(
                        passed=True,
                        message=f"Text is in an allowed language: {best_lang}",
                        metadata={
                            "language": best_lang,
                            "confidence": best_prob,
                            "allowed_languages": allowed_languages,
                        },
                    )
                else:
                    # Failed validation
                    if best_lang not in allowed_languages:
                        message = f"Text is in {best_lang}, which is not an allowed language"
                    else:
                        message = f"Language detection confidence ({best_prob:.2f}) is below threshold ({threshold})"

                    return RuleResult(
                        passed=False,
                        message=message,
                        metadata={
                            "language": best_lang,
                            "confidence": best_prob,
                            "allowed_languages": allowed_languages,
                            "threshold": threshold,
                        },
                    )
            except Exception as e:
                return RuleResult(
                    passed=False,
                    message=f"Language detection failed: {str(e)}",
                    metadata={"error": str(e)},
                )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Language validation error: {str(e)}",
                metadata={"error": str(e)},
            )

    # Create a function rule
    return FunctionRule(
        func=validate_language,
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
