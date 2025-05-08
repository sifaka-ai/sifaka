"""
Language validation rules for Sifaka.

This module provides rules for validating the language of text, ensuring
that text is in the expected language(s).
"""

import importlib
from typing import List, Optional, Any, Dict, Union

from pydantic import Field, PrivateAttr

from sifaka.rules.base import (
    Rule,
    RuleConfig,
    RuleResult,
    BaseValidator,
    create_rule,
)
from sifaka.utils.state import StateManager, RuleState, create_rule_state


class LanguageValidator(BaseValidator[str]):
    """
    Validator for text language.

    This validator checks if text is in one of the allowed languages
    using the langdetect library.

    Attributes:
        allowed_languages: List of language codes that are allowed
        threshold: Confidence threshold for language detection
    """

    allowed_languages: List[str] = Field(
        default=["en"],
        description="List of language codes that are allowed",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for language detection",
    )

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(
        self, allowed_languages: Optional[List[str]] = None, threshold: float = 0.7, **kwargs
    ):
        """Initialize the validator."""
        super().__init__(**kwargs)
        if allowed_languages is not None:
            self.allowed_languages = allowed_languages
        self.threshold = threshold

    def warm_up(self) -> None:
        """Initialize the validator if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            try:
                state.langdetect = importlib.import_module("langdetect")
                # Set seed for consistent results
                state.langdetect.DetectorFactory.seed = 0
            except ImportError:
                state.langdetect = None
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate that text is in one of the allowed languages.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Check if langdetect is available
        if state.langdetect is None:
            return RuleResult(
                passed=False,
                message="langdetect package is required for language validation. Install with: pip install langdetect",
                metadata={"reason": "missing_dependency"},
            )

        try:
            # Detect language
            lang_probs = state.langdetect.detect_langs(text)

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
            if best_lang in self.allowed_languages and best_prob >= self.threshold:
                return RuleResult(
                    passed=True,
                    message=f"Text is in an allowed language: {best_lang}",
                    metadata={
                        "language": best_lang,
                        "confidence": best_prob,
                        "allowed_languages": self.allowed_languages,
                    },
                )
            else:
                # Failed validation
                if best_lang not in self.allowed_languages:
                    message = f"Text is in {best_lang}, which is not an allowed language"
                else:
                    message = f"Language detection confidence ({best_prob:.2f}) is below threshold ({self.threshold})"

                return RuleResult(
                    passed=False,
                    message=message,
                    metadata={
                        "language": best_lang,
                        "confidence": best_prob,
                        "allowed_languages": self.allowed_languages,
                        "threshold": self.threshold,
                    },
                )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Language detection failed: {str(e)}",
                metadata={"error": str(e)},
            )


class LanguageRule(Rule[str, RuleResult, LanguageValidator, Any]):
    """
    Rule that validates text language.

    This rule ensures that text is in one of the allowed languages.
    It uses the langdetect library to detect the language of the text
    and validates that it matches one of the allowed languages.

    Attributes:
        _name: The name of the rule
        _description: Description of the rule
        _config: Rule configuration
        _validator: The validator used by this rule
    """

    # State management
    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def warm_up(self) -> None:
        """Initialize the rule if needed."""
        if not self._state_manager.is_initialized:
            state = self._state_manager.get_state()
            state.validator = self._create_default_validator()
            state.initialized = True

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """Validate the text."""
        # Ensure resources are initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Check cache
        cache_key = text
        if cache_key in state.cache:
            return state.cache[cache_key]

        # Delegate to validator
        result = state.validator.validate(text, **kwargs)

        # Cache result
        state.cache[cache_key] = result

        return result

    def _create_default_validator(self) -> LanguageValidator:
        """
        Create a default validator.

        Returns:
            A LanguageValidator with default settings
        """
        params = self._config.params
        return LanguageValidator(
            allowed_languages=params.get("allowed_languages", ["en"]),
            threshold=params.get("threshold", 0.7),
        )


def create_language_validator(
    allowed_languages: Optional[List[str]] = None,
    threshold: float = 0.7,
    **kwargs: Any,
) -> LanguageValidator:
    """
    Create a language validator.

    This factory function creates a validator that ensures text is in one of the
    allowed languages. It uses langdetect to detect the language of the text
    and validates that it matches one of the allowed languages.

    Args:
        allowed_languages: List of language codes that are allowed (e.g., ["en", "fr"])
        threshold: Confidence threshold for language detection
        **kwargs: Additional parameters for the validator

    Returns:
        A validator that validates text language

    Examples:
        ```python
        from sifaka.rules.content.language import create_language_validator

        # Create a validator that only allows English
        validator = create_language_validator(allowed_languages=["en"])

        # Validate text
        result = validator.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'langdetect' package to be installed:
    pip install langdetect
    """
    if allowed_languages is None:
        allowed_languages = ["en"]

    return LanguageValidator(
        allowed_languages=allowed_languages,
        threshold=threshold,
        **kwargs,
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
        **kwargs: Additional parameters for the rule or validator

    Returns:
        A rule that validates text language

    Examples:
        ```python
        from sifaka.rules.content.language import create_language_rule

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

    Requires the 'langdetect' package to be installed:
    pip install langdetect
    """
    # Default to English if no languages specified
    if allowed_languages is None:
        allowed_languages = ["en"]

    # Create rule configuration
    rule_config = config or RuleConfig()
    rule_config = rule_config.with_params(
        allowed_languages=allowed_languages,
        threshold=threshold,
        **{k: v for k, v in kwargs.items() if k not in ["name", "description"]},
    )

    # Create validator
    validator = create_language_validator(
        allowed_languages=allowed_languages,
        threshold=threshold,
    )

    # Create rule
    return create_rule(
        rule_type=LanguageRule,
        name=name,
        description=description,
        config=rule_config,
        validator=validator,
    )
