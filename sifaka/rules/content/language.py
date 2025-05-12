"""
Language validation rules for Sifaka.

This module provides rules for validating the language of text, ensuring
that text is in the expected language(s).

Usage Example:
    ```python
    from sifaka.rules.content.language import create_language_rule

    # Create a language rule
    rule = create_language_rule(
        allowed_languages=["en", "fr"],
        threshold=0.7
    )

    # Validate text
    result = (rule and rule.validate("This is English text.")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
"""
import importlib
import time
from typing import List, Optional, Any, Dict
from pydantic import Field, PrivateAttr
from sifaka.rules.base import Rule, RuleConfig, RuleResult, BaseValidator
from sifaka.utils.logging import get_logger
logger = get_logger(__name__)


class LanguageValidator(BaseValidator[str]):
    """
    Validator for text language.

    This validator checks if text is in one of the allowed languages
    using the langdetect library.

    Lifecycle:
        1. Initialization: Set up with allowed languages and threshold
        2. Validation: Detect language and check against allowed languages
        3. Result: Return detailed validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.language import LanguageValidator

        # Create validator
        validator = LanguageValidator(
            allowed_languages=["en", "fr"],
            threshold=0.7
        )

        # Validate text
        result = (validator and validator.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Attributes:
        allowed_languages: List of language codes that are allowed
        threshold: Confidence threshold for language detection
    """
    allowed_languages: List[str] = Field(default=['en'], description=
        'List of language codes that are allowed')
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description=
        'Confidence threshold for language detection')
    _langdetect = PrivateAttr(default=None)

    def __init__(self, allowed_languages: Optional[Optional[List[str]]] = None,
        threshold: float=0.7, **kwargs) ->None:
        """
        Initialize the validator.

        Args:
            allowed_languages: List of language codes that are allowed
            threshold: Confidence threshold for language detection
            **kwargs: Additional keyword arguments
        """
        super().__init__(validation_type=str, **kwargs)
        if allowed_languages is not None:
            self.allowed_languages = allowed_languages
        self.threshold = threshold
        try:
            self._langdetect = (importlib and importlib.import_module('langdetect')
            self._langdetect.DetectorFactory.seed = 0
        except ImportError:
            self._langdetect = None
            (logger and logger.warning(
                'langdetect package is not installed. Language validation will fail.'
                )

    def validate(self, text: str) ->RuleResult:
        """
        Validate that text is in one of the allowed languages.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = (time.time()
        empty_result = (self.handle_empty_text(text)
        if empty_result:
            return empty_result
        if self._langdetect is None:
            result = RuleResult(passed=False, message=
                'langdetect package is required for language validation. Install with: pip install langdetect'
                , metadata={'reason': 'missing_dependency',
                'validator_type': self.__class__.__name__}, score=0.0,
                issues=['Missing dependency: langdetect'], suggestions=[
                'Install langdetect with: pip install langdetect'],
                processing_time_ms=(time.time() - start_time)
            (self.update_statistics(result)
            return result
        try:
            lang_probs = self.(_langdetect.detect_langs(text)
            best_lang = None
            best_prob = 0.0
            for lang_prob in lang_probs:
                lang_code = getattr(lang_prob, 'lang', None)
                prob = float(getattr(lang_prob, 'prob', 0.0))
                if lang_code and prob > best_prob:
                    best_lang = lang_code
                    best_prob = prob
            if (best_lang in self.allowed_languages and best_prob >= self.
                threshold):
                result = RuleResult(passed=True, message=
                    f'Text is in an allowed language: {best_lang}',
                    metadata={'language': best_lang, 'confidence':
                    best_prob, 'allowed_languages': self.allowed_languages,
                    'validator_type': self.__class__.__name__}, score=
                    best_prob, issues=[], suggestions=[],
                    processing_time_ms=(time.time() - start_time)
                (self.update_statistics(result)
                return result
            else:
                issues = []
                suggestions = []
                if best_lang not in self.allowed_languages:
                    message = (
                        f'Text is in {best_lang}, which is not an allowed language'
                        )
                    (issues.append(message)
                    (suggestions.append(
                        f"Translate text to one of the allowed languages: {', '.join(self.allowed_languages)}"
                        )
                else:
                    message = (
                        f'Language detection confidence ({best_prob:.2f}) is below threshold ({self.threshold})'
                        )
                    (issues.append(message)
                    (suggestions.append(
                        'Make the text more clearly in the target language')
                result = RuleResult(passed=False, message=message, metadata
                    ={'language': best_lang, 'confidence': best_prob,
                    'allowed_languages': self.allowed_languages,
                    'threshold': self.threshold, 'validator_type': self.
                    __class__.__name__}, score=best_prob / self.threshold if
                    self.threshold > 0 else 0.0, issues=issues, suggestions
                    =suggestions, processing_time_ms=(time.time() - start_time)
                (self.update_statistics(result)
                return result
        except Exception as e:
            (self.record_error(e)
            (logger.error(f'Language detection failed: {e}')
            result = RuleResult(passed=False, message=
                f'Language detection failed: {str(e)}', metadata={'error':
                str(e), 'error_type': type(e).__name__, 'validator_type':
                self.__class__.__name__}, score=0.0, issues=[
                f'Language detection error: {str(e)}'], suggestions=[
                'Check if the text is valid and try again'],
                processing_time_ms=(time.time() - start_time)
            (self.update_statistics(result)
            return result


class LanguageRule(Rule[str]):
    """
    Rule that validates text language.

    This rule ensures that text is in one of the allowed languages.
    It uses the langdetect library to detect the language of the text
    and validates that it matches one of the allowed languages.

    Lifecycle:
        1. Initialization: Set up with allowed languages and threshold
        2. Validation: Delegate to validator to detect language and check against allowed languages
        3. Result: Return standardized validation results with metadata

    Examples:
        ```python
        from sifaka.rules.content.language import LanguageRule, LanguageValidator

        # Create validator
        validator = LanguageValidator(
            allowed_languages=["en", "fr"],
            threshold=0.7
        )

        # Create rule
        rule = LanguageRule(
            name="language_rule",
            description="Validates text language",
            validator=validator
        )

        # Validate text
        result = (rule.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, name: str='language_rule', description: str=
        'Validates that text is in the allowed language(s)', config:
        Optional[RuleConfig]=None, validator: Optional[LanguageValidator]=
        None, **kwargs) ->None:
        """
        Initialize the language rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(name=name, description=description, config=config or
            RuleConfig(name=name, description=description, rule_id=kwargs.
            pop('rule_id', name), **kwargs), validator=validator)
        self._language_validator = validator or (self._create_default_validator(
            )

    def _create_default_validator(self) ->LanguageValidator:
        """
        Create a default validator.

        Returns:
            A LanguageValidator with default settings
        """
        params = self.config.params
        return LanguageValidator(allowed_languages=(params.get(
            'allowed_languages', ['en']), threshold=(params.get('threshold',
            0.7))


def create_language_validator(allowed_languages: Optional[Optional[List[str]]] = None,
    threshold: float=0.7, **kwargs: Any) ->LanguageValidator:
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

        # Create a validator that allows English or French
        validator = create_language_validator(
            allowed_languages=["en", "fr"],
            threshold=0.6
        )

        # Validate text
        result = (validator.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'langdetect' package to be installed:
    pip install langdetect
    """
    if allowed_languages is None:
        allowed_languages = ['en']
    return LanguageValidator(allowed_languages=allowed_languages, threshold
        =threshold, **kwargs)


def create_language_rule(allowed_languages: Optional[Optional[List[str]]] = None,
    threshold: float=0.7, name: str='language_rule', description: str=
    'Validates that text is in the allowed language(s)', rule_id: Optional[
    str]=None, config: Optional[RuleConfig]=None, **kwargs: Any
    ) ->LanguageRule:
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
        rule_id: Unique identifier for the rule
        config: Optional rule configuration
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

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
            name="english_or_french_rule",
            rule_id="language_validator",
            severity="warning",
            category="content",
            tags=["language", "content", "validation"]
        )

        # Validate text
        result = (rule.validate("This is English text.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Requires the 'langdetect' package to be installed:
    pip install langdetect
    """
    if allowed_languages is None:
        allowed_languages = ['en']
    validator = create_language_validator(allowed_languages=
        allowed_languages, threshold=threshold)
    params = {'allowed_languages': allowed_languages, 'threshold': threshold}
    rule_name = name or rule_id or 'language_rule'
    rule_config = config or RuleConfig(name=rule_name, description=
        description, rule_id=rule_id or rule_name, params=params, **kwargs)
    return LanguageRule(name=rule_name, description=description, config=
        rule_config, validator=validator)
