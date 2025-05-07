"""
Safety-related content validation rules for Sifaka.

This module provides rules for validating text against various safety concerns,
including toxicity, bias, and harmful content.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - Factory functions handle configuration
    - Validator factory functions create standalone validators

Usage Example:
    from sifaka.rules.content.safety import create_toxicity_rule, create_bias_rule, create_harmful_content_rule

    # Create a toxicity rule
    toxicity_rule = create_toxicity_rule(threshold=0.4)

    # Create a bias rule
    bias_rule = create_bias_rule(threshold=0.3)

    # Create a harmful content rule
    harmful_rule = create_harmful_content_rule(
        categories={
            "violence": ["violent", "threatening"],
            "misinformation": ["false", "misleading"]
        }
    )

    # Validate text
    result = toxicity_rule.validate("This is a test.")
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.classifiers.toxicity import ToxicityClassifier
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    ValidationError,
    RuleResultHandler,
)
from sifaka.rules.content.base import CategoryAnalyzer
from sifaka.adapters.classifier import create_classifier_rule


__all__ = [
    # Config classes
    "HarmfulContentConfig",
    # Validator classes
    "HarmfulContentValidator",
    # Rule classes
    "HarmfulContentRule",
    # Factory functions
    "create_toxicity_validator",
    "create_toxicity_rule",
    "create_bias_validator",
    "create_bias_rule",
    "create_harmful_content_validator",
    "create_harmful_content_rule",
]


# Default harmful content categories
DEFAULT_HARMFUL_CATEGORIES: Dict[str, List[str]] = {
    "violence": [
        "violent",
        "threatening",
        "harm",
        "danger",
        "attack",
        "kill",
        "murder",
    ],
    "hate_speech": [
        "hate",
        "racist",
        "sexist",
        "discriminatory",
        "prejudiced",
        "bigoted",
    ],
    "misinformation": [
        "false",
        "misleading",
        "deceptive",
        "fake",
        "hoax",
        "conspiracy",
    ],
    "harassment": [
        "harass",
        "bully",
        "intimidate",
        "threaten",
        "stalk",
        "abuse",
    ],
}


class HarmfulContentConfig(BaseModel):
    """Configuration for harmful content validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    categories: Dict[str, List[str]] = Field(
        default_factory=lambda: DEFAULT_HARMFUL_CATEGORIES,
        description="Dictionary of harmful content categories and their indicators",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for validation",
    )
    fail_if_any: bool = Field(
        default=True,
        description="Whether to fail if any category exceeds the threshold",
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate that categories are not empty and have indicators."""
        if not v:
            raise ValueError("Categories cannot be empty")
        for category, indicators in v.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")
        return v


class HarmfulContentAnalyzer(CategoryAnalyzer):
    """Analyzer for harmful content detection."""

    def __init__(self, config: HarmfulContentConfig) -> None:
        """Initialize the analyzer."""
        super().__init__(
            categories=config.categories,
            threshold=config.threshold,
            fail_if_any=config.fail_if_any,
            higher_is_better=False,
        )


class HarmfulContentValidator(BaseValidator[str]):
    """Validator that checks for harmful content."""

    def __init__(self, config: RuleConfig) -> None:
        """Initialize the validator."""
        self._config = config
        self._harmful_config = HarmfulContentConfig(**config.params)
        self._analyzer = HarmfulContentAnalyzer(config=self._harmful_config)

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **_: Any) -> RuleResult:
        """Validate that the text does not contain harmful content.

        Args:
            text: The text to validate
            **_: Additional validation context (unused)

        Returns:
            RuleResult: The result of the validation
        """
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            result = self._analyzer.analyze(text)
            return result

        except Exception as e:
            raise ValidationError(f"Content validation failed: {str(e)}") from e


class HarmfulContentRule(
    Rule[str, RuleResult, HarmfulContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for harmful content in text."""

    def __init__(
        self,
        name: str = "harmful_content_rule",
        description: str = "Validates text for harmful content",
        config: Optional[RuleConfig] = None,
        validator: Optional[HarmfulContentValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the harmful content rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> HarmfulContentValidator:
        """Create a default validator from config."""
        return HarmfulContentValidator(self.config)


def create_harmful_content_validator(
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
    fail_if_any: bool = True,
    **kwargs: Any,
) -> HarmfulContentValidator:
    """Create a harmful content validator.

    Args:
        categories: Dictionary of harmful content categories and their indicators
        threshold: Minimum score threshold for validation
        fail_if_any: Whether to fail if any category exceeds the threshold
        **kwargs: Additional keyword arguments for the config

    Returns:
        HarmfulContentValidator: The created validator
    """
    # Create params dictionary
    params = {
        "categories": categories or DEFAULT_HARMFUL_CATEGORIES,
        "threshold": threshold,
        "fail_if_any": fail_if_any,
    }

    # Add any remaining params
    params.update(kwargs)

    # Create RuleConfig
    config = RuleConfig(params=params)

    # Create validator
    return HarmfulContentValidator(config)


def create_toxicity_validator(
    threshold: float = 0.5,
    **kwargs: Any,
) -> BaseValidator[str]:
    """
    Create a toxicity validator using the classifier adapter.

    This factory function creates a configured toxicity validator instance using the
    ToxicityClassifier through the classifier adapter.

    Args:
        threshold: Threshold for toxicity detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the validator

    Returns:
        Configured toxicity validator instance
    """
    from sifaka.adapters.classifier import ClassifierAdapter

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create classifier
    classifier = ToxicityClassifier()

    # Create adapter with classifier
    adapter = ClassifierAdapter(
        classifier=classifier, threshold=threshold, valid_labels=["non-toxic"], **kwargs
    )

    return adapter


def create_toxicity_rule(
    name: str = "toxicity_rule",
    description: str = "Validates text for toxic content",
    threshold: float = 0.5,
    **kwargs: Any,
) -> Rule[str, RuleResult, BaseValidator[str], RuleResultHandler[RuleResult]]:
    """
    Create a toxicity rule using the classifier adapter.

    This factory function creates a configured toxicity rule instance using the
    ToxicityClassifier through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for toxicity detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured toxicity rule instance
    """
    # Create rule using create_classifier_rule
    return create_classifier_rule(
        classifier=ToxicityClassifier(),
        name=name,
        description=description,
        threshold=threshold,
        valid_labels=["non-toxic"],
        **kwargs,
    )


def create_bias_validator(
    threshold: float = 0.3,
    **kwargs: Any,
) -> BaseValidator[str]:
    """
    Create a bias validator using the classifier adapter.

    This factory function creates a configured bias validator instance using the
    BiasDetector through the classifier adapter.

    Args:
        threshold: Threshold for bias detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the validator

    Returns:
        Configured bias validator instance
    """
    # Import BiasDetector here to avoid circular imports
    from sifaka.classifiers.bias import BiasDetector
    from sifaka.adapters.classifier import ClassifierAdapter

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create classifier
    classifier = BiasDetector()

    # Create adapter with classifier
    adapter = ClassifierAdapter(
        classifier=classifier, threshold=threshold, valid_labels=["unbiased"], **kwargs
    )

    return adapter


def create_bias_rule(
    name: str = "bias_rule",
    description: str = "Validates text for biased content",
    threshold: float = 0.3,
    **kwargs: Any,
) -> Rule[str, RuleResult, BaseValidator[str], RuleResultHandler[RuleResult]]:
    """
    Create a bias rule using the classifier adapter.

    This factory function creates a configured bias rule instance using the
    BiasDetector through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for bias detection (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured bias rule instance
    """
    # Import BiasDetector here to avoid circular imports
    from sifaka.classifiers.bias import BiasDetector

    return create_classifier_rule(
        classifier=BiasDetector(),
        name=name,
        description=description,
        threshold=threshold,
        valid_labels=["unbiased"],
        **kwargs,
    )


def create_harmful_content_rule(
    name: str = "harmful_content_rule",
    description: str = "Validates text for harmful content",
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
    fail_if_any: bool = True,
    **kwargs: Any,
) -> HarmfulContentRule:
    """
    Create a harmful content rule with configuration.

    This factory function creates a configured HarmfulContentRule instance.

    Args:
        name: The name of the rule
        description: Description of the rule
        categories: Dictionary of harmful content categories and their indicators
        threshold: Minimum score threshold for validation
        fail_if_any: Whether to fail if any category exceeds the threshold
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured HarmfulContentRule instance
    """
    # Extract rule-specific parameters
    validator_kwargs = {
        "categories": categories,
        "threshold": threshold,
        "fail_if_any": fail_if_any,
    }

    # Extract RuleConfig parameters
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Add remaining kwargs to validator_kwargs
    validator_kwargs.update(kwargs)

    # Create validator using validator factory
    validator = create_harmful_content_validator(**validator_kwargs)

    # Create params dictionary for RuleConfig
    params = {
        "categories": categories or DEFAULT_HARMFUL_CATEGORIES,
        "threshold": threshold,
        "fail_if_any": fail_if_any,
    }

    # Create RuleConfig
    rule_config = RuleConfig(params=params, **rule_config_params)

    # Create rule
    return HarmfulContentRule(
        name=name,
        description=description,
        config=rule_config,
        validator=validator,
    )
