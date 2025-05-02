"""
Prohibited content validation rules for Sifaka.

This module provides validators and rules for checking text against prohibited content.

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- All rule-specific configuration is stored in RuleConfig.params
- Factory functions handle configuration extraction
- Validator factory functions create standalone validators
- Rule factory functions use validator factory functions internally

## Usage Example

```python
from sifaka.rules.content.prohibited import create_prohibited_content_rule

# Create a prohibited content rule
rule = create_prohibited_content_rule(
    terms=["inappropriate", "offensive", "vulgar"],
    threshold=0.5
)

# Validate text
result = rule.validate("This is a test.")
```
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from sifaka.classifiers.profanity import ProfanityClassifier
from sifaka.classifiers.base import ClassifierConfig
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
)


__all__ = [
    # Config classes
    "ProhibitedContentConfig",
    # Analyzer classes
    "ProhibitedContentAnalyzer",
    # Validator classes
    "DefaultProhibitedContentValidator",
    # Rule classes
    "ProhibitedContentRule",
    # Factory functions
    "create_prohibited_content_validator",
    "create_prohibited_content_rule",
]


class ProhibitedContentConfig(BaseModel):
    """Configuration for prohibited content validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terms: List[str] = Field(
        default_factory=list,
        description="List of prohibited terms to check for",
        min_length=0,
        json_schema_extra={"examples": ["inappropriate", "offensive", "vulgar"]},
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for prohibited content detection",
        json_schema_extra={"examples": [0.5, 0.7]},
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether to perform case-sensitive matching",
    )
    cache_size: int = Field(
        default=100,
        ge=1,
        description="Size of the validation cache",
    )
    priority: int = Field(
        default=1,
        ge=0,
        description="Priority of the rule",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost of running the rule",
    )


class ProhibitedContentAnalyzer:
    """Analyzer for prohibited content detection."""

    def __init__(self, config: ProhibitedContentConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the analyzer
        """
        self._config = config
        self._terms = config.terms
        self._threshold = config.threshold
        self._case_sensitive = config.case_sensitive

        # Create the classifier with custom words
        self._classifier = ProfanityClassifier(
            config=ClassifierConfig(
                labels=["clean", "profane", "unknown"], params={"custom_words": self._terms}
            )
        )

    def analyze(self, text: str) -> RuleResult:
        """Analyze text for prohibited content.

        Args:
            text: The text to analyze

        Returns:
            RuleResult: The result of the analysis
        """
        # Use the classifier to detect prohibited content
        result = self._classifier.classify(text)

        # Determine if the text passes validation
        is_valid = result.label == "clean"
        confidence = result.confidence

        return RuleResult(
            passed=is_valid,
            message="No prohibited content detected" if is_valid else "Prohibited content detected",
            metadata={
                "confidence": confidence,
                "label": result.label,
                "threshold": self._threshold,
                "classifier_metadata": result.metadata,
            },
        )

    def can_analyze(self, text: str) -> bool:
        """Check if this analyzer can analyze the given text."""
        return isinstance(text, str)


class DefaultProhibitedContentValidator(BaseValidator[str]):
    """Default validator for prohibited content."""

    def __init__(self, config: ProhibitedContentConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__()
        self._config = config
        self._analyzer = ProhibitedContentAnalyzer(config)

    @property
    def config(self) -> ProhibitedContentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **_: Any) -> RuleResult:
        """Validate the given text for prohibited content.

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

        # Delegate to analyzer
        return self._analyzer.analyze(text)


class ProhibitedContentRule(
    Rule[str, RuleResult, DefaultProhibitedContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule for validating prohibited content."""

    def __init__(
        self,
        name: str = "prohibited_content_rule",
        description: str = "Validates text for prohibited content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultProhibitedContentValidator] = None,
    ) -> None:
        """Initialize with configuration.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
        """
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
        )

    def _create_default_validator(self) -> DefaultProhibitedContentValidator:
        """Create a default validator from config."""
        # Extract prohibited content specific params
        params = self.config.params
        config = ProhibitedContentConfig(
            terms=params.get("terms", []),
            threshold=params.get("threshold", 0.5),
            case_sensitive=params.get("case_sensitive", False),
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )
        return DefaultProhibitedContentValidator(config)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate the given text for prohibited content.

        Args:
            text: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult: The result of the validation
        """
        # Delegate to validator
        result = self._validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._name)


def create_prohibited_content_validator(
    terms: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs: Any,
) -> DefaultProhibitedContentValidator:
    """Create a prohibited content validator.

    Args:
        terms: List of prohibited terms to check for
        threshold: Threshold for prohibited content detection
        case_sensitive: Whether to perform case-sensitive matching
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultProhibitedContentValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if terms is not None:
        config_params["terms"] = terms
    if threshold is not None:
        config_params["threshold"] = threshold
    if case_sensitive is not None:
        config_params["case_sensitive"] = case_sensitive

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = ProhibitedContentConfig(**config_params)

    # Create validator
    return DefaultProhibitedContentValidator(config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    terms: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    case_sensitive: Optional[bool] = None,
    **kwargs: Any,
) -> ProhibitedContentRule:
    """Create a prohibited content rule.

    Args:
        name: The name of the rule
        description: Description of the rule
        terms: List of prohibited terms to check for
        threshold: Threshold for prohibited content detection
        case_sensitive: Whether to perform case-sensitive matching
        **kwargs: Additional keyword arguments for the rule

    Returns:
        ProhibitedContentRule: The created rule
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create validator using the validator factory
    validator = create_prohibited_content_validator(
        terms=terms,
        threshold=threshold,
        case_sensitive=case_sensitive,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Create params dictionary for RuleConfig
    params = {}
    if terms is not None:
        params["terms"] = terms
    if threshold is not None:
        params["threshold"] = threshold
    if case_sensitive is not None:
        params["case_sensitive"] = case_sensitive

    # Create RuleConfig
    config = RuleConfig(params=params, **rule_config_params)

    # Create rule
    return ProhibitedContentRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )
