"""
Safety-related content validation rules for Sifaka.

This module provides rules for validating text against various safety concerns,
including toxicity, bias, and harmful content.

Usage Example:
    from sifaka.rules.content.safety import create_toxicity_rule, create_bias_rule, create_harmful_content_rule

    # Create a toxicity rule
    toxicity_rule = create_toxicity_rule(
        threshold=0.4,
        indicators=["offensive", "vulgar", "profanity"]
    )

    # Create a bias rule
    bias_rule = create_bias_rule(
        threshold=0.3,
        categories={
            "gender": ["sexist", "gender bias"],
            "racial": ["racist", "racial bias"]
        }
    )

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

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.content.base import ContentAnalyzer, ContentValidator, DefaultContentAnalyzer


__all__ = [
    # Data classes
    "ToxicityIndicators",
    "BiasCategories",
    "HarmfulCategories",
    # Validator classes
    "ToxicityValidator",
    "BiasValidator",
    "HarmfulContentValidator",
    "DefaultToxicityValidator",
    "DefaultBiasValidator",
    "DefaultHarmfulContentValidator",
    # Rule classes
    "ToxicityRule",
    "BiasRule",
    "HarmfulContentRule",
    # Factory functions
    "create_toxicity_validator",
    "create_toxicity_rule",
    "create_bias_validator",
    "create_bias_rule",
    "create_harmful_content_validator",
    "create_harmful_content_rule",
]


@dataclass(frozen=True)
class ToxicityIndicators:
    """Immutable container for toxicity indicators configuration."""

    indicators: frozenset[str]
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.indicators:
            raise ConfigurationError("Toxicity indicators list cannot be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigurationError("Threshold must be between 0.0 and 1.0")

    def with_indicators(self, indicators: List[str]) -> "ToxicityIndicators":
        """Create new instance with updated indicators."""
        return ToxicityIndicators(indicators=frozenset(indicators), threshold=self.threshold)

    def with_threshold(self, threshold: float) -> "ToxicityIndicators":
        """Create new instance with updated threshold."""
        return ToxicityIndicators(indicators=self.indicators, threshold=threshold)


@dataclass(frozen=True)
class BiasCategories:
    """Immutable container for bias categories configuration."""

    categories: Dict[str, frozenset[str]]
    threshold: float = 0.3

    def __post_init__(self) -> None:
        if not self.categories:
            raise ConfigurationError("Bias categories cannot be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigurationError("Threshold must be between 0.0 and 1.0")

        # Convert to frozenset if not already
        categories_copy = {}
        for category, indicators in self.categories.items():
            if not indicators:
                raise ConfigurationError(f"Category {category} must have at least one indicator")
            if not isinstance(indicators, frozenset):
                categories_copy[category] = frozenset(indicators)
            else:
                categories_copy[category] = indicators

        if categories_copy:
            object.__setattr__(self, "categories", categories_copy)

    def with_categories(self, categories: Dict[str, List[str]]) -> "BiasCategories":
        """Create new instance with updated categories."""
        categories_copy = {}
        for category, indicators in categories.items():
            categories_copy[category] = frozenset(indicators)

        return BiasCategories(categories=categories_copy, threshold=self.threshold)

    def with_threshold(self, threshold: float) -> "BiasCategories":
        """Create new instance with updated threshold."""
        return BiasCategories(categories=self.categories, threshold=threshold)


@dataclass(frozen=True)
class HarmfulCategories:
    """Immutable container for harmful content categories configuration."""

    categories: Dict[str, frozenset[str]]

    def __post_init__(self) -> None:
        if not self.categories:
            raise ConfigurationError("Harmful content categories cannot be empty")

        # Convert to frozenset if not already
        categories_copy = {}
        for category, indicators in self.categories.items():
            if not indicators:
                raise ConfigurationError(f"Category {category} must have at least one indicator")
            if not isinstance(indicators, frozenset):
                categories_copy[category] = frozenset(indicators)
            else:
                categories_copy[category] = indicators

        if categories_copy:
            object.__setattr__(self, "categories", categories_copy)

    def with_categories(self, categories: Dict[str, List[str]]) -> "HarmfulCategories":
        """Create new instance with updated categories."""
        categories_copy = {}
        for category, indicators in categories.items():
            categories_copy[category] = frozenset(indicators)

        return HarmfulCategories(categories=categories_copy)


class ToxicityValidator(ContentValidator):
    """Validator that checks for toxic content."""

    def __init__(
        self,
        indicators: ToxicityIndicators,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._indicators: Final[ToxicityIndicators] = indicators

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output does not contain toxic content."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            output_lower = output.lower()
            toxic_indicators = [
                indicator
                for indicator in self._indicators.indicators
                if indicator.lower() in output_lower
            ]

            toxicity_score = len(toxic_indicators) / len(self._indicators.indicators)

            if toxicity_score > self._indicators.threshold:
                return RuleResult(
                    passed=False,
                    message=f"Output contains toxic content (score: {toxicity_score:.2f})",
                    metadata={
                        "toxicity_score": toxicity_score,
                        "toxic_indicators": toxic_indicators,
                        "analysis": self._analyzer.analyze(output),
                    },
                )

            return RuleResult(
                passed=True,
                message="No toxic content detected",
                metadata={
                    "toxicity_score": toxicity_score,
                    "analysis": self._analyzer.analyze(output),
                },
            )

        except Exception as e:
            raise ValidationError(f"Toxicity validation failed: {str(e)}") from e


class BiasValidator(ContentValidator):
    """Validator that checks for biased content."""

    def __init__(
        self,
        categories: BiasCategories,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._categories: Final[BiasCategories] = categories

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output does not contain biased content."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            output_lower = output.lower()
            bias_scores: Dict[str, float] = {}
            detected_biases: Dict[str, List[str]] = {}

            for category, indicators in self._categories.categories.items():
                found_indicators = [
                    indicator for indicator in indicators if indicator.lower() in output_lower
                ]
                if found_indicators:
                    bias_scores[category] = len(found_indicators) / len(indicators)
                    detected_biases[category] = found_indicators

            analysis = self._analyzer.analyze(output)

            if not bias_scores:
                return RuleResult(
                    passed=True,
                    message="No biased content detected",
                    metadata={
                        "bias_scores": {},
                        "detected_biases": {},
                        "analysis": analysis,
                    },
                )

            overall_bias_score = sum(bias_scores.values()) / len(self._categories.categories)

            if overall_bias_score > self._categories.threshold:
                return RuleResult(
                    passed=False,
                    message=f"Output contains biased content (score: {overall_bias_score:.2f})",
                    metadata={
                        "bias_scores": bias_scores,
                        "detected_biases": detected_biases,
                        "overall_score": overall_bias_score,
                        "analysis": analysis,
                    },
                )

            return RuleResult(
                passed=True,
                message="No significant bias detected",
                metadata={
                    "bias_scores": bias_scores,
                    "detected_biases": detected_biases,
                    "overall_score": overall_bias_score,
                    "analysis": analysis,
                },
            )

        except Exception as e:
            raise ValidationError(f"Bias validation failed: {str(e)}") from e


class HarmfulContentValidator(ContentValidator):
    """Validator that checks for harmful content."""

    def __init__(
        self,
        categories: HarmfulCategories,
        analyzer: Optional[ContentAnalyzer] = None,
    ) -> None:
        """Initialize the validator."""
        super().__init__(analyzer or DefaultContentAnalyzer())
        self._categories: Final[HarmfulCategories] = categories

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate that the output does not contain harmful content."""
        try:
            if not isinstance(output, str):
                raise TypeError("Output must be a string")

            output_lower = output.lower()
            harmful_content: Dict[str, List[str]] = {}

            for category, indicators in self._categories.categories.items():
                found_indicators = [
                    indicator for indicator in indicators if indicator.lower() in output_lower
                ]
                if found_indicators:
                    harmful_content[category] = found_indicators

            analysis = self._analyzer.analyze(output)

            if harmful_content:
                return RuleResult(
                    passed=False,
                    message="Output contains harmful content",
                    metadata={
                        "harmful_content": harmful_content,
                        "categories_found": list(harmful_content.keys()),
                        "analysis": analysis,
                    },
                )

            return RuleResult(
                passed=True,
                message="No harmful content detected",
                metadata={
                    "harmful_content": {},
                    "categories_found": [],
                    "analysis": analysis,
                },
            )

        except Exception as e:
            raise ValidationError(f"Harmful content validation failed: {str(e)}") from e


class DefaultToxicityValidator(BaseValidator[str]):
    """Default implementation of toxicity validation."""

    # Default toxicity indicators
    DEFAULT_TOXICITY_INDICATORS: Final[List[str]] = [
        "hate",
        "offensive",
        "vulgar",
        "profanity",
        "obscene",
        "racist",
        "sexist",
        "discriminatory",
        "threatening",
        "harassing",
    ]

    DEFAULT_THRESHOLD: Final[float] = 0.5

    def __init__(self, config: RuleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for toxicity."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        threshold = self.config.params.get("threshold", self.DEFAULT_THRESHOLD)
        indicators = self.config.params.get("indicators", self.DEFAULT_TOXICITY_INDICATORS)

        text_lower = text.lower()
        toxic_indicators = [
            indicator for indicator in indicators if indicator.lower() in text_lower
        ]

        toxicity_score = len(toxic_indicators) / len(indicators) if indicators else 0

        if toxicity_score > threshold:
            return RuleResult(
                passed=False,
                message=f"Text contains toxic content (score: {toxicity_score:.2f})",
                metadata={
                    "toxicity_score": toxicity_score,
                    "toxic_indicators": toxic_indicators,
                    "threshold": threshold,
                },
            )

        return RuleResult(
            passed=True,
            message="No toxic content detected",
            metadata={
                "toxicity_score": toxicity_score,
                "threshold": threshold,
            },
        )


class DefaultBiasValidator(BaseValidator[str]):
    """Default implementation of bias validation."""

    # Default bias categories
    DEFAULT_BIAS_CATEGORIES: Final[Dict[str, List[str]]] = {
        "gender": [
            "sexist",
            "misogynistic",
            "patriarchal",
            "gender bias",
            "gender stereotype",
        ],
        "racial": [
            "racist",
            "racial bias",
            "ethnic stereotype",
            "racial prejudice",
            "discriminatory",
        ],
        "age": [
            "ageist",
            "age discrimination",
            "age bias",
            "age stereotype",
            "generational bias",
        ],
        "cultural": [
            "cultural bias",
            "xenophobic",
            "ethnocentric",
            "cultural stereotype",
            "cultural prejudice",
        ],
    }

    DEFAULT_THRESHOLD: Final[float] = 0.3

    def __init__(self, config: RuleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for bias."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        threshold = self.config.params.get("threshold", self.DEFAULT_THRESHOLD)
        categories = self.config.params.get("categories", self.DEFAULT_BIAS_CATEGORIES)

        text_lower = text.lower()
        bias_scores: Dict[str, float] = {}
        detected_biases: Dict[str, List[str]] = {}

        for category, indicators in categories.items():
            found_indicators = [
                indicator for indicator in indicators if indicator.lower() in text_lower
            ]
            if found_indicators:
                bias_scores[category] = len(found_indicators) / len(indicators)
                detected_biases[category] = found_indicators

        if not bias_scores:
            return RuleResult(
                passed=True,
                message="No biased content detected",
                metadata={
                    "bias_scores": {},
                    "detected_biases": {},
                    "threshold": threshold,
                },
            )

        overall_bias_score = sum(bias_scores.values()) / len(categories)

        if overall_bias_score > threshold:
            return RuleResult(
                passed=False,
                message=f"Text contains biased content (score: {overall_bias_score:.2f})",
                metadata={
                    "bias_scores": bias_scores,
                    "detected_biases": detected_biases,
                    "overall_score": overall_bias_score,
                    "threshold": threshold,
                },
            )

        return RuleResult(
            passed=True,
            message="No significant bias detected",
            metadata={
                "bias_scores": bias_scores,
                "detected_biases": detected_biases,
                "overall_score": overall_bias_score,
                "threshold": threshold,
            },
        )


class DefaultHarmfulContentValidator(BaseValidator[str]):
    """Default implementation of harmful content validation."""

    # Default harmful content categories
    DEFAULT_HARMFUL_CATEGORIES: Final[Dict[str, List[str]]] = {
        "violence": [
            "violent",
            "brutal",
            "aggressive",
            "threatening",
            "dangerous",
        ],
        "self_harm": [
            "suicide",
            "self-harm",
            "self-injury",
            "self-destructive",
            "harmful behavior",
        ],
        "exploitation": [
            "exploitative",
            "manipulative",
            "coercive",
            "predatory",
            "abusive",
        ],
        "misinformation": [
            "false",
            "misleading",
            "deceptive",
            "propaganda",
            "disinformation",
        ],
    }

    def __init__(self, config: RuleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> RuleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for harmful content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Get configuration from params for consistency
        categories = self.config.params.get("categories", self.DEFAULT_HARMFUL_CATEGORIES)

        text_lower = text.lower()
        harmful_content: Dict[str, List[str]] = {}

        for category, indicators in categories.items():
            found_indicators = [
                indicator for indicator in indicators if indicator.lower() in text_lower
            ]
            if found_indicators:
                harmful_content[category] = found_indicators

        if harmful_content:
            return RuleResult(
                passed=False,
                message="Text contains harmful content",
                metadata={
                    "harmful_content": harmful_content,
                    "categories_found": list(harmful_content.keys()),
                },
            )

        return RuleResult(
            passed=True,
            message="No harmful content detected",
            metadata={
                "harmful_content": {},
                "categories_found": [],
            },
        )


class ToxicityRule(Rule[str, RuleResult, DefaultToxicityValidator, RuleResultHandler[RuleResult]]):
    """Rule that checks for toxic content."""

    def __init__(
        self,
        name: str = "toxicity_rule",
        description: str = "Validates text for toxic content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultToxicityValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the toxicity rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Create default config if not provided and no validator is provided
        if config is None and validator is None:
            config = RuleConfig(
                params={
                    "threshold": DefaultToxicityValidator.DEFAULT_THRESHOLD,
                    "indicators": DefaultToxicityValidator.DEFAULT_TOXICITY_INDICATORS,
                    "cache_size": 100,
                    "priority": 1,
                    "cost": 1.0,
                }
            )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultToxicityValidator:
        """Create a default validator from config."""
        return DefaultToxicityValidator(self.config)


class BiasRule(Rule[str, RuleResult, DefaultBiasValidator, RuleResultHandler[RuleResult]]):
    """Rule that checks for biased content."""

    def __init__(
        self,
        name: str = "bias_rule",
        description: str = "Validates text for biased content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultBiasValidator] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the bias rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
            **kwargs: Additional keyword arguments for the rule
        """
        # Create default config if not provided and no validator is provided
        if config is None and validator is None:
            config = RuleConfig(
                params={
                    "threshold": DefaultBiasValidator.DEFAULT_THRESHOLD,
                    "categories": DefaultBiasValidator.DEFAULT_BIAS_CATEGORIES,
                    "cache_size": 100,
                    "priority": 1,
                    "cost": 1.0,
                }
            )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultBiasValidator:
        """Create a default validator from config."""
        return DefaultBiasValidator(self.config)


class HarmfulContentRule(
    Rule[str, RuleResult, DefaultHarmfulContentValidator, RuleResultHandler[RuleResult]]
):
    """Rule that checks for harmful content."""

    def __init__(
        self,
        name: str = "harmful_content_rule",
        description: str = "Validates text for harmful content",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultHarmfulContentValidator] = None,
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
        # Create default config if not provided and no validator is provided
        if config is None and validator is None:
            config = RuleConfig(
                params={
                    "categories": DefaultHarmfulContentValidator.DEFAULT_HARMFUL_CATEGORIES,
                    "cache_size": 100,
                    "priority": 1,
                    "cost": 1.0,
                }
            )

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            result_handler=None,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultHarmfulContentValidator:
        """Create a default validator from config."""
        return DefaultHarmfulContentValidator(self.config)


def create_toxicity_validator(
    threshold: float = None,
    indicators: List[str] = None,
    **kwargs,
) -> BaseValidator[str]:
    """
    Create a toxicity validator with the specified configuration.

    This factory function creates a configured toxicity validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        threshold: Threshold for toxicity detection (0.0 to 1.0)
        indicators: List of toxicity indicators to check for
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured toxicity validator
    """
    # Set default values if not provided
    if threshold is None:
        threshold = DefaultToxicityValidator.DEFAULT_THRESHOLD
    if indicators is None:
        indicators = DefaultToxicityValidator.DEFAULT_TOXICITY_INDICATORS

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config dictionary
    config_dict = {
        "threshold": threshold,
        "indicators": indicators,
        **rule_config_params,
    }

    # Create RuleConfig
    rule_config = RuleConfig(params=config_dict)

    # Return configured validator
    return DefaultToxicityValidator(rule_config)


def create_toxicity_rule(
    name: str = "toxicity_rule",
    description: str = "Validates text for toxic content",
    threshold: float = None,
    indicators: List[str] = None,
    **kwargs,
) -> ToxicityRule:
    """
    Create a toxicity rule with configuration.

    This factory function creates a configured ToxicityRule instance.
    It uses create_toxicity_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for toxicity detection (0.0 to 1.0)
        indicators: List of toxicity indicators to check for
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured ToxicityRule instance
    """
    # Create validator using the validator factory
    validator = create_toxicity_validator(
        threshold=threshold,
        indicators=indicators,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return ToxicityRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_bias_validator(
    threshold: float = None,
    categories: Dict[str, List[str]] = None,
    **kwargs,
) -> BaseValidator[str]:
    """
    Create a bias validator with the specified configuration.

    This factory function creates a configured bias validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        threshold: Threshold for bias detection (0.0 to 1.0)
        categories: Dictionary mapping bias categories to indicator terms
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured bias validator
    """
    # Set default values if not provided
    if threshold is None:
        threshold = DefaultBiasValidator.DEFAULT_THRESHOLD
    if categories is None:
        categories = DefaultBiasValidator.DEFAULT_BIAS_CATEGORIES

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config dictionary
    config_dict = {
        "threshold": threshold,
        "categories": categories,
        **rule_config_params,
    }

    # Create RuleConfig
    rule_config = RuleConfig(params=config_dict)

    # Return configured validator
    return DefaultBiasValidator(rule_config)


def create_bias_rule(
    name: str = "bias_rule",
    description: str = "Validates text for biased content",
    threshold: float = None,
    categories: Dict[str, List[str]] = None,
    **kwargs,
) -> BiasRule:
    """
    Create a bias rule with configuration.

    This factory function creates a configured BiasRule instance.
    It uses create_bias_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for bias detection (0.0 to 1.0)
        categories: Dictionary mapping bias categories to indicator terms
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured BiasRule instance
    """
    # Create validator using the validator factory
    validator = create_bias_validator(
        threshold=threshold,
        categories=categories,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return BiasRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )


def create_harmful_content_validator(
    categories: Dict[str, List[str]] = None,
    **kwargs,
) -> BaseValidator[str]:
    """
    Create a harmful content validator with the specified configuration.

    This factory function creates a configured harmful content validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        categories: Dictionary mapping harmful content categories to indicator terms
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured harmful content validator
    """
    # Set default values if not provided
    if categories is None:
        categories = DefaultHarmfulContentValidator.DEFAULT_HARMFUL_CATEGORIES

    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config dictionary
    config_dict = {
        "categories": categories,
        **rule_config_params,
    }

    # Create RuleConfig
    rule_config = RuleConfig(params=config_dict)

    # Return configured validator
    return DefaultHarmfulContentValidator(rule_config)


def create_harmful_content_rule(
    name: str = "harmful_content_rule",
    description: str = "Validates text for harmful content",
    categories: Dict[str, List[str]] = None,
    **kwargs,
) -> HarmfulContentRule:
    """
    Create a harmful content rule with configuration.

    This factory function creates a configured HarmfulContentRule instance.
    It uses create_harmful_content_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        categories: Dictionary mapping harmful content categories to indicator terms
        **kwargs: Additional keyword arguments for the rule

    Returns:
        Configured HarmfulContentRule instance
    """
    # Create validator using the validator factory
    validator = create_harmful_content_validator(
        categories=categories,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return HarmfulContentRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
