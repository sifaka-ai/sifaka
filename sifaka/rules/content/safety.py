"""
Safety-related content validation rules for Sifaka.

This module provides rules for validating text against various safety concerns,
including toxicity, bias, and harmful content.

## Overview
The safety validation rules help ensure that text meets specific safety
requirements by detecting and flagging potentially harmful, toxic, or biased
content. This is essential for content moderation, ensuring appropriate
responses, and maintaining ethical AI outputs.

## Components
- **HarmfulContentConfig**: Configuration for harmful content validation
- **HarmfulContentAnalyzer**: Analyzer for harmful content detection
- **HarmfulContentValidator**: Validator for harmful content requirements
- **HarmfulContentRule**: Rule for validating text for harmful content
- **Factory Functions**:
  - create_toxicity_rule, create_toxicity_validator
  - create_bias_rule, create_bias_validator
  - create_harmful_content_rule, create_harmful_content_validator

## Usage Examples
```python
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
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

## Error Handling
- Empty text handling through BaseValidator.handle_empty_text
- Category validation with detailed error messages
- Detailed validation results with metadata for debugging
- Fallback mechanisms for missing components (e.g., BiasDetector)
"""

import time
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
)
from sifaka.rules.content.base import CategoryAnalyzer
from sifaka.adapters.classifier import create_classifier_rule
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


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
    """
    Configuration for harmful content validation.

    This class defines the configuration options for harmful content validation,
    including categories of harmful content, detection threshold, and validation behavior.

    ## Architecture
    The class uses Pydantic for validation and immutability, with field validators
    to ensure categories are properly structured.

    ## Lifecycle
    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for threshold (0.0-1.0)
       - Category validation to ensure non-empty categories
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by HarmfulContentAnalyzer
       - Used by HarmfulContentValidator
       - Used by HarmfulContentRule

    ## Examples
    ```python
    from sifaka.rules.content.safety import HarmfulContentConfig

    # Create with default values
    config = HarmfulContentConfig()

    # Create with custom values
    config = HarmfulContentConfig(
        categories={
            "violence": ["violent", "threatening"],
            "misinformation": ["false", "misleading"]
        },
        threshold=0.3,
        fail_if_any=True
    )
    ```
    """

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
    """
    Analyzer for harmful content detection.

    This class is responsible for analyzing text for harmful content across various
    categories such as violence, hate speech, misinformation, and harassment.

    ## Architecture
    The class extends CategoryAnalyzer to leverage common category-based analysis
    functionality while specializing for harmful content detection.

    ## Lifecycle
    1. **Initialization**: Set up with harmful content configuration
       - Initialize with HarmfulContentConfig
       - Configure category analysis parameters

    2. **Analysis**: Analyze text for harmful content
       - Inherited from CategoryAnalyzer
       - Check text against configured harmful content categories
       - Return RuleResult with validation results

    ## Examples
    ```python
    from sifaka.rules.content.safety import HarmfulContentAnalyzer, HarmfulContentConfig

    # Create configuration
    config = HarmfulContentConfig(
        categories={
            "violence": ["violent", "threatening"],
            "misinformation": ["false", "misleading"]
        },
        threshold=0.3,
        fail_if_any=True
    )

    # Create analyzer
    analyzer = HarmfulContentAnalyzer(config)

    # Analyze text
    result = analyzer.analyze("This is a test.")
    print(f"Passed: {result.passed}")
    ```
    """

    def __init__(self, config: HarmfulContentConfig) -> None:
        """
        Initialize the analyzer.

        Args:
            config: Configuration for harmful content detection
        """
        super().__init__(
            categories=config.categories,
            threshold=config.threshold,
            fail_if_any=config.fail_if_any,
            higher_is_better=False,
        )


class HarmfulContentValidator(BaseValidator[str]):
    """
    Validator that checks for harmful content.

    This validator analyzes text for harmful content across various categories
    such as violence, hate speech, misinformation, and harassment.

    ## Architecture
    The HarmfulContentValidator follows a component-based architecture:
    - Inherits from BaseValidator for common validation functionality
    - Uses StateManager for state management via _state_manager
    - Delegates analysis to HarmfulContentAnalyzer
    - Uses RuleConfig for configuration
    - Implements caching for performance optimization
    - Provides detailed validation results with metadata

    ## Lifecycle
    1. **Initialization**: Set up with harmful content categories and threshold
       - Initialize with RuleConfig containing harmful content parameters
       - Create HarmfulContentConfig from params
       - Create HarmfulContentAnalyzer with config
       - Store components in state manager
       - Set metadata for tracking and debugging

    2. **Validation**: Analyze text for harmful content
       - Handle empty text through BaseValidator.handle_empty_text
       - Delegate to HarmfulContentAnalyzer for content analysis
       - Add processing time and validator type to metadata
       - Update validation statistics
       - Cache results if caching is enabled

    3. **Error Handling**: Manage validation errors
       - Type checking for input text
       - Try-except block for validation errors
       - Detailed error reporting in result metadata
       - Error logging for debugging

    ## Examples
        ```python
        from sifaka.rules.content.safety import HarmfulContentValidator, HarmfulContentConfig
        from sifaka.rules.base import RuleConfig

        # Create config
        params = {
            "categories": {
                "violence": ["violent", "threatening"],
                "misinformation": ["false", "misleading"]
            },
            "threshold": 0.3,
            "fail_if_any": True
        }
        config = RuleConfig(params=params)

        # Create validator
        validator = HarmfulContentValidator(config)

        # Validate text
        result = validator.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(self, config: RuleConfig) -> None:
        """
        Initialize the validator.

        Args:
            config: Rule configuration containing harmful content parameters
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        harmful_config = HarmfulContentConfig(**config.params)
        self._state_manager.update("config", config)
        self._state_manager.update("harmful_config", harmful_config)
        self._state_manager.update("analyzer", HarmfulContentAnalyzer(config=harmful_config))

        # Set metadata
        self._state_manager.set_metadata("validator_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def config(self) -> RuleConfig:
        """
        Get the validator configuration.

        Returns:
            The rule configuration
        """
        return self._state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate that the text does not contain harmful content.

        Args:
            text: The text to validate

        Returns:
            Validation result
        """
        start_time = time.time()

        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Analyze text for harmful content
            result = analyzer.analyze(text)

            # Add additional metadata
            result = result.with_metadata(
                validator_type=self.__class__.__name__, processing_time_ms=time.time() - start_time
            )

            # Update statistics
            self.update_statistics(result)

            # Update validation count in metadata
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Cache result if caching is enabled
            if self.config.cache_size > 0:
                cache = self._state_manager.get("cache", {})
                if len(cache) >= self.config.cache_size:
                    # Clear cache if it's full
                    cache = {}
                cache[text] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            self.record_error(e)
            logger.error(f"Harmful content validation failed: {e}")

            error_message = f"Content validation failed: {str(e)}"
            result = RuleResult(
                passed=False,
                message=error_message,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[error_message],
                suggestions=["Check input format and try again"],
                processing_time_ms=time.time() - start_time,
            )

            self.update_statistics(result)
            return result


class HarmfulContentRule(Rule[str]):
    """
    Rule that checks for harmful content in text.

    This rule analyzes text for harmful content across various categories
    such as violence, hate speech, misinformation, and harassment.

    ## Architecture
    The HarmfulContentRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Uses StateManager for state management via _state_manager
    - Delegates validation to HarmfulContentValidator
    - Uses RuleConfig for configuration
    - Creates a default validator if none is provided
    - Provides standardized validation results with metadata

    ## Lifecycle
    1. **Initialization**: Set up with harmful content categories and threshold
       - Initialize with name, description, config, and optional validator
       - Create default validator if none is provided
       - Store validator in state manager
       - Set metadata for tracking and debugging

    2. **Validation**: Delegate to validator to analyze text for harmful content
       - Inherited from Rule base class
       - Delegate to HarmfulContentValidator for content validation
       - Add rule_id to metadata for traceability
       - Return standardized RuleResult with validation results

    3. **Default Validator Creation**: Create validator from config
       - Extract parameters from rule config
       - Create HarmfulContentValidator with appropriate configuration
       - Store validator config in state for reference

    ## Examples
        ```python
        from sifaka.rules.content.safety import HarmfulContentRule, HarmfulContentValidator, HarmfulContentConfig
        from sifaka.rules.base import RuleConfig

        # Create config
        params = {
            "categories": {
                "violence": ["violent", "threatening"],
                "misinformation": ["false", "misleading"]
            },
            "threshold": 0.3,
            "fail_if_any": True
        }
        config = RuleConfig(params=params)

        # Create validator
        validator = HarmfulContentValidator(config)

        # Create rule
        rule = HarmfulContentRule(
            name="harmful_content_rule",
            description="Validates text for harmful content",
            validator=validator
        )

        # Validate text
        result = rule.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

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
            config=config
            or RuleConfig(
                name=name, description=description, rule_id=kwargs.pop("rule_id", name), **kwargs
            ),
            validator=validator,
        )

        # Store validator in state
        harmful_content_validator = validator or self._create_default_validator()
        self._state_manager.update("harmful_content_validator", harmful_content_validator)

        # Set additional metadata
        self._state_manager.set_metadata("rule_type", "HarmfulContentRule")
        self._state_manager.set_metadata("creation_time", time.time())

    def _create_default_validator(self) -> HarmfulContentValidator:
        """
        Create a default validator from config.

        Returns:
            A configured HarmfulContentValidator
        """
        # Store config in state for reference
        self._state_manager.update("validator_config", self.config)

        return HarmfulContentValidator(self.config)


def create_harmful_content_validator(
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
    fail_if_any: bool = True,
    **kwargs: Any,
) -> HarmfulContentValidator:
    """
    Create a harmful content validator.

    This factory function creates a configured HarmfulContentValidator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        categories: Dictionary of harmful content categories and their indicators
        threshold: Minimum score threshold for validation
        fail_if_any: Whether to fail if any category exceeds the threshold
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured HarmfulContentValidator

    Examples:
        ```python
        from sifaka.rules.content.safety import create_harmful_content_validator

        # Create a basic validator
        validator = create_harmful_content_validator(threshold=0.3)

        # Create a validator with custom categories
        validator = create_harmful_content_validator(
            categories={
                "violence": ["violent", "threatening"],
                "misinformation": ["false", "misleading"]
            },
            threshold=0.3,
            fail_if_any=True
        )
        ```
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

    # Create and return the validator
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

    Examples:
        ```python
        from sifaka.rules.content.safety import create_toxicity_validator

        # Create a basic validator
        validator = create_toxicity_validator(threshold=0.4)

        # Validate text
        result = validator.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """
    from sifaka.adapters.classifier import ClassifierAdapter

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
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a toxicity rule using the classifier adapter.

    This factory function creates a configured toxicity rule instance using the
    ToxicityClassifier through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for toxicity detection (0.0 to 1.0)
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured toxicity rule instance

    Examples:
        ```python
        from sifaka.rules.content.safety import create_toxicity_rule

        # Create a basic rule
        rule = create_toxicity_rule(threshold=0.4)

        # Create a rule with metadata
        rule = create_toxicity_rule(
            threshold=0.4,
            name="custom_toxicity_rule",
            description="Validates text for toxic content",
            rule_id="toxicity_validator",
            severity="warning",
            category="content",
            tags=["toxicity", "content", "validation"]
        )
        ```
    """
    # Determine rule name
    rule_name = name or rule_id or "toxicity_rule"

    # Create rule using create_classifier_rule
    return create_classifier_rule(
        classifier=ToxicityClassifier(),
        name=rule_name,
        description=description,
        threshold=threshold,
        valid_labels=["non-toxic"],
        rule_id=rule_id,
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

    Examples:
        ```python
        from sifaka.rules.content.safety import create_bias_validator

        # Create a basic validator
        validator = create_bias_validator(threshold=0.3)

        # Validate text
        result = validator.validate("This is a test.")
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```

    Note:
        Requires the BiasDetector to be properly implemented in the codebase.
    """
    # Import BiasDetector here to avoid circular imports
    try:
        from sifaka.classifiers.implementations.content.bias import BiasDetector
    except ImportError:
        logger.warning("BiasDetector not found. Using ToxicityClassifier as a fallback.")
        # Use ToxicityClassifier as a fallback
        classifier = ToxicityClassifier()
        valid_labels = ["non-toxic"]
    else:
        # Create classifier
        classifier = BiasDetector()
        valid_labels = ["unbiased"]

    from sifaka.adapters.classifier import ClassifierAdapter

    # Create adapter with classifier
    adapter = ClassifierAdapter(
        classifier=classifier, threshold=threshold, valid_labels=valid_labels, **kwargs
    )

    return adapter


def create_bias_rule(
    name: str = "bias_rule",
    description: str = "Validates text for biased content",
    threshold: float = 0.3,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> Rule:
    """
    Create a bias rule using the classifier adapter.

    This factory function creates a configured bias rule instance using the
    BiasDetector through the classifier adapter.

    Args:
        name: The name of the rule
        description: Description of the rule
        threshold: Threshold for bias detection (0.0 to 1.0)
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured bias rule instance

    Examples:
        ```python
        from sifaka.rules.content.safety import create_bias_rule

        # Create a basic rule
        rule = create_bias_rule(threshold=0.3)

        # Create a rule with metadata
        rule = create_bias_rule(
            threshold=0.3,
            name="custom_bias_rule",
            description="Validates text for biased content",
            rule_id="bias_validator",
            severity="warning",
            category="content",
            tags=["bias", "content", "validation"]
        )
        ```

    Note:
        Requires the BiasDetector to be properly implemented in the codebase.
    """
    # Import BiasDetector here to avoid circular imports
    try:
        from sifaka.classifiers.implementations.content.bias import BiasDetector
    except ImportError:
        logger.warning("BiasDetector not found. Using ToxicityClassifier as a fallback.")
        # Use ToxicityClassifier as a fallback
        classifier = ToxicityClassifier()
        valid_labels = ["non-toxic"]
    else:
        # Create classifier
        classifier = BiasDetector()
        valid_labels = ["unbiased"]

    # Determine rule name
    rule_name = name or rule_id or "bias_rule"

    return create_classifier_rule(
        classifier=classifier,
        name=rule_name,
        description=description,
        threshold=threshold,
        valid_labels=valid_labels,
        rule_id=rule_id,
        **kwargs,
    )


def create_harmful_content_rule(
    name: str = "harmful_content_rule",
    description: str = "Validates text for harmful content",
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
    fail_if_any: bool = True,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> HarmfulContentRule:
    """
    Create a harmful content rule with configuration.

    This factory function creates a configured HarmfulContentRule instance.
    It uses create_harmful_content_validator internally to create the validator.

    Args:
        name: The name of the rule
        description: Description of the rule
        categories: Dictionary of harmful content categories and their indicators
        threshold: Minimum score threshold for validation
        fail_if_any: Whether to fail if any category exceeds the threshold
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        Configured HarmfulContentRule instance

    Examples:
        ```python
        from sifaka.rules.content.safety import create_harmful_content_rule

        # Create a basic rule
        rule = create_harmful_content_rule(threshold=0.3)

        # Create a rule with custom categories and metadata
        rule = create_harmful_content_rule(
            categories={
                "violence": ["violent", "threatening"],
                "misinformation": ["false", "misleading"]
            },
            threshold=0.3,
            fail_if_any=True,
            name="custom_harmful_content_rule",
            description="Validates text for specific harmful content",
            rule_id="harmful_content_validator",
            severity="warning",
            category="content",
            tags=["harmful", "content", "validation"]
        )
        ```
    """
    # Create validator using validator factory
    validator = create_harmful_content_validator(
        categories=categories,
        threshold=threshold,
        fail_if_any=fail_if_any,
    )

    # Create params dictionary for RuleConfig
    params = {
        "categories": categories or DEFAULT_HARMFUL_CATEGORIES,
        "threshold": threshold,
        "fail_if_any": fail_if_any,
    }

    # Determine rule name
    rule_name = name or rule_id or "harmful_content_rule"

    # Create RuleConfig
    rule_config = RuleConfig(
        name=rule_name,
        description=description,
        rule_id=rule_id or rule_name,
        params=params,
        **kwargs,
    )

    # Create and return the rule
    return HarmfulContentRule(
        name=rule_name,
        description=description,
        config=rule_config,
        validator=validator,
    )
