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
result = toxicity_rule.validate("This is a test.") if toxicity_rule else ""
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

## Error Handling
- Empty text handling through BaseValidator.handle_empty_text
- Category validation with detailed error messages
- Detailed validation results with metadata for debugging
- Fallback mechanisms for missing components (e.g., BiasDetector)
"""

import time
from typing import Dict, List, Optional, Any, cast, TypeVar

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.classifiers.implementations.content.toxicity import ToxicityClassifier
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
)
from sifaka.rules.content.base import CategoryAnalyzer
from sifaka.core.results import ClassificationResult
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
    result = analyzer.analyze("This is a test.") if analyzer else ""
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
    Validator for harmful content requirements.

    This class is responsible for validating text against harmful content requirements,
    ensuring that the text does not contain harmful content above specified thresholds.

    ## Architecture
    The class extends BaseValidator to leverage standard validation functionality
    while specializing for harmful content validation.

    ## Lifecycle
    1. **Initialization**: Set up with configuration
       - Initialize with RuleConfig containing params
       - Extract parameters from config
       - Set up analyzer and threshold

    2. **Validation**: Validate text against harmful content requirements
       - Check if text is empty
       - Run analysis on text
       - Compare results against thresholds
       - Return RuleResult with validation results

    ## Examples
    ```python
    from sifaka.rules.content.safety import HarmfulContentValidator, HarmfulContentConfig
    from sifaka.rules.base import RuleConfig

    # Create configuration
    config = RuleConfig(
        name="harmful_content_validator",
        description="Validates text for harmful content",
        params={
            "categories": {
                "violence": ["violent", "threatening"],
                "misinformation": ["false", "misleading"]
            },
            "threshold": 0.3,
            "fail_if_any": True
        }
    )

    # Create validator
    validator = HarmfulContentValidator(config)

    # Validate text
    result = validator.validate("This is a test.") if validator else ""
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
    ```
    """

    def __init__(self, config: RuleConfig) -> None:
        """
        Initialize the validator.

        Args:
            config: Configuration for the validator
        """
        super().__init__()

        # Store config
        self._config = config

        # Extract parameters from config
        params = config.params if hasattr(config, "params") else {}
        # Get categories or default
        self.categories = params.get("categories", DEFAULT_HARMFUL_CATEGORIES)
        # Get threshold or default
        self.threshold = params.get("threshold", 0.0)
        # Get fail_if_any or default
        self.fail_if_any = params.get("fail_if_any", True)

        # Create harmful content config
        self.harmful_content_config = HarmfulContentConfig(
            categories=self.categories,
            threshold=self.threshold,
            fail_if_any=self.fail_if_any,
        )

        # Create analyzer
        self.analyzer = HarmfulContentAnalyzer(self.harmful_content_config)

    @property
    def config(self) -> RuleConfig:
        """
        Get the validator configuration.

        Returns:
            The rule configuration
        """
        # Get the config from state manager
        config_obj: Any = self._state_manager.get("config")

        # Create a default config
        default_config: RuleConfig = RuleConfig(
            name="harmful_content_validator", description="Validates text for harmful content"
        )

        # Return the config if it's a RuleConfig, otherwise return the default
        if config_obj is not None and isinstance(config_obj, RuleConfig):
            result: RuleConfig = config_obj
            return result
        else:
            return default_config

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
        if self is not None:
            empty_result = self.handle_empty_text(text)
            if empty_result:
                # Ensure we're returning a RuleResult
                if isinstance(empty_result, RuleResult):
                    return empty_result
                else:
                    # Convert to RuleResult if it's not already one
                    return RuleResult(
                        passed=empty_result.passed,
                        message=empty_result.message,
                        metadata=empty_result.metadata,
                        score=empty_result.score,
                        issues=empty_result.issues,
                        suggestions=empty_result.suggestions,
                        processing_time_ms=empty_result.processing_time_ms,
                    )

        try:
            if not isinstance(text, str):
                raise TypeError("Input must be a string")

            # Get analyzer from state
            analyzer = self._state_manager.get("analyzer")

            # Analyze text for harmful content
            if analyzer is not None:
                result = analyzer.analyze(text)

                # Add additional metadata
                if result is not None:
                    elapsed_ms = (time.time() - start_time) * 1000
                    result = result.with_metadata(
                        validator_type=self.__class__.__name__, processing_time_ms=elapsed_ms
                    )

                # Update statistics
                if self is not None:
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

                # Ensure we're returning a RuleResult
                if isinstance(result, RuleResult):
                    return result
                else:
                    # Convert to RuleResult if it's not already one
                    return RuleResult(
                        passed=result.passed,
                        message=result.message,
                        metadata=result.metadata,
                        score=result.score,
                        issues=result.issues,
                        suggestions=result.suggestions,
                        processing_time_ms=result.processing_time_ms,
                    )
            else:
                raise ValueError("Analyzer not found in state manager")

        except Exception as e:
            if self is not None:
                self.record_error(e)
            if logger is not None:
                logger.error(f"Harmful content validation failed: {e}")

            error_message = f"Content validation failed: {str(e)}"
            elapsed_ms = (time.time() - start_time) * 1000
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
                processing_time_ms=elapsed_ms,
            )

            if self is not None:
                self.update_statistics(result)
            # Ensure we're returning a RuleResult
            return result  # type: ignore[no-any-return]


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
        result = rule.validate("This is a test.") if rule else ""
        print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
        ```
    """

    def __init__(
        self,
        name: str = "harmful_content_rule",
        description: str = "Validates text for harmful content",
        config: Optional[RuleConfig] = None,
        validator: Optional[HarmfulContentValidator] = None,
        **kwargs: Any,
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
        rule_id = kwargs.pop("rule_id", name) if kwargs else name

        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name,
                description=description,
                rule_id=rule_id,
                **kwargs,
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
        Create the default validator.

        This method creates a default validator with the rule's configuration.
        It's used when no validator is provided during rule initialization.

        Returns:
            A configured HarmfulContentValidator
        """
        # Store config in state for reference
        self._state_manager.update("validator_config", self.config)

        # Ensure we're using a RuleConfig
        rule_config = cast(RuleConfig, self.config)

        return HarmfulContentValidator(config=rule_config)


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

    # Create RuleConfig with required name and description
    config = RuleConfig(
        name="harmful_content_validator",
        description="Validates text for harmful content",
        params=params,
    )

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
    from sifaka.adapters.classifier.adapter import Classifier

    # Create classifier
    classifier = ToxicityClassifier()

    # Create a wrapper class that implements the Classifier protocol correctly
    class ToxicityClassifierWrapper:
        def __init__(self, toxicity_classifier: ToxicityClassifier):
            self.toxicity_classifier = toxicity_classifier

        @property
        def name(self) -> str:
            return str(self.toxicity_classifier.name)

        @property
        def description(self) -> str:
            return str(self.toxicity_classifier.description)

        @property
        def config(self) -> Any:
            return self.toxicity_classifier.config

        def classify(self, text: str) -> ClassificationResult[Any, Any]:
            # Convert the result to the expected type
            result = self.toxicity_classifier.classify(text)
            # Create a new ClassificationResult with the same values
            return ClassificationResult(
                label=result.label,
                confidence=result.confidence,
                passed=result.passed,
                message=result.message,
                metadata=result.metadata,
                score=result.score,
                issues=result.issues,
                suggestions=result.suggestions,
                processing_time_ms=result.processing_time_ms,
                timestamp=result.timestamp,
            )

        def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, Any]]:
            results = self.toxicity_classifier.batch_classify(texts)
            # Convert each result to the expected type
            return [
                ClassificationResult(
                    label=r.label,
                    confidence=r.confidence,
                    passed=r.passed,
                    message=r.message,
                    metadata=r.metadata,
                    score=r.score,
                    issues=r.issues,
                    suggestions=r.suggestions,
                    processing_time_ms=r.processing_time_ms,
                    timestamp=r.timestamp,
                )
                for r in results
            ]

    # Wrap the toxicity classifier
    wrapped_classifier = ToxicityClassifierWrapper(classifier)

    # Cast the wrapped classifier to the Classifier protocol type
    classifier_protocol: Classifier = cast(Classifier, wrapped_classifier)

    adapter = ClassifierAdapter(
        classifier=classifier_protocol,
        threshold=threshold,
        valid_labels=["non-toxic"],
        **kwargs,
    )

    # Return adapter as BaseValidator[str]
    return cast(BaseValidator[str], adapter)


def create_toxicity_rule(
    name: str = "toxicity_rule",
    description: str = "Validates text for toxic content",
    threshold: float = 0.5,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> Rule[str]:
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
    # Create the validator
    validator = create_toxicity_validator(threshold=threshold, **kwargs)

    # Create a new RuleConfig for the rule
    config = RuleConfig(
        name=name or "toxicity_rule",
        description=description or "Validates text for toxic content",
        rule_id=rule_id or name or "toxicity_rule",
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["severity", "category", "tags", "priority", "cache_size", "cost"]
        },
    )

    # Create a concrete rule class that extends Rule
    class ToxicityRule(Rule[str]):
        def _create_default_validator(self) -> BaseValidator[str]:
            return validator

    # Create and return the rule
    return ToxicityRule(
        name=name or "toxicity_rule",
        description=description or "Validates text for toxic content",
        config=config,
        validator=validator,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["severity", "category", "tags", "priority", "cache_size", "cost"]
        },
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
    from sifaka.adapters.classifier import ClassifierAdapter
    from sifaka.adapters.classifier.adapter import Classifier

    # Import BiasDetector here to avoid circular imports
    # Define a variable to hold the classifier
    classifier: Any

    try:
        from sifaka.classifiers.implementations.content.bias import BiasDetector

        # Create classifier
        bias_classifier = BiasDetector()
        valid_labels = ["unbiased"]
        # Use the bias classifier as a generic classifier type
        classifier = bias_classifier
    except ImportError:
        if logger is not None:
            logger.warning("BiasDetector not found. Using ToxicityClassifier as a fallback.")
        # Use ToxicityClassifier as a fallback
        toxicity_classifier = ToxicityClassifier()
        valid_labels = ["non-toxic"]
        # Use the toxicity classifier as a generic classifier type
        classifier = toxicity_classifier

    # Create a wrapper class that implements the Classifier protocol correctly
    class ClassifierWrapper:
        def __init__(self, classifier_impl: Any):
            self.classifier_impl = classifier_impl

        @property
        def name(self) -> str:
            return str(self.classifier_impl.name)

        @property
        def description(self) -> str:
            return str(self.classifier_impl.description)

        @property
        def config(self) -> Any:
            return self.classifier_impl.config

        def classify(self, text: str) -> ClassificationResult[Any, Any]:
            # Convert the result to the expected type
            result = self.classifier_impl.classify(text)
            # Create a new ClassificationResult with the same values
            return ClassificationResult(
                label=result.label,
                confidence=result.confidence,
                passed=result.passed,
                message=result.message,
                metadata=result.metadata,
                score=result.score,
                issues=result.issues,
                suggestions=result.suggestions,
                processing_time_ms=result.processing_time_ms,
                timestamp=result.timestamp,
            )

        def batch_classify(self, texts: List[str]) -> List[ClassificationResult[Any, Any]]:
            results = self.classifier_impl.batch_classify(texts)
            # Convert each result to the expected type
            return [
                ClassificationResult(
                    label=r.label,
                    confidence=r.confidence,
                    passed=r.passed,
                    message=r.message,
                    metadata=r.metadata,
                    score=r.score,
                    issues=r.issues,
                    suggestions=r.suggestions,
                    processing_time_ms=r.processing_time_ms,
                    timestamp=r.timestamp,
                )
                for r in results
            ]

    # Wrap the classifier
    wrapped_classifier = ClassifierWrapper(classifier)

    # Cast the wrapped classifier to the Classifier protocol type
    classifier_protocol: Classifier = cast(Classifier, wrapped_classifier)

    # Create adapter with wrapped classifier
    adapter = ClassifierAdapter(
        classifier=classifier_protocol,
        threshold=threshold,
        valid_labels=valid_labels,
        **kwargs,
    )

    # Return adapter as BaseValidator[str]
    return cast(BaseValidator[str], adapter)


def create_bias_rule(
    name: str = "bias_rule",
    description: str = "Validates text for biased content",
    threshold: float = 0.3,
    rule_id: Optional[str] = None,
    **kwargs: Any,
) -> Rule[str]:
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
        ```
    """
    # Create the validator
    validator = create_bias_validator(threshold=threshold, **kwargs)

    # Create a new RuleConfig for the rule
    config = RuleConfig(
        name=name or "bias_rule",
        description=description or "Validates text for biased content",
        rule_id=rule_id or name or "bias_rule",
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["severity", "category", "tags", "priority", "cache_size", "cost"]
        },
    )

    # Create a concrete rule class that extends Rule
    class BiasRule(Rule[str]):
        def _create_default_validator(self) -> BaseValidator[str]:
            return validator

    # Create and return the rule
    return BiasRule(
        name=name or "bias_rule",
        description=description or "Validates text for biased content",
        config=config,
        validator=validator,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["severity", "category", "tags", "priority", "cache_size", "cost"]
        },
    )


def create_harmful_content_rule(
    name: str = "harmful_content_rule",
    description: str = "Validates text for harmful content",
    categories: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.0,
    fail_if_any: bool = True,
    rule_id: Optional[Optional[str]] = None,
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
