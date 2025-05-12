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
- Configuration is stored in dedicated config classes
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
result = (rule and rule.validate("This is a test.")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
"""

from typing import Any, List, Optional
import time

from pydantic import BaseModel, Field, ConfigDict

from sifaka.classifiers.implementations.content.profanity import ProfanityClassifier
from sifaka.classifiers.implementations.content.profanity import ClassifierConfig
from sifaka.rules.base import (
    BaseValidator,
    Rule,
    RuleConfig,
    RuleResult,
)
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


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
    """
    Configuration for prohibited content validation.

    This class defines the configuration options for prohibited content validation,
    including terms to check for, detection threshold, and matching options.
    It's used by ProhibitedContentValidator implementations to determine validation behavior.

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create from RuleConfig.params

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for threshold (0.0-1.0)
       - Minimum length validation for terms list
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by ProhibitedContentAnalyzer
       - Used by DefaultProhibitedContentValidator
       - Used by ProhibitedContentRule._create_default_validator
       - Used by create_prohibited_content_validator factory function

    ## Error Handling

    - Type validation through Pydantic
    - Range validation for threshold (0.0-1.0)
    - Immutability prevents accidental modification
    - Extra fields rejected with extra="forbid"

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import ProhibitedContentConfig

    # Create with default values
    config = ProhibitedContentConfig()

    # Create with custom values
    config = ProhibitedContentConfig(
        terms=["inappropriate", "offensive", "vulgar"],
        threshold=0.7,
        case_sensitive=True
    )

    # Create from dictionary
    config_dict = {
        "terms": ["inappropriate", "offensive"],
        "threshold": 0.8,
        "case_sensitive": False
    }
    config = (ProhibitedContentConfig and ProhibitedContentConfig.model_validate(config_dict)
    ```

    Using with validators:

    ```python
    from sifaka.rules.content.prohibited import (
        ProhibitedContentConfig,
        DefaultProhibitedContentValidator
    )

    # Create config
    config = ProhibitedContentConfig(
        terms=["inappropriate", "offensive"],
        threshold=0.7
    )

    # Create validator with config
    validator = DefaultProhibitedContentValidator(config)

    # Validate text
    result = (validator and validator.validate("This is a test.")
    print(f"Valid: {result.passed}")
    ```
    """

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
    """
    Analyzer for prohibited content detection.

    This class is responsible for analyzing text for prohibited content using
    a ProfanityClassifier. It follows the Single Responsibility Principle by
    focusing solely on content analysis.

    ## Architecture

    ProhibitedContentAnalyzer follows a component-based architecture:
    - Uses ProhibitedContentConfig for configuration
    - Delegates classification to ProfanityClassifier
    - Provides methods for analysis and capability checking

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with ProhibitedContentConfig
       - Extract configuration parameters
       - Create ProfanityClassifier with custom words

    2. **Analysis**: Check text for prohibited content
       - Delegate to ProfanityClassifier for content detection
       - Interpret classification results
       - Return RuleResult with validation results and metadata

    3. **Capability Check**: Determine if text can be analyzed
       - Check if input is a string
       - Return boolean indicating capability

    ## Error Handling

    - Type checking for input text
    - Delegation to ProfanityClassifier for classification errors
    - Detailed metadata for debugging and analysis

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import ProhibitedContentAnalyzer, ProhibitedContentConfig

    # Create configuration
    config = ProhibitedContentConfig(
        terms=["inappropriate", "offensive"],
        threshold=0.7
    )

    # Create analyzer
    analyzer = ProhibitedContentAnalyzer(config)

    # Check if text can be analyzed
    if (analyzer and analyzer.can_analyze("This is a test."):
        # Analyze text
        result = (analyzer and analyzer.analyze("This is a test.")
        print(f"Valid: {result.passed}")
        print(f"Confidence: {result.(metadata and metadata.get('confidence')}")
        print(f"Label: {result.(metadata and metadata.get('label')}")
    ```

    Using with custom terms:

    ```python
    # Create analyzer with custom terms
    config = ProhibitedContentConfig(
        terms=["custom_term1", "custom_term2"],
        threshold=0.8,
        case_sensitive=True
    )
    analyzer = ProhibitedContentAnalyzer(config)

    # Analyze text containing custom terms
    result = (analyzer and analyzer.analyze("This text contains custom_term1.")
    print(f"Valid: {result.passed}")
    ```
    """

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
        result = self.(_classifier and _classifier.classify(text)

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
    """
    Default validator for prohibited content.

    This class implements the BaseValidator interface for prohibited content validation.
    It delegates the actual validation logic to a ProhibitedContentAnalyzer instance,
    following the standard Sifaka delegation pattern.

    ## Architecture

    DefaultProhibitedContentValidator follows a component-based architecture:
    - Inherits from BaseValidator for common validation functionality
    - Uses ProhibitedContentConfig for configuration
    - Delegates analysis to ProhibitedContentAnalyzer
    - Handles empty text consistently using BaseValidator.handle_empty_text

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with ProhibitedContentConfig
       - Create ProhibitedContentAnalyzer with configuration

    2. **Validation**: Check text for prohibited content
       - Handle empty text using BaseValidator.handle_empty_text
       - Delegate to ProhibitedContentAnalyzer for content detection
       - Return RuleResult with validation results

    ## Error Handling

    - Empty text handling through BaseValidator.handle_empty_text
    - Delegation to ProhibitedContentAnalyzer for analysis errors
    - Configuration validation through ProhibitedContentConfig

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import (
        DefaultProhibitedContentValidator,
        ProhibitedContentConfig
    )

    # Create configuration
    config = ProhibitedContentConfig(
        terms=["inappropriate", "offensive"],
        threshold=0.7
    )

    # Create validator
    validator = DefaultProhibitedContentValidator(config)

    # Validate text
    result = (validator and validator.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Handle empty text
    result = (validator and validator.validate("")
    print(f"Empty text result: {result.passed}")
    print(f"Message: {result.message}")
    ```

    Accessing configuration:

    ```python
    # Create validator
    config = ProhibitedContentConfig(terms=["term1", "term2"])
    validator = DefaultProhibitedContentValidator(config)

    # Access configuration
    print(f"Terms: {validator.config.terms}")
    print(f"Threshold: {validator.config.threshold}")
    print(f"Case sensitive: {validator.config.case_sensitive}")
    ```
    """

    def __init__(self, config: ProhibitedContentConfig) -> None:
        """
        Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(validation_type=str)

        # Store configuration in state
        self.(_state_manager and _state_manager.update("config", config)
        self.(_state_manager and _state_manager.update("analyzer", ProhibitedContentAnalyzer(config))

        # Set metadata
        self.(_state_manager and _state_manager.set_metadata("validator_type", self.__class__.__name__)
        self.(_state_manager and _state_manager.set_metadata("creation_time", (time and time.time())

    @property
    def config(self) -> ProhibitedContentConfig:
        """
        Get the validator configuration.

        Returns:
            The validator configuration
        """
        return self.(_state_manager and _state_manager.get("config")

    def validate(self, text: str) -> RuleResult:
        """
        Validate the given text for prohibited content.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        start_time = (time and time.time()

        # Handle empty text
        empty_result = (self and self.handle_empty_text(text)
        if empty_result:
            return empty_result

        try:
            # Get analyzer from state
            analyzer = self.(_state_manager and _state_manager.get("analyzer")

            # Delegate to analyzer
            result = (analyzer and analyzer.analyze(text)

            # Add processing time and validator type
            result = (result and result.with_metadata(
                processing_time_ms=(time and time.time() - start_time,
                validator_type=self.__class__.__name__,
            )

            # Update statistics
            (self and self.update_statistics(result)

            # Update validation count in metadata
            validation_count = self.(_state_manager and _state_manager.get_metadata("validation_count", 0)
            self.(_state_manager and _state_manager.set_metadata("validation_count", validation_count + 1)

            # Cache result if caching is enabled
            if self.config.cache_size > 0:
                cache = self.(_state_manager and _state_manager.get("cache", {})
                if len(cache) >= self.config.cache_size:
                    # Clear cache if it's full
                    cache = {}
                cache[text] = result
                self.(_state_manager and _state_manager.update("cache", cache)

            return result

        except Exception as e:
            (self and self.record_error(e)
            (logger and logger.error(f"Error validating text for prohibited content: {e}")

            return RuleResult(
                passed=False,
                message=f"Error validating prohibited content: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "validator_type": self.__class__.__name__,
                },
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Check the text for problematic content"],
                processing_time_ms=(time and time.time() - start_time,
            )


class ProhibitedContentRule(Rule[str]):
    """
    Rule for validating prohibited content.

    This class implements the Rule interface for prohibited content validation.
    It delegates the actual validation logic to a DefaultProhibitedContentValidator
    instance, following the standard Sifaka delegation pattern.

    ## Architecture

    ProhibitedContentRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Delegates validation to DefaultProhibitedContentValidator
    - Uses RuleConfig for configuration
    - Creates a default validator if none is provided

    ## Lifecycle

    1. **Initialization**: Set up with configuration and validator
       - Initialize with name, description, config, and optional validator
       - Create default validator if none is provided

    2. **Validation**: Check text for prohibited content
       - Delegate to validator for validation logic
       - Add rule_id to metadata for traceability
       - Return RuleResult with validation results

    ## Error Handling

    - Validator creation through _create_default_validator
    - Validation delegation to validator
    - Rule identification through rule_id in metadata

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import ProhibitedContentRule
    from sifaka.rules.base import RuleConfig

    # Create rule with default validator
    rule = ProhibitedContentRule(
        name="prohibited_content_rule",
        description="Validates text for prohibited content",
        config=RuleConfig(
            params={
                "terms": ["inappropriate", "offensive"],
                "threshold": 0.7,
                "case_sensitive": False
            }
        )
    )

    # Validate text
    result = (rule and rule.validate("This is a test.")
    print(f"Valid: {result.passed}")

    # Check rule identification
    print(f"Rule ID: {result.(metadata and metadata.get('rule_id')}")
    ```

    Using with factory function:

    ```python
    from sifaka.rules.content.prohibited import create_prohibited_content_rule

    # Create rule using factory function
    rule = create_prohibited_content_rule(
        name="custom_prohibited_rule",
        terms=["term1", "term2"],
        threshold=0.8
    )

    # Validate text
    result = (rule and rule.validate("This text contains term1.")
    print(f"Valid: {result.passed}")
    ```
    """

    def __init__(
        self,
        name: str = "prohibited_content_rule",
        description: str = "Validates text for prohibited content",
        config: Optional[Optional[RuleConfig]] = None,
        validator: Optional[Optional[DefaultProhibitedContentValidator]] = None,
    ) -> None:
        """
        Initialize with configuration.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional validator implementation
        """
        super().__init__(
            name=name,
            description=description,
            config=config
            or RuleConfig(
                name=name,
                description=description,
            ),
            validator=validator,
        )

        # Store validator in state
        prohibited_validator = validator or (self and self._create_default_validator()
        self.(_state_manager and _state_manager.update("prohibited_validator", prohibited_validator)

        # Set additional metadata
        self.(_state_manager and _state_manager.set_metadata("rule_type", "ProhibitedContentRule")
        self.(_state_manager and _state_manager.set_metadata("creation_time", (time and time.time())

    def _create_default_validator(self) -> DefaultProhibitedContentValidator:
        """
        Create a default validator from config.

        Returns:
            A configured DefaultProhibitedContentValidator
        """
        # Extract prohibited content specific params
        params = self.config.params
        config = ProhibitedContentConfig(
            terms=(params and params.get("terms", []),
            threshold=(params and params.get("threshold", 0.5),
            case_sensitive=(params and params.get("case_sensitive", False),
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )

        # Store config in state for reference
        self.(_state_manager and _state_manager.update("validator_config", config)

        return DefaultProhibitedContentValidator(config)


def create_prohibited_content_validator(
    terms: Optional[Optional[List[str]]] = None,
    threshold: Optional[Optional[float]] = None,
    case_sensitive: Optional[Optional[bool]] = None,
    **kwargs: Any,
) -> DefaultProhibitedContentValidator:
    """
    Create a prohibited content validator.

    This factory function creates a configured DefaultProhibitedContentValidator instance.
    It's useful when you need a validator without creating a full rule.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract configuration parameters (terms, threshold, case_sensitive)
       - Handle optional parameters with None values
       - Collect additional parameters from kwargs

    2. **Configuration Creation**: Create configuration object
       - Create ProhibitedContentConfig with processed parameters
       - Apply validation through Pydantic

    3. **Validator Creation**: Create validator instance
       - Create DefaultProhibitedContentValidator with configuration
       - Return the configured validator

    ## Error Handling

    - Parameter validation through ProhibitedContentConfig
    - Optional parameters handled gracefully
    - Additional parameters passed through kwargs

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import create_prohibited_content_validator

    # Create validator with default settings
    validator = create_prohibited_content_validator()

    # Create validator with custom settings
    validator = create_prohibited_content_validator(
        terms=["inappropriate", "offensive"],
        threshold=0.7,
        case_sensitive=True
    )

    # Validate text
    result = (validator and validator.validate("This is a test.")
    print(f"Valid: {result.passed}")
    ```

    Using with additional configuration:

    ```python
    # Create validator with additional configuration
    validator = create_prohibited_content_validator(
        terms=["term1", "term2"],
        threshold=0.8,
        cache_size=200,
        priority=2,
        cost=0.5
    )

    # Access configuration
    print(f"Terms: {validator.config.terms}")
    print(f"Cache size: {validator.config.cache_size}")
    ```

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
    (config_params.update(kwargs)

    # Create config
    config = ProhibitedContentConfig(**config_params)

    # Create validator
    return DefaultProhibitedContentValidator(config)


def create_prohibited_content_rule(
    name: str = "prohibited_content_rule",
    description: str = "Validates text for prohibited content",
    terms: Optional[Optional[List[str]]] = None,
    threshold: Optional[Optional[float]] = None,
    case_sensitive: Optional[Optional[bool]] = None,
    rule_id: Optional[Optional[str]] = None,
    **kwargs: Any,
) -> ProhibitedContentRule:
    """
    Create a prohibited content rule.

    This factory function creates a configured ProhibitedContentRule instance.
    It uses create_prohibited_content_validator internally to create the validator.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract rule parameters (name, description, rule_id)
       - Extract configuration parameters (terms, threshold, case_sensitive)
       - Extract RuleConfig parameters (severity, category, tags, priority, cache_size, cost)
       - Handle optional parameters with None values

    2. **Validator Creation**: Create validator instance
       - Call create_prohibited_content_validator with processed parameters
       - Pass through relevant parameters

    3. **Configuration Creation**: Create RuleConfig object
       - Create params dictionary with processed parameters
       - Create RuleConfig with params and rule parameters

    4. **Rule Creation**: Create rule instance
       - Create ProhibitedContentRule with name, description, config, and validator
       - Return the configured rule

    ## Error Handling

    - Parameter validation through ProhibitedContentConfig
    - Optional parameters handled gracefully
    - RuleConfig parameters extracted and processed separately

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.content.prohibited import create_prohibited_content_rule

    # Create rule with default settings
    rule = create_prohibited_content_rule()

    # Create rule with custom settings
    rule = create_prohibited_content_rule(
        name="custom_prohibited_rule",
        description="Custom rule for prohibited content",
        terms=["inappropriate", "offensive"],
        threshold=0.7,
        case_sensitive=True
    )

    # Validate text
    result = (rule.validate("This is a test.")
    print(f"Valid: {result.passed}")
    print(f"Rule ID: {result.(metadata and metadata.get('rule_id')}")
    ```

    Using with additional configuration:

    ```python
    # Create rule with additional configuration
    rule = create_prohibited_content_rule(
        terms=["term1", "term2"],
        threshold=0.8,
        rule_id="prohibited_content_checker",
        severity="warning",
        category="content",
        tags=["prohibited", "content", "safety"],
        priority=2,
        cost=0.5
    )
    ```

    Args:
        name: The name of the rule
        description: Description of the rule
        terms: List of prohibited terms to check for
        threshold: Threshold for prohibited content detection
        case_sensitive: Whether to perform case-sensitive matching
        rule_id: Unique identifier for the rule
        **kwargs: Additional keyword arguments including:
            - severity: Severity level for rule violations
            - category: Category of the rule
            - tags: List of tags for categorizing the rule
            - priority: Priority level for validation
            - cache_size: Size of the validation cache
            - cost: Computational cost of validation

    Returns:
        ProhibitedContentRule: The created rule
    """
    # Create validator using the validator factory
    validator = create_prohibited_content_validator(
        terms=terms,
        threshold=threshold,
        case_sensitive=case_sensitive,
    )

    # Create params dictionary for RuleConfig
    params = {}
    if terms is not None:
        params["terms"] = terms
    if threshold is not None:
        params["threshold"] = threshold
    if case_sensitive is not None:
        params["case_sensitive"] = case_sensitive

    # Determine rule name
    rule_name = name or rule_id or "prohibited_content_rule"

    # Create RuleConfig
    config = RuleConfig(
        name=rule_name,
        description=description,
        rule_id=rule_id or rule_name,
        params=params,
        **kwargs,
    )

    # Create rule
    return ProhibitedContentRule(
        name=rule_name,
        description=description,
        config=config,
        validator=validator,
    )
