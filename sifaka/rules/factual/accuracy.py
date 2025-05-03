"""
Accuracy validation rules for Sifaka.

This module provides rules for validating factual accuracy in text, including:
- Fact verification
- Accuracy scoring
- Knowledge base validation

## Architecture Overview

The accuracy validation system follows a component-based architecture:

1. **AccuracyConfig**: Configuration class for accuracy validation
   - Defines knowledge base, threshold, and other parameters
   - Provides validation for configuration values
   - Uses Pydantic for schema validation

2. **DefaultAccuracyValidator**: Validator implementation
   - Implements BaseFactualValidator interface
   - Validates text against knowledge base
   - Calculates accuracy score

3. **AccuracyRule**: Rule implementation
   - Implements Rule interface
   - Delegates validation to DefaultAccuracyValidator
   - Provides standard rule interface

4. **Factory Functions**: Creation helpers
   - create_accuracy_validator: Creates standalone validator
   - create_accuracy_rule: Creates rule with validator

## Component Lifecycle

### AccuracyConfig
1. **Creation**: Instantiate with knowledge base and threshold
2. **Validation**: Values are validated by Pydantic
3. **Usage**: Pass to validator and rule constructors

### DefaultAccuracyValidator
1. **Initialization**: Set up with AccuracyConfig
2. **Validation**: Check text against knowledge base
3. **Result**: Return RuleResult with score and message

### AccuracyRule
1. **Initialization**: Set up with AccuracyConfig
2. **Validator Creation**: Create DefaultAccuracyValidator
3. **Validation**: Delegate to validator
4. **Result**: Return RuleResult from validator

## Error Handling Patterns

The accuracy validation system implements several error handling patterns:

1. **Configuration Validation**: Validates all configuration values
   - Ensures knowledge base is not empty
   - Ensures threshold is between 0.0 and 1.0
   - Rejects invalid configuration values

2. **Empty Knowledge Base Handling**: Handles empty knowledge base
   - Returns 0.0 score if knowledge base is empty
   - Provides clear error message

3. **Input Validation**: Validates input text
   - Handles empty text through BaseFactualValidator
   - Provides clear error message for invalid input

## Usage Examples

Basic usage with create_accuracy_rule:

```python
from sifaka.rules.factual.accuracy import create_accuracy_rule

# Create an accuracy rule
rule = create_accuracy_rule(
    knowledge_base=[
        "The Earth is round",
        "Water boils at 100°C at sea level",
        "The capital of France is Paris"
    ],
    threshold=0.8
)

# Validate text
result = rule.validate("The Earth is round and the capital of France is Paris.")
print(f"Valid: {result.is_valid}")
print(f"Score: {result.score}")
print(f"Message: {result.message}")
```

Using with custom configuration:

```python
from sifaka.rules.factual.accuracy import create_accuracy_rule

# Create rule with custom configuration
rule = create_accuracy_rule(
    name="science_facts_rule",
    description="Validates scientific facts",
    knowledge_base=[
        "The Earth orbits the Sun",
        "Gravity causes objects to fall",
        "Water is composed of hydrogen and oxygen"
    ],
    threshold=0.7,
    cache_size=200,
    priority=2,
    cost=1.5
)

# Validate text
result = rule.validate("Gravity causes objects to fall toward Earth.")
print(f"Valid: {result.is_valid}")
```

Using standalone validator:

```python
from sifaka.rules.factual.accuracy import create_accuracy_validator

# Create standalone validator
validator = create_accuracy_validator(
    knowledge_base=["The sky is blue", "Grass is green"],
    threshold=0.5
)

# Validate text
result = validator.validate("The sky is blue.")
print(f"Valid: {result.is_valid}")
print(f"Score: {result.score}")
```

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- All rule-specific configuration is stored in AccuracyConfig
- Factory functions handle configuration creation
- Validator factory functions create standalone validators
- Rule factory functions create rules with validators
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from sifaka.rules.base import (
    BaseValidator,
    ConfigurationError,
    Rule,
    RuleConfig,
    RuleResult,
    RuleResultHandler,
    ValidationError,
)
from sifaka.rules.factual.base import BaseFactualValidator


class AccuracyConfig(BaseModel):
    """
    Configuration for accuracy validation.

    This class defines the configuration options for accuracy validation,
    including the knowledge base, threshold, and other parameters.
    It's used by DefaultAccuracyValidator and AccuracyRule to determine
    validation behavior.

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
       - Create directly with parameters
       - Create from dictionary with model_validate
       - Create through factory functions

    2. **Validation**: Values are validated by Pydantic
       - Type checking for all fields
       - Range validation for threshold (0.0-1.0)
       - Minimum length validation for knowledge_base
       - Immutability enforced by frozen=True

    3. **Usage**: Pass to validators and rules
       - Used by DefaultAccuracyValidator
       - Used by AccuracyRule
       - Used by create_accuracy_validator
       - Used by create_accuracy_rule

    ## Error Handling

    - Type validation through Pydantic
    - Range validation for threshold (0.0-1.0)
    - Minimum length validation for knowledge_base
    - Immutability prevents accidental modification
    - Extra fields rejected with extra="forbid"

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.factual.accuracy import AccuracyConfig

    # Create with default values
    config = AccuracyConfig()

    # Create with custom values
    config = AccuracyConfig(
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Create from dictionary
    config_dict = {
        "knowledge_base": ["The Earth is round"],
        "threshold": 0.8,
        "cache_size": 150
    }
    config = AccuracyConfig(**config_dict)
    ```

    Using with validators:

    ```python
    from sifaka.rules.factual.accuracy import AccuracyConfig, DefaultAccuracyValidator

    # Create config
    config = AccuracyConfig(
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7
    )

    # Create validator with config
    validator = DefaultAccuracyValidator(config)

    # Validate text
    result = validator.validate("The Earth is round.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knowledge_base: List[str] = Field(
        default_factory=list,
        description="List of known facts for validation",
        min_length=1,
        json_schema_extra={"examples": ["The Earth is round", "Water boils at 100°C at sea level"]},
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy score required",
        json_schema_extra={"examples": [0.8, 0.9]},
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


class DefaultAccuracyValidator(BaseFactualValidator):
    """
    Default validator for accuracy validation.

    This class implements the BaseFactualValidator interface for accuracy validation.
    It validates text against a knowledge base of facts and calculates an accuracy
    score based on the number of matching facts.

    ## Architecture

    DefaultAccuracyValidator follows a component-based architecture:
    - Inherits from BaseFactualValidator for common validation functionality
    - Uses AccuracyConfig for configuration
    - Implements validate method to check text against knowledge base
    - Calculates accuracy score based on matching facts

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with AccuracyConfig
       - Extract knowledge base and threshold from config
       - Store configuration for use during validation

    2. **Validation**: Check text against knowledge base
       - Count matching facts in the text
       - Calculate accuracy score
       - Determine validity based on threshold
       - Return RuleResult with validation results

    ## Error Handling

    - Empty knowledge base handling (returns 0.0 score)
    - Input validation through BaseFactualValidator
    - Configuration validation through AccuracyConfig

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.factual.accuracy import DefaultAccuracyValidator, AccuracyConfig

    # Create configuration
    config = AccuracyConfig(
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7
    )

    # Create validator
    validator = DefaultAccuracyValidator(config)

    # Validate text
    result = validator.validate("The Earth is round and water freezes at 0°C.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    print(f"Message: {result.message}")
    ```

    Using with factory function:

    ```python
    from sifaka.rules.factual.accuracy import create_accuracy_validator

    # Create validator using factory function
    validator = create_accuracy_validator(
        knowledge_base=["The sky is blue", "Grass is green"],
        threshold=0.5
    )

    # Validate text
    result = validator.validate("The sky is blue.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```
    """

    def __init__(self, config: AccuracyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the validator
        """
        super().__init__(config)
        self._knowledge_base = config.knowledge_base
        self._threshold = config.threshold

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for accuracy.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        # Count matching facts
        matching_facts = sum(1 for fact in self._knowledge_base if fact.lower() in text.lower())
        total_facts = len(self._knowledge_base)

        # Calculate accuracy score
        accuracy_score = matching_facts / total_facts if total_facts > 0 else 0.0
        is_valid = accuracy_score >= self._threshold

        return RuleResult(
            is_valid=is_valid,
            score=accuracy_score,
            message=f"Accuracy score: {accuracy_score:.2f} (threshold: {self._threshold})",
        )


class AccuracyRule(Rule):
    """
    Rule for validating accuracy.

    This class implements the Rule interface for accuracy validation.
    It delegates the actual validation logic to a DefaultAccuracyValidator
    instance, following the standard Sifaka delegation pattern.

    ## Architecture

    AccuracyRule follows a component-based architecture:
    - Inherits from Rule for common rule functionality
    - Delegates validation to DefaultAccuracyValidator
    - Uses AccuracyConfig for configuration
    - Creates validator during initialization

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Initialize with AccuracyConfig
       - Create DefaultAccuracyValidator with configuration
       - Store validator for use during validation

    2. **Validation**: Check text for accuracy
       - Delegate to validator for validation logic
       - Return RuleResult with validation results

    ## Error Handling

    - Configuration validation through AccuracyConfig
    - Validation delegation to DefaultAccuracyValidator
    - Input validation through BaseFactualValidator

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.factual.accuracy import AccuracyRule, AccuracyConfig

    # Create configuration
    config = AccuracyConfig(
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7
    )

    # Create rule
    rule = AccuracyRule(config)

    # Validate text
    result = rule.validate("The Earth is round and water freezes at 0°C.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    print(f"Message: {result.message}")
    ```

    Using with factory function:

    ```python
    from sifaka.rules.factual.accuracy import create_accuracy_rule

    # Create rule using factory function
    rule = create_accuracy_rule(
        knowledge_base=["The sky is blue", "Grass is green"],
        threshold=0.5
    )

    # Validate text
    result = rule.validate("The sky is blue.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```
    """

    def __init__(self, config: AccuracyConfig) -> None:
        """Initialize with configuration.

        Args:
            config: The configuration for the rule
        """
        super().__init__(config)
        self._validator = DefaultAccuracyValidator(config)

    def validate(self, text: str) -> RuleResult:
        """Validate the given text for accuracy.

        Args:
            text: The text to validate

        Returns:
            RuleResult: The result of the validation
        """
        return self._validator.validate(text)


def create_accuracy_validator(
    knowledge_base: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> DefaultAccuracyValidator:
    """
    Create an accuracy validator.

    This factory function creates a configured DefaultAccuracyValidator instance.
    It's useful when you need a validator without creating a full rule.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract configuration parameters (knowledge_base, threshold)
       - Handle optional parameters with None values
       - Collect additional parameters from kwargs

    2. **Configuration Creation**: Create configuration object
       - Create AccuracyConfig with processed parameters
       - Apply validation through Pydantic

    3. **Validator Creation**: Create validator instance
       - Create DefaultAccuracyValidator with configuration
       - Return the configured validator

    ## Error Handling

    - Parameter validation through AccuracyConfig
    - Optional parameters handled gracefully
    - Additional parameters passed through kwargs

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.factual.accuracy import create_accuracy_validator

    # Create validator with default settings
    validator = create_accuracy_validator(
        knowledge_base=["The Earth is round"]
    )

    # Create validator with custom settings
    validator = create_accuracy_validator(
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7
    )

    # Validate text
    result = validator.validate("The Earth is round.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```

    Using with additional configuration:

    ```python
    # Create validator with additional configuration
    validator = create_accuracy_validator(
        knowledge_base=["The sky is blue", "Grass is green"],
        threshold=0.8,
        cache_size=200,
        priority=2,
        cost=0.5
    )

    # Access configuration
    print(f"Knowledge base: {validator._knowledge_base}")
    print(f"Threshold: {validator._threshold}")
    ```

    Args:
        knowledge_base: List of known facts for validation
        threshold: Minimum accuracy score required
        **kwargs: Additional keyword arguments for the config

    Returns:
        DefaultAccuracyValidator: The created validator
    """
    # Create config with default or provided values
    config_params = {}
    if knowledge_base is not None:
        config_params["knowledge_base"] = knowledge_base
    if threshold is not None:
        config_params["threshold"] = threshold

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = AccuracyConfig(**config_params)

    # Create validator
    return DefaultAccuracyValidator(config)


def create_accuracy_rule(
    name: str = "accuracy_rule",
    description: str = "Validates text for factual accuracy",
    knowledge_base: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> AccuracyRule:
    """
    Create an accuracy rule.

    This factory function creates a configured AccuracyRule instance.
    It provides a convenient way to create and configure an accuracy rule
    with a single function call.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract rule parameters (name, description)
       - Extract configuration parameters (knowledge_base, threshold)
       - Handle optional parameters with None values
       - Collect additional parameters from kwargs

    2. **Configuration Creation**: Create configuration object
       - Create config dictionary with processed parameters
       - Create AccuracyConfig with config dictionary
       - Apply validation through Pydantic

    3. **Rule Creation**: Create rule instance
       - Create AccuracyRule with configuration
       - Return the configured rule

    ## Error Handling

    - Parameter validation through AccuracyConfig
    - Optional parameters handled gracefully
    - Additional parameters passed through kwargs

    ## Examples

    Basic usage:

    ```python
    from sifaka.rules.factual.accuracy import create_accuracy_rule

    # Create rule with default settings
    rule = create_accuracy_rule(
        knowledge_base=["The Earth is round"]
    )

    # Create rule with custom settings
    rule = create_accuracy_rule(
        name="science_facts_rule",
        description="Validates scientific facts",
        knowledge_base=["The Earth is round", "Water boils at 100°C at sea level"],
        threshold=0.7
    )

    # Validate text
    result = rule.validate("The Earth is round.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```

    Using with additional configuration:

    ```python
    # Create rule with additional configuration
    rule = create_accuracy_rule(
        knowledge_base=["The sky is blue", "Grass is green"],
        threshold=0.8,
        cache_size=200,
        priority=2,
        cost=0.5
    )

    # Validate text
    result = rule.validate("The sky is blue and grass is green.")
    print(f"Valid: {result.is_valid}")
    print(f"Score: {result.score}")
    ```

    Args:
        name: The name of the rule
        description: Description of the rule
        knowledge_base: List of known facts for validation
        threshold: Minimum accuracy score required
        **kwargs: Additional keyword arguments for the rule

    Returns:
        AccuracyRule: The created rule
    """
    # Create config dictionary
    config_dict = {
        "knowledge_base": knowledge_base or [],
        "threshold": threshold or 0.8,
        **kwargs,
    }

    # Create config
    config = AccuracyConfig(**config_dict)

    # Create rule
    return AccuracyRule(config)
