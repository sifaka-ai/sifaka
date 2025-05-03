"""
Base module for critics that provide feedback and validation on prompts.

This module defines the core interfaces and base implementations for critics
that analyze and improve text outputs based on rule violations. Critics are
a key component in the Sifaka framework, providing feedback and improvement
suggestions for text that fails validation rules.

## Architecture Overview

The critic system follows a layered architecture:

1. **BaseCritic**: High-level interface for text critique and improvement
2. **CriticConfig**: Configuration and settings management
3. **CriticMetadata**: Standardized critique result format
4. **CriticOutput**: Complete output from critique and improvement process
5. **Protocol Classes**: Interface definitions for validators, improvers, and critics

## Component Lifecycle

### CriticConfig
1. **Creation**: Instantiate with name, description, and optional parameters
2. **Validation**: Values are validated in __post_init__
3. **Modification**: Create new instances with with_params()
4. **Usage**: Pass to critics for configuration

### CriticMetadata
1. **Creation**: Instantiate with score, feedback, and optional details
2. **Access**: Read score, feedback, issues, and suggestions
3. **Enhancement**: Create new instances with additional data using with_extra()

### BaseCritic
1. **Initialization**: Set up with configuration
2. **Validation**: Check if text meets quality standards
3. **Critique**: Analyze text and provide feedback
4. **Improvement**: Enhance text based on violations or feedback
5. **Processing**: Combine critique and improvement in a single operation

## Error Handling Patterns

The critic system implements several error handling patterns:

1. **Input Validation**: Validates all inputs before processing
   - Checks input types with is_valid_text()
   - Handles empty text gracefully
   - Validates configuration in _validate_config()

2. **Critique Errors**: Handles errors during critique
   - Returns valid CriticMetadata even when errors occur
   - Includes error details in metadata
   - Sets appropriate score for failed critiques

3. **Improvement Errors**: Handles errors during text improvement
   - Falls back to original text when improvement fails
   - Returns appropriate result status (SUCCESS, NEEDS_IMPROVEMENT, FAILURE)
   - Preserves original critique metadata

## Usage Examples

```python
from sifaka.critics.base import create_basic_critic, CriticResult

# Create a basic critic
critic = create_basic_critic(
    name="quality_critic",
    description="Checks text quality and provides improvements",
    min_confidence=0.7
)

# Validate text
text = "This is a sample text that needs improvement."
is_valid = critic.validate(text)
print(f"Text is valid: {is_valid}")

# Critique text
metadata = critic.critique(text)
print(f"Critique score: {metadata.score:.2f}")
print(f"Feedback: {metadata.feedback}")

if metadata.issues:
    print("Issues:")
    for issue in metadata.issues:
        print(f"- {issue}")

if metadata.suggestions:
    print("Suggestions:")
    for suggestion in metadata.suggestions:
        print(f"- {suggestion}")

# Process text with violations
violations = [
    {"rule": "length", "message": "Text is too short", "fix": lambda t: t + " Additional content."}
]
output = critic.process(text, violations)

print(f"Result: {output.result}")
print(f"Improved text: {output.improved_text}")
```

## Instantiation Pattern

The recommended way to create critics is through factory functions:

```python
from sifaka.critics.base import create_critic, Critic
from sifaka.critics.prompt import create_prompt_critic

# Create a basic critic
basic_critic = create_basic_critic(
    name="basic_critic",
    description="Basic text quality critic",
    min_confidence=0.6
)

# Create a custom critic
custom_critic = create_critic(
    critic_class=Critic,
    name="custom_critic",
    description="Custom critic implementation",
    min_confidence=0.8,
    max_attempts=5,
    params={"custom_param": "value"}
)

# Create a specialized prompt critic
prompt_critic = create_prompt_critic(
    llm_provider=model_provider,
    name="prompt_critic",
    description="LLM-based prompt critic",
    system_prompt="You are an expert editor that improves text."
)
```

Each critic type typically provides specialized factory functions for easier instantiation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    overload,
    runtime_checkable,
)

from typing_extensions import TypeGuard


# Input and output type variables
T = TypeVar("T")  # Input type (usually str)
R = TypeVar("R")  # Result type
C = TypeVar("C", bound="BaseCritic")  # Critic type


class CriticResult(str, Enum):
    """
    Enumeration of possible critic results.

    This enum defines the standard result states that can be returned by
    a critic's process() method, indicating the outcome of the critique
    and improvement process.

    ## Lifecycle

    1. **Definition**: Defined as enum values (SUCCESS, NEEDS_IMPROVEMENT, FAILURE)
    2. **Assignment**: Assigned in CriticOutput during processing
    3. **Usage**: Used to determine next steps in processing pipelines

    ## Values

    - **SUCCESS**: Text meets quality standards and needs no improvement
    - **NEEDS_IMPROVEMENT**: Text has been improved but may need further refinement
    - **FAILURE**: Text could not be improved or improvement failed

    ## Examples

    ```python
    from sifaka.critics.base import CriticResult, create_basic_critic

    critic = create_basic_critic()
    output = critic.process("Sample text", violations=[])

    if output.result == CriticResult.SUCCESS:
        print("Text meets quality standards")
    elif output.result == CriticResult.NEEDS_IMPROVEMENT:
        print("Text has been improved but may need further review")
        print(f"Improved text: {output.improved_text}")
    else:  # CriticResult.FAILURE
        print("Text could not be improved")
        print(f"Issues: {output.metadata.issues}")
    ```

    Using in conditional logic:

    ```python
    def handle_critic_result(output):
        if output.result == CriticResult.SUCCESS:
            return output.improved_text
        elif output.result == CriticResult.NEEDS_IMPROVEMENT:
            # Apply additional processing
            return further_improve(output.improved_text)
        else:  # CriticResult.FAILURE
            # Fall back to default text
            return "Unable to generate appropriate text"
    ```
    """

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticConfig(Generic[T]):
    """
    Immutable configuration for critics.

    This class provides a standardized way to configure critics with
    immutable properties. It follows the same pattern as RuleConfig and
    ClassifierConfig, where critic-specific configuration options are
    placed in the params dictionary.

    The immutable design ensures configuration consistency during critic
    operation and prevents accidental modification of settings.

    ## Lifecycle

    1. **Creation**: Instantiate with required and optional parameters
       - Provide name and description (required)
       - Set min_confidence, max_attempts, cache_size as needed
       - Add critic-specific options in params dictionary

    2. **Validation**: Values are validated in __post_init__
       - Name and description must be non-empty
       - min_confidence must be between 0 and 1
       - max_attempts must be positive
       - cache_size, priority, and cost must be non-negative

    3. **Usage**: Access configuration properties during critic operation
       - Read name and description for identification
       - Use min_confidence for quality thresholds
       - Use max_attempts for retry limits
       - Access critic-specific params as needed

    4. **Modification**: Create new instances with updated values
       - Use with_params() to update critic-specific parameters
       - Original configuration remains unchanged (immutable)

    ## Error Handling

    The class implements these error handling patterns:
    - Validation of all parameters in __post_init__
    - Immutability to prevent runtime configuration errors
    - Type checking for critical parameters
    - Range validation for numeric parameters

    ## Examples

    Creating and using a critic config:

    ```python
    from sifaka.critics.base import CriticConfig

    # Create a basic config
    config = CriticConfig(
        name="quality_critic",
        description="Checks text quality and provides improvements",
        min_confidence=0.7,
        max_attempts=3,
        params={
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    )

    # Create a modified version
    updated_config = config.with_params(temperature=0.5, use_gpu=True)

    # Access configuration values
    print(f"Critic name: {config.name}")
    print(f"Min confidence: {config.min_confidence}")
    print(f"Temperature: {updated_config.params['temperature']}")
    ```

    Error handling with validation:

    ```python
    from sifaka.critics.base import CriticConfig

    try:
        # This will raise an error due to invalid min_confidence
        config = CriticConfig(
            name="invalid_critic",
            description="Invalid configuration example",
            min_confidence=1.5  # Must be between 0 and 1
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        # Use default values instead
        config = CriticConfig(
            name="valid_critic",
            description="Valid configuration example",
            min_confidence=0.7  # Valid value
        )
    ```

    Creating specialized configurations:

    ```python
    from sifaka.critics.base import CriticConfig

    # Create a config for a high-precision critic
    high_precision = CriticConfig(
        name="precision_critic",
        description="High-precision text critic",
        min_confidence=0.9,  # Require high confidence
        params={"precision_focused": True}
    )

    # Create a config for a high-recall critic
    high_recall = CriticConfig(
        name="recall_critic",
        description="High-recall text critic",
        min_confidence=0.3,  # Accept lower confidence
        params={"recall_focused": True}
    )
    ```

    Attributes:
        name: Name of the critic
        description: Description of what this critic does
        min_confidence: Minimum confidence threshold for valid critiques
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the critique result cache (0 to disable)
        priority: Priority of the critic (higher values = higher priority)
        cost: Computational cost of using this critic
        params: Dictionary of critic-specific parameters
    """

    name: str
    description: str
    min_confidence: float = 0.7
    max_attempts: int = 3
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.name or not self.name.strip():
            raise ValueError("name cannot be empty or whitespace")
        if not self.description or not self.description.strip():
            raise ValueError("description cannot be empty or whitespace")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")

    def with_params(self, **kwargs: Any) -> "CriticConfig[T]":
        """Create a new config with updated parameters."""
        new_params = {**self.params, **kwargs}
        return CriticConfig(
            name=self.name,
            description=self.description,
            min_confidence=self.min_confidence,
            max_attempts=self.max_attempts,
            cache_size=self.cache_size,
            priority=self.priority,
            cost=self.cost,
            params=new_params,
        )


@dataclass(frozen=True)
class CriticMetadata(Generic[R]):
    """
    Immutable metadata for critic results.

    This class provides a standardized way to represent critique results,
    including a quality score, feedback text, identified issues, and
    improvement suggestions. The immutable design ensures result consistency
    and prevents accidental modification after critique.

    ## Lifecycle

    1. **Creation**: Instantiate with critique results
       - Provide score and feedback (required)
       - Add optional issues and suggestions lists
       - Include processing metrics and attempt information
       - Values are validated during creation

    2. **Access**: Read properties to get critique details
       - Access score for quantitative quality assessment
       - Read feedback for qualitative assessment
       - Examine issues list for identified problems
       - Review suggestions for improvement ideas

    3. **Enhancement**: Create new instances with additional data
       - Use with_extra() to add or update extra metadata
       - Original metadata remains unchanged (immutable)
       - Chain multiple with_extra() calls as needed

    4. **Usage**: Use in application logic
       - Compare score against confidence thresholds
       - Present feedback to users or systems
       - Apply suggestions for text improvement
       - Track processing metrics for performance analysis

    ## Error Handling

    The class implements these error handling patterns:
    - Validation of score range (0-1)
    - Validation of attempt_number (must be positive)
    - Validation of processing_time_ms (must be non-negative)
    - Immutability to prevent result tampering
    - Structured extra dictionary for additional error details

    ## Examples

    Creating and using critic metadata:

    ```python
    from sifaka.critics.base import CriticMetadata

    # Create basic metadata
    metadata = CriticMetadata(
        score=0.75,
        feedback="Text is generally good but could use some improvements",
        issues=["Text is slightly too short", "Missing conclusion"],
        suggestions=["Add more details", "Include a concluding paragraph"]
    )

    # Access metadata properties
    print(f"Quality score: {metadata.score:.2f}")
    print(f"Feedback: {metadata.feedback}")

    if metadata.issues:
        print("Issues:")
        for issue in metadata.issues:
            print(f"- {issue}")

    if metadata.suggestions:
        print("Suggestions:")
        for suggestion in metadata.suggestions:
            print(f"- {suggestion}")
    ```

    Using with confidence thresholds:

    ```python
    from sifaka.critics.base import CriticMetadata

    # Create metadata with different scores
    high_quality = CriticMetadata(score=0.95, feedback="Excellent text")
    medium_quality = CriticMetadata(score=0.7, feedback="Good text with minor issues")
    low_quality = CriticMetadata(score=0.3, feedback="Text needs significant improvement")

    # Apply different handling based on score
    def process_critique(metadata, min_confidence=0.8):
        if metadata.score >= min_confidence:
            print(f"High quality text: {metadata.feedback}")
            return "accept"
        elif metadata.score >= 0.5:
            print(f"Needs minor revisions: {metadata.feedback}")
            return "revise"
        else:
            print(f"Needs major revisions: {metadata.feedback}")
            print(f"Issues: {', '.join(metadata.issues)}")
            return "reject"

    # Process the metadata
    process_critique(high_quality)    # "accept"
    process_critique(medium_quality)  # "revise"
    process_critique(low_quality)     # "reject"
    ```

    Adding extra information:

    ```python
    from sifaka.critics.base import CriticMetadata
    import time

    # Create initial metadata
    start_time = time.time()
    metadata = CriticMetadata(
        score=0.8,
        feedback="Good quality text",
        processing_time_ms=(time.time() - start_time) * 1000
    )

    # Add extra information
    enhanced = metadata.with_extra(
        model_name="gpt-3.5-turbo",
        tokens_used=150
    ).with_extra(
        timestamp=time.time()
    )

    # Access extra information
    print(f"Model used: {enhanced.extra['model_name']}")
    print(f"Tokens used: {enhanced.extra['tokens_used']}")
    print(f"Timestamp: {enhanced.extra['timestamp']}")
    ```

    Attributes:
        score: Quality score between 0 and 1
        feedback: Textual feedback about the critique
        issues: List of identified issues
        suggestions: List of improvement suggestions
        attempt_number: Current attempt number (for multi-attempt processes)
        processing_time_ms: Processing time in milliseconds
        extra: Additional metadata as key-value pairs
    """

    score: float
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    attempt_number: int = 1
    processing_time_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata values."""
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")

    def with_extra(self, **kwargs: Any) -> "CriticMetadata[R]":
        """Create a new metadata with additional extra data."""
        new_extra = {**self.extra, **kwargs}
        return CriticMetadata(
            score=self.score,
            feedback=self.feedback,
            issues=self.issues,
            suggestions=self.suggestions,
            attempt_number=self.attempt_number,
            processing_time_ms=self.processing_time_ms,
            extra=new_extra,
        )


@dataclass(frozen=True)
class CriticOutput(Generic[T, R]):
    """
    Immutable output from a critic.

    This class represents the complete output from a critic's process() method,
    combining the result status, improved text, and detailed metadata. The
    immutable design ensures output consistency and prevents accidental
    modification after processing.

    ## Lifecycle

    1. **Creation**: Instantiate with process results
       - Provide result status (SUCCESS, NEEDS_IMPROVEMENT, FAILURE)
       - Include the improved text (or original if improvement failed)
       - Include detailed metadata from critique

    2. **Access**: Read properties to get process details
       - Check result status to determine outcome
       - Access improved_text for the enhanced content
       - Examine metadata for detailed critique information

    3. **Usage**: Use in application logic
       - Make decisions based on result status
       - Present improved text to users or systems
       - Use metadata for detailed feedback

    ## Examples

    Creating and using critic output:

    ```python
    from sifaka.critics.base import CriticOutput, CriticResult, CriticMetadata

    # Create metadata
    metadata = CriticMetadata(
        score=0.75,
        feedback="Text is generally good but could use some improvements",
        issues=["Text is slightly too short"],
        suggestions=["Add more details"]
    )

    # Create output
    output = CriticOutput(
        result=CriticResult.NEEDS_IMPROVEMENT,
        improved_text="This is the improved version of the text with more details.",
        metadata=metadata
    )

    # Use the output
    if output.result == CriticResult.SUCCESS:
        print("Text meets quality standards")
        print(output.improved_text)
    elif output.result == CriticResult.NEEDS_IMPROVEMENT:
        print("Text has been improved but may need further review")
        print(f"Improved text: {output.improved_text}")
        print(f"Feedback: {output.metadata.feedback}")
    else:  # CriticResult.FAILURE
        print("Text could not be improved")
        print(f"Issues: {', '.join(output.metadata.issues)}")
    ```

    Using in a processing pipeline:

    ```python
    from sifaka.critics.base import create_basic_critic, CriticResult

    critic = create_basic_critic()

    def process_text(text, violations):
        output = critic.process(text, violations)

        if output.result == CriticResult.SUCCESS:
            return output.improved_text, True

        elif output.result == CriticResult.NEEDS_IMPROVEMENT:
            # Log improvement details
            print(f"Text improved: {output.metadata.feedback}")

            # Check if score meets threshold
            if output.metadata.score >= 0.7:
                return output.improved_text, True
            else:
                # Try another improvement iteration
                return process_text(output.improved_text, violations)

        else:  # CriticResult.FAILURE
            print(f"Improvement failed: {output.metadata.feedback}")
            return text, False  # Return original text

    # Use the processing function
    improved_text, success = process_text("Original text", [])
    ```

    Attributes:
        result: The result status (SUCCESS, NEEDS_IMPROVEMENT, FAILURE)
        improved_text: The improved text (or original if improvement failed)
        metadata: Detailed metadata from the critique process
    """

    result: CriticResult
    improved_text: T
    metadata: CriticMetadata[R]


@runtime_checkable
class TextValidator(Protocol[T]):
    """
    Protocol for text validation.

    This protocol defines the interface for components that validate text
    against quality standards. It's used as a common interface for validation
    operations in the Sifaka critic framework.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the required methods
       - Implement config property to expose configuration
       - Implement validate() method to check text quality
       - Ensure method signatures match the protocol

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use the validator for text validation
       - Access configuration through the config property
       - Pass text to the validate() method
       - Receive boolean result indicating validity

    ## Error Handling

    Implementations should handle these error cases:
    - Empty or invalid text inputs
    - Validation failures
    - Resource availability issues

    ## Examples

    Implementing a simple text validator:

    ```python
    from sifaka.critics.base import TextValidator, CriticConfig
    from typing import runtime_checkable

    class LengthValidator:
        def __init__(self, min_length: int = 10, max_length: int = 1000):
            self._config = CriticConfig(
                name="length_validator",
                description="Validates text length",
                params={"min_length": min_length, "max_length": max_length}
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def validate(self, text: str) -> bool:
            # Handle empty text
            if not text or not text.strip():
                return False

            # Validate length
            text_length = len(text)
            min_length = self.config.params.get("min_length", 10)
            max_length = self.config.params.get("max_length", 1000)

            return min_length <= text_length <= max_length

    # Check if it adheres to the protocol
    validator = LengthValidator(min_length=20, max_length=500)
    assert isinstance(validator, TextValidator)

    # Use the validator
    is_valid = validator.validate("This is a test text")
    print(f"Text is valid: {is_valid}")
    ```

    Using with error handling:

    ```python
    from sifaka.critics.base import TextValidator, CriticConfig
    import logging

    logger = logging.getLogger(__name__)

    class RobustValidator:
        def __init__(self):
            self._config = CriticConfig(
                name="robust_validator",
                description="Validator with robust error handling"
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def validate(self, text: str) -> bool:
            try:
                if not text or not isinstance(text, str):
                    logger.warning("Invalid input: empty or non-string text")
                    return False

                # Perform validation
                # ...

                return True
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return False
    ```

    Type Parameters:
        T: The input type (usually str)
    """

    @property
    def config(self) -> CriticConfig[T]:
        """
        Get validator configuration.

        Returns:
            The validator configuration
        """
        ...

    def validate(self, text: T) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise
        """
        ...


@runtime_checkable
class TextImprover(Protocol[T, R]):
    """
    Protocol for text improvement.

    This protocol defines the interface for components that improve text
    based on rule violations. It's used as a common interface for text
    improvement operations in the Sifaka critic framework.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the required methods
       - Implement config property to expose configuration
       - Implement improve() method to enhance text quality
       - Ensure method signatures match the protocol

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use the improver for text enhancement
       - Access configuration through the config property
       - Pass text and violations to the improve() method
       - Receive improved text as result

    ## Error Handling

    Implementations should handle these error cases:
    - Empty or invalid text inputs
    - Invalid violation formats
    - Improvement failures
    - Resource availability issues

    ## Examples

    Implementing a simple text improver:

    ```python
    from sifaka.critics.base import TextImprover, CriticConfig
    from typing import Dict, List, Any, runtime_checkable

    class SimpleImprover:
        def __init__(self):
            self._config = CriticConfig(
                name="simple_improver",
                description="Improves text based on rule violations"
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
            # Handle empty text
            if not text or not text.strip():
                return "Default text content"

            # Apply improvements based on violations
            improved = text
            for violation in violations:
                rule_id = violation.get("rule_id", "unknown")
                message = violation.get("message", "")

                if rule_id == "length" and "too_short" in message:
                    improved += " Additional content to increase length."
                elif rule_id == "style" and "capitalization" in message:
                    improved = improved.capitalize()
                # Add more improvement logic for other violation types

            return improved

    # Check if it adheres to the protocol
    improver = SimpleImprover()
    assert isinstance(improver, TextImprover)

    # Use the improver
    violations = [
        {"rule_id": "length", "message": "Text is too_short"}
    ]
    improved = improver.improve("Short text", violations)
    print(f"Improved text: {improved}")
    ```

    Using with error handling:

    ```python
    from sifaka.critics.base import TextImprover, CriticConfig
    import logging

    logger = logging.getLogger(__name__)

    class RobustImprover:
        def __init__(self):
            self._config = CriticConfig(
                name="robust_improver",
                description="Improver with robust error handling"
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
            try:
                if not text or not isinstance(text, str):
                    logger.warning("Invalid input: empty or non-string text")
                    return "Default text content"

                if not violations:
                    return text  # No improvements needed

                # Apply improvements
                improved = text
                for violation in violations:
                    # Improvement logic
                    # ...

                return improved
            except Exception as e:
                logger.error(f"Improvement error: {e}")
                return text  # Return original text on error
    ```

    Type Parameters:
        T: The input type (usually str)
        R: The result type
    """

    @property
    def config(self) -> CriticConfig[T]:
        """
        Get improver configuration.

        Returns:
            The improver configuration
        """
        ...

    def improve(self, text: T, violations: List[Dict[str, Any]]) -> T:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        ...


@runtime_checkable
class TextCritic(Protocol[T, R]):
    """
    Protocol for text critiquing.

    This protocol defines the interface for components that critique text
    and provide detailed feedback. It's used as a common interface for
    critique operations in the Sifaka critic framework.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the required methods
       - Implement config property to expose configuration
       - Implement critique() method to analyze text quality
       - Ensure method signatures match the protocol

    2. **Verification**: Verify protocol compliance
       - Use isinstance() to check if an object implements the protocol
       - No explicit registration or inheritance is needed

    3. **Usage**: Use the critic for text analysis
       - Access configuration through the config property
       - Pass text to the critique() method
       - Receive detailed CriticMetadata as result

    ## Error Handling

    Implementations should handle these error cases:
    - Empty or invalid text inputs
    - Critique failures
    - Resource availability issues

    ## Examples

    Implementing a simple text critic:

    ```python
    from sifaka.critics.base import TextCritic, CriticConfig, CriticMetadata
    from typing import runtime_checkable

    class SimpleCritic:
        def __init__(self):
            self._config = CriticConfig(
                name="simple_critic",
                description="Analyzes text quality"
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def critique(self, text: str) -> CriticMetadata:
            # Handle empty text
            if not text or not text.strip():
                return CriticMetadata(
                    score=0.0,
                    feedback="Empty text",
                    issues=["Text must not be empty"]
                )

            # Analyze text quality
            word_count = len(text.split())
            has_punctuation = any(p in text for p in ".!?")

            # Calculate score based on simple metrics
            score = min(1.0, word_count / 50)  # Higher score for longer text
            if has_punctuation:
                score += 0.1  # Bonus for punctuation
            score = min(1.0, score)  # Cap at 1.0

            # Generate feedback
            if score > 0.8:
                feedback = "Excellent text quality"
                issues = []
                suggestions = []
            elif score > 0.5:
                feedback = "Good text quality with minor issues"
                issues = ["Text could be more detailed"]
                suggestions = ["Consider adding more content"]
            else:
                feedback = "Text needs improvement"
                issues = ["Text is too short", "Lacks detail"]
                suggestions = ["Add more content", "Include specific examples"]

            return CriticMetadata(
                score=score,
                feedback=feedback,
                issues=issues,
                suggestions=suggestions
            )

    # Check if it adheres to the protocol
    critic = SimpleCritic()
    assert isinstance(critic, TextCritic)

    # Use the critic
    metadata = critic.critique("This is a test text.")
    print(f"Score: {metadata.score:.2f}")
    print(f"Feedback: {metadata.feedback}")
    ```

    Using with error handling:

    ```python
    from sifaka.critics.base import TextCritic, CriticConfig, CriticMetadata
    import logging
    import time

    logger = logging.getLogger(__name__)

    class RobustCritic:
        def __init__(self):
            self._config = CriticConfig(
                name="robust_critic",
                description="Critic with robust error handling"
            )

        @property
        def config(self) -> CriticConfig:
            return self._config

        def critique(self, text: str) -> CriticMetadata:
            start_time = time.time()
            try:
                if not text or not isinstance(text, str):
                    logger.warning("Invalid input: empty or non-string text")
                    return CriticMetadata(
                        score=0.0,
                        feedback="Invalid input",
                        issues=["Text must be a non-empty string"],
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

                # Perform critique
                # ...

                # Return results
                return CriticMetadata(
                    score=0.8,
                    feedback="Good quality text",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            except Exception as e:
                logger.error(f"Critique error: {e}")
                return CriticMetadata(
                    score=0.0,
                    feedback=f"Error during critique: {str(e)}",
                    issues=["Critique process failed"],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
    ```

    Type Parameters:
        T: The input type (usually str)
        R: The result type
    """

    @property
    def config(self) -> CriticConfig[T]:
        """
        Get critic configuration.

        Returns:
            The critic configuration
        """
        ...

    def critique(self, text: T) -> CriticMetadata[R]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        ...


class BaseCritic(ABC, Generic[T, R]):
    """
    Abstract base class for critics implementing all protocols.

    This class defines the interface for critics and provides common functionality.
    All critics should inherit from this class and implement the required methods.
    It serves as the foundation for all critic implementations in the Sifaka
    framework, providing a consistent interface and behavior.

    ## Architecture

    BaseCritic follows a component-based architecture where functionality is
    delegated to specialized components:

    - **PromptManager**: Creates prompts for validation, critique, improvement, and reflection
    - **ResponseParser**: Parses responses from language models
    - **MemoryManager**: Manages memory for critics (optional)
    - **CritiqueService**: Provides methods for validation, critique, and improvement

    The class implements all three critic protocols:
    - TextValidator: For validating text quality
    - TextImprover: For improving text based on violations
    - TextCritic: For critiquing text and providing feedback

    ## Lifecycle

    1. **Initialization**: Set up with configuration
       - Create with CriticConfig
       - Validate configuration
       - Initialize internal state

    2. **Validation**: Check if text meets quality standards
       - Implement validate() to check text quality
       - Return boolean result

    3. **Critique**: Analyze text and provide feedback
       - Implement critique() to analyze text
       - Return CriticMetadata with score, feedback, issues, and suggestions

    4. **Improvement**: Enhance text based on violations or feedback
       - Implement improve() to enhance text based on violations
       - Implement improve_with_feedback() to enhance text based on feedback
       - Return improved text

    5. **Processing**: Combine critique and improvement
       - Use process() to perform critique and improvement in one operation
       - Return CriticOutput with result, improved text, and metadata

    ## Error Handling

    The class implements these error handling patterns:
    - Input validation with is_valid_text()
    - Configuration validation in _validate_config()
    - Graceful handling of empty or invalid inputs
    - Structured error reporting in metadata

    ## Examples

    Creating a simple critic implementation:

    ```python
    from sifaka.critics.base import BaseCritic, CriticConfig, CriticMetadata

    class SimpleCritic(BaseCritic[str, str]):
        def validate(self, text: str) -> bool:
            if not self.is_valid_text(text):
                return False

            # Simple validation based on length
            return len(text.split()) >= 10

        def critique(self, text: str) -> CriticMetadata[str]:
            if not self.is_valid_text(text):
                return CriticMetadata(
                    score=0.0,
                    feedback="Invalid or empty text",
                    issues=["Text must be a non-empty string"]
                )

            # Simple critique based on length
            word_count = len(text.split())
            score = min(1.0, word_count / 50)

            if score >= 0.8:
                feedback = "Excellent text"
                issues = []
                suggestions = []
            elif score >= 0.5:
                feedback = "Good text, but could be improved"
                issues = ["Text could be more detailed"]
                suggestions = ["Add more content"]
            else:
                feedback = "Text needs improvement"
                issues = ["Text is too short"]
                suggestions = ["Add more content", "Expand on key points"]

            return CriticMetadata(
                score=score,
                feedback=feedback,
                issues=issues,
                suggestions=suggestions
            )

        def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
            if not self.is_valid_text(text):
                return "Default text content"

            # Simple improvement logic
            improved = text
            for violation in violations:
                if "length" in violation.get("rule_id", ""):
                    improved += " Additional content to improve length."

            return improved

        def improve_with_feedback(self, text: str, feedback: str) -> str:
            if not self.is_valid_text(text):
                return "Default text content"

            # Simple improvement based on feedback
            return f"{text}\n\nImproved based on feedback: {feedback}"

    # Create and use the critic
    critic = SimpleCritic(
        CriticConfig(
            name="simple_critic",
            description="A simple text critic implementation"
        )
    )

    text = "This is a test."
    is_valid = critic.validate(text)
    metadata = critic.critique(text)
    improved = critic.improve(text, [{"rule_id": "length", "message": "Too short"}])

    print(f"Valid: {is_valid}")
    print(f"Score: {metadata.score:.2f}")
    print(f"Feedback: {metadata.feedback}")
    print(f"Improved: {improved}")
    ```

    Using the process method:

    ```python
    from sifaka.critics.base import BaseCritic, CriticConfig, CriticResult

    # Using the SimpleCritic from the previous example
    critic = SimpleCritic(
        CriticConfig(
            name="simple_critic",
            description="A simple text critic implementation",
            min_confidence=0.7
        )
    )

    text = "This is a test."
    violations = [{"rule_id": "length", "message": "Text is too short"}]

    output = critic.process(text, violations)

    print(f"Result: {output.result}")
    print(f"Improved text: {output.improved_text}")
    print(f"Score: {output.metadata.score:.2f}")
    print(f"Feedback: {output.metadata.feedback}")

    if output.result == CriticResult.SUCCESS:
        print("Text meets quality standards")
    elif output.result == CriticResult.NEEDS_IMPROVEMENT:
        print("Text has been improved")
    else:
        print("Text could not be improved")
    ```

    Type parameters:
        T: The input type (usually str)
        R: The result type
    """

    def __init__(self, config: CriticConfig[T]) -> None:
        """
        Initialize the critic with configuration.

        Args:
            config: The critic configuration
        """
        self._config = config
        self._validate_config()

    @property
    def config(self) -> CriticConfig[T]:
        """
        Get critic configuration.

        Returns:
            The critic configuration
        """
        return self._config

    def _validate_config(self) -> None:
        """
        Validate critic configuration.

        Raises:
            TypeError: If config is invalid
        """
        # Allow both dataclass and pydantic CriticConfig
        if not hasattr(self.config, "name") or not hasattr(self.config, "description"):
            raise TypeError("config must have name and description attributes")

    def is_valid_text(self, text: Any) -> TypeGuard[T]:
        """
        Type guard to ensure text is a valid string.

        Args:
            text: The text to check

        Returns:
            True if text is a valid string, False otherwise
        """
        return isinstance(text, str) and bool(text.strip())

    @abstractmethod
    def validate(self, text: T) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def improve(self, text: T, violations: List[Dict[str, Any]]) -> T:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def critique(self, text: T) -> CriticMetadata[R]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def improve_with_feedback(self, text: T, feedback: str) -> T:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        pass

    def process(self, text: T, violations: List[Dict[str, Any]]) -> CriticOutput[T, R]:
        """
        Process text and return improved version with metadata.

        This method combines critique and improvement in a single operation,
        providing a complete processing pipeline for text. It first critiques
        the text, then determines if improvement is needed based on the critique
        score, and finally improves the text if necessary.

        ## Lifecycle

        1. **Input Validation**: Check input validity
           - Validate text with is_valid_text()
           - Raise ValueError for empty text

        2. **Critique**: Analyze text quality
           - Call critique() to get quality assessment
           - Get score, feedback, issues, and suggestions

        3. **Decision**: Determine if improvement is needed
           - Compare score against min_confidence threshold
           - Return SUCCESS result if text meets quality standards

        4. **Improvement**: Enhance text if needed
           - Call improve() with text and violations
           - Validate improved text

        5. **Result Creation**: Create standardized output
           - Create CriticOutput with result status
           - Include improved text (or original if improvement failed)
           - Include detailed metadata from critique

        ## Error Handling

        This method implements these error handling patterns:
        - Input validation with is_valid_text()
        - Graceful handling of improvement failures
        - Appropriate result status based on outcome

        ## Examples

        Basic usage:

        ```python
        from sifaka.critics.base import create_basic_critic

        critic = create_basic_critic(
            name="quality_critic",
            description="Checks text quality",
            min_confidence=0.7
        )

        text = "This is a sample text."
        violations = [
            {"rule_id": "length", "message": "Text is too short"}
        ]

        output = critic.process(text, violations)

        print(f"Result: {output.result}")
        print(f"Improved text: {output.improved_text}")
        print(f"Score: {output.metadata.score:.2f}")
        print(f"Feedback: {output.metadata.feedback}")
        ```

        Handling different result types:

        ```python
        from sifaka.critics.base import create_basic_critic, CriticResult

        critic = create_basic_critic(min_confidence=0.8)

        def handle_text(text, violations):
            output = critic.process(text, violations)

            if output.result == CriticResult.SUCCESS:
                print("Text already meets quality standards")
                return output.improved_text

            elif output.result == CriticResult.NEEDS_IMPROVEMENT:
                print(f"Text improved: {output.metadata.feedback}")
                return output.improved_text

            else:  # CriticResult.FAILURE
                print(f"Improvement failed: {output.metadata.feedback}")
                # Fall back to original text
                return text

        # Process different texts
        good_text = "This is a high-quality text with sufficient length and good structure."
        short_text = "Too short."

        result1 = handle_text(good_text, [])
        result2 = handle_text(short_text, [{"rule_id": "length", "message": "Too short"}])
        ```

        Args:
            text: The text to process
            violations: List of rule violations

        Returns:
            CriticOutput containing the result, improved text, and metadata

        Raises:
            ValueError: If text is empty
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        metadata = self.critique(text)

        if metadata.score >= self.config.min_confidence:
            return CriticOutput(result=CriticResult.SUCCESS, improved_text=text, metadata=metadata)

        improved_text = self.improve(text, violations)
        if not self.is_valid_text(improved_text):
            return CriticOutput(
                result=CriticResult.FAILURE,
                improved_text=text,  # Return original if improvement failed
                metadata=metadata,
            )

        return CriticOutput(
            result=CriticResult.NEEDS_IMPROVEMENT, improved_text=improved_text, metadata=metadata
        )


# Common validation patterns
DEFAULT_MIN_CONFIDENCE: Final[float] = 0.7
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_CACHE_SIZE: Final[int] = 100


@overload
def create_critic(
    critic_class: Type[C],
    name: str,
    description: str,
) -> C: ...


@overload
def create_critic(
    critic_class: Type[C],
    name: str,
    description: str,
    min_confidence: float,
    max_attempts: int,
    cache_size: int,
    priority: int,
    cost: float,
    **kwargs: Any,
) -> C: ...


def create_critic(
    critic_class: Type[C],
    name: str = "custom_critic",
    description: str = "Custom critic implementation",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_size: int = DEFAULT_CACHE_SIZE,
    priority: int = 1,
    cost: float = 1.0,
    config: Optional[CriticConfig[Any]] = None,
    **kwargs: Any,
) -> C:
    """
    Factory function to create a critic instance with configuration.

    This function creates a critic instance of the specified class with
    the given configuration parameters. It provides a consistent way to
    instantiate critics with proper configuration, handling the creation
    of the CriticConfig object and setting up the critic with the
    appropriate parameters.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract configuration parameters (name, description, etc.)
       - Extract critic-specific parameters from kwargs
       - Use provided config or create one from parameters

    2. **Configuration Creation**: Create configuration object
       - Create CriticConfig with provided parameters
       - Set up min_confidence, max_attempts, etc.
       - Include critic-specific params

    3. **Instance Creation**: Create critic instance
       - Instantiate the specified critic class
       - Pass configuration and additional kwargs
       - Verify the instance is a BaseCritic
       - Return the configured instance

    ## Error Handling

    This function handles these error cases:
    - Parameter validation (delegated to CriticConfig)
    - Type checking for the created instance
    - Proper extraction of params dictionary

    ## Examples

    Basic usage:

    ```python
    from sifaka.critics.base import create_critic, Critic

    # Create a basic critic
    critic = create_critic(
        critic_class=Critic,
        name="basic_critic",
        description="Basic text quality critic",
        min_confidence=0.7
    )

    # Use the critic
    is_valid = critic.validate("This is a test text")
    metadata = critic.critique("This is a test text")
    print(f"Valid: {is_valid}")
    print(f"Score: {metadata.score:.2f}")
    ```

    Creating with specialized parameters:

    ```python
    from sifaka.critics.base import create_critic
    from sifaka.critics.prompt import PromptCritic

    # Create a prompt critic with specific configuration
    critic = create_critic(
        critic_class=PromptCritic,
        name="prompt_critic",
        description="LLM-based prompt critic",
        min_confidence=0.8,
        max_attempts=5,
        params={
            "system_prompt": "You are an expert editor that improves text.",
            "temperature": 0.7,
            "model_name": "gpt-3.5-turbo"
        },
        llm_provider=model_provider  # Additional parameter for PromptCritic
    )
    ```

    Using a pre-configured config:

    ```python
    from sifaka.critics.base import create_critic, CriticConfig, Critic

    # Create a config
    config = CriticConfig(
        name="custom_critic",
        description="Custom critic with pre-configured settings",
        min_confidence=0.75,
        max_attempts=4,
        params={
            "custom_param1": "value1",
            "custom_param2": "value2"
        }
    )

    # Create a critic with the config
    critic = create_critic(
        critic_class=Critic,
        config=config,
        additional_param="value"  # Additional parameters passed to the critic
    )
    ```

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        An instance of the specified critic class

    Raises:
        TypeError: If the created instance is not a BaseCritic
    """
    # Extract params from kwargs if present
    params: Dict[str, Any] = kwargs.pop("params", {})

    # Use provided config or create one from parameters
    if config is None:
        config = CriticConfig[Any](
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            params=params,
        )

    # Create the critic instance with the config
    result = critic_class(config, **kwargs)
    assert isinstance(result, BaseCritic), f"Expected BaseCritic, got {type(result)}"
    return result


class Critic(BaseCritic[str, str]):
    """
    Default implementation of a text critic.

    This class provides a simple implementation of the BaseCritic interface
    for basic text validation, critique, and improvement.
    """

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise
        """
        if not self.is_valid_text(text):
            return False
        return len(text.strip()) > 0

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        if not violations:
            return text
        # Apply basic improvements based on violations
        improved = text
        for violation in violations:
            if "fix" in violation:
                improved = violation["fix"](improved)
        return improved

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text
        """
        # Simple implementation that just appends the feedback
        return f"{text}\n\nImproved based on feedback: {feedback}"

    def critique(self, text: str) -> CriticMetadata[str]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        if not self.is_valid_text(text):
            return CriticMetadata[str](
                score=0.0,
                feedback="Invalid or empty text",
                issues=["Text must be a non-empty string"],
                suggestions=["Provide non-empty text input"],
            )

        # Basic critique based on text length and content
        score = min(1.0, len(text.strip()) / 100)  # Simple scoring based on length
        feedback = "Text meets basic requirements" if score >= 0.5 else "Text needs improvement"
        issues = [] if score >= 0.5 else ["Text may be too short"]
        suggestions = [] if score >= 0.5 else ["Consider adding more content"]

        return CriticMetadata[str](
            score=score,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions,
        )


def create_basic_critic(
    name: str = "basic_critic",
    description: str = "Basic text critic",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **kwargs: Any,
) -> Critic:
    """
    Create a basic text critic with the given configuration.

    This function creates an instance of the default Critic class with
    the specified configuration parameters. It provides a simple way to
    create a basic critic without needing to specify the critic class.

    ## Lifecycle

    1. **Parameter Processing**: Process input parameters
       - Extract configuration parameters (name, description, etc.)
       - Extract critic-specific parameters from kwargs

    2. **Critic Creation**: Create critic instance
       - Call create_critic with Critic class
       - Pass all parameters to create_critic
       - Return the configured instance

    ## Examples

    Basic usage:

    ```python
    from sifaka.critics.base import create_basic_critic

    # Create a basic critic with default settings
    critic = create_basic_critic()

    # Use the critic
    is_valid = critic.validate("This is a test text")
    metadata = critic.critique("This is a test text")
    print(f"Valid: {is_valid}")
    print(f"Score: {metadata.score:.2f}")
    ```

    Creating with custom settings:

    ```python
    from sifaka.critics.base import create_basic_critic

    # Create a basic critic with custom settings
    critic = create_basic_critic(
        name="custom_basic_critic",
        description="Custom basic text critic",
        min_confidence=0.8,
        max_attempts=5,
        params={
            "custom_param": "value"
        }
    )

    # Process text with violations
    violations = [
        {"rule_id": "length", "message": "Text is too short"}
    ]
    output = critic.process("This is a test text", violations)
    print(f"Result: {output.result}")
    print(f"Improved text: {output.improved_text}")
    ```

    Using with different confidence thresholds:

    ```python
    from sifaka.critics.base import create_basic_critic

    # Create critics with different confidence thresholds
    strict_critic = create_basic_critic(
        name="strict_critic",
        min_confidence=0.9  # High threshold for quality
    )

    lenient_critic = create_basic_critic(
        name="lenient_critic",
        min_confidence=0.5  # Lower threshold for quality
    )

    # Compare results
    text = "This is a test text."
    strict_result = strict_critic.critique(text)
    lenient_result = lenient_critic.critique(text)

    print(f"Strict critic score: {strict_result.score:.2f}")
    print(f"Strict critic passes: {strict_result.score >= strict_critic.config.min_confidence}")

    print(f"Lenient critic score: {lenient_result.score:.2f}")
    print(f"Lenient critic passes: {lenient_result.score >= lenient_critic.config.min_confidence}")
    ```

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional configuration parameters

    Returns:
        A configured Critic instance
    """
    return create_critic(
        critic_class=Critic,
        name=name,
        description=description,
        min_confidence=min_confidence,
        max_attempts=max_attempts,
        **kwargs,
    )


__all__ = [
    "Critic",
    "BaseCritic",
    "CriticConfig",
    "CriticMetadata",
    "CriticOutput",
    "CriticResult",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "create_critic",
    "create_basic_critic",
]
