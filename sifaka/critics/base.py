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
    Union,
    cast,
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
    """Protocol for text validation."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get validator configuration."""
        ...

    def validate(self, text: T) -> bool:
        """Validate text against quality standards."""
        ...


@runtime_checkable
class TextImprover(Protocol[T, R]):
    """Protocol for text improvement."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get improver configuration."""
        ...

    def improve(self, text: T, violations: List[Dict[str, Any]]) -> T:
        """Improve text based on violations."""
        ...


@runtime_checkable
class TextCritic(Protocol[T, R]):
    """Protocol for text critiquing."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get critic configuration."""
        ...

    def critique(self, text: T) -> CriticMetadata[R]:
        """Critique text and provide feedback."""
        ...


class BaseCritic(ABC, Generic[T, R]):
    """
    Abstract base class for critics implementing all protocols.

    This class defines the interface for critics and provides common functionality.
    All critics should inherit from this class and implement the required methods.

    The BaseCritic follows a component-based architecture where functionality is
    delegated to specialized components:
    - PromptManager: Creates prompts for validation, critique, improvement, and reflection
    - ResponseParser: Parses responses from language models
    - MemoryManager: Manages memory for critics (optional)
    - CritiqueService: Provides methods for validation, critique, and improvement

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
    the given configuration parameters.

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
