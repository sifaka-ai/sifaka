"""
Base Module for Critics

A comprehensive module providing the foundational components for the Sifaka critic framework,
including base classes, protocols, and type definitions for text validation, improvement, and critiquing.

## Overview
This module serves as the core foundation for the critic system, defining interfaces and base implementations
that enable text validation, improvement, and critiquing functionality. Critics work alongside rules to provide
a complete validation and improvement system.

## Components
1. **Protocols**
   - TextValidator: Interface for text validation
   - TextImprover: Interface for text improvement
   - TextCritic: Interface for text critiquing

2. **Base Classes**
   - BaseCritic: Abstract base class for critics
   - Critic: Concrete implementation of BaseCritic

3. **Data Models**
   - CriticMetadata: Metadata for critic results
   - CriticOutput: Output from critic operations

## Usage Examples
```python
from sifaka.critics.base import BaseCritic
from sifaka.critics.models import CriticConfig, CriticMetadata

class MyCritic(BaseCritic[str, str]):
    def __init__(self, config: CriticConfig):
        super().__init__(config)

    def validate(self, text: str) -> bool:
        return True

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        return "Improved text"

    def critique(self, text: str) -> CriticMetadata[str]:
        return CriticMetadata(
            score=0.8,
            feedback="Good text",
            issues=[],
            suggestions=[]
        )

# Create and use the critic
critic = MyCritic(CriticConfig(
    name="my_critic",
    description="A custom critic implementation"
))
text = "This is a test."
is_valid = critic.validate(text)
improved = critic.improve(text, [])
feedback = critic.critique(text)
```

## Error Handling
The module implements comprehensive error handling for:
1. Input Validation
   - Empty text checks
   - Type validation
   - Format verification
   - Content validation

2. Processing Errors
   - Validation failures
   - Improvement errors
   - Critique failures
   - Resource errors

3. Recovery Strategies
   - Default values
   - Fallback methods
   - State preservation
   - Error logging

## Configuration
The module supports configuration through CriticConfig objects, which can specify:
- Name and description
- Minimum confidence threshold
- Maximum improvement attempts
- Cache size
- Priority and cost
- Custom parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
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

# Import the interfaces
from .interfaces.critic import TextValidator, TextImprover, TextCritic, CritiqueResult

# Import the Pydantic models
from .config import CriticConfig, CriticMetadata

# Default configuration values
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CACHE_SIZE = 100

# Input and output type variables
T = TypeVar("T")  # Input type (usually str)
R = TypeVar("R")  # Result type
C = TypeVar("C", bound="BaseCritic")  # Critic type


class CriticResultEnum(str, Enum):
    """
    Enumeration of possible critic results.

    This enum defines the possible outcomes of critic operations, providing
    a standardized way to represent the success or failure of critic operations.

    ## Overview
    The enum provides three possible states:
    - SUCCESS: Operation completed successfully
    - NEEDS_IMPROVEMENT: Text needs improvement
    - FAILURE: Operation failed

    ## Usage Examples
    ```python
    result = CriticResultEnum.SUCCESS
    if result == CriticResultEnum.NEEDS_IMPROVEMENT:
        print("Text needs improvement")
    elif result == CriticResultEnum.FAILURE:
        print("Operation failed")
    ```

    ## Error Handling
    The enum values are immutable and type-safe, ensuring consistent
    representation of operation results throughout the system.
    """

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticMetadata(Generic[R]):
    """
    Immutable metadata for critic results.

    This class defines the metadata structure for critic results, providing
    a standardized way to store and access information about critic operations.

    ## Overview
    The metadata includes:
    - Score: Quality score of the text
    - Feedback: Detailed feedback about the text
    - Issues: List of identified issues
    - Suggestions: List of improvement suggestions
    - Processing time and attempt information
    - Additional custom metadata

    ## Usage Examples
    ```python
    metadata = CriticMetadata(
        score=0.8,
        feedback="Good text",
        issues=["Needs more detail"],
        suggestions=["Add examples"]
    )

    # Create modified metadata
    new_metadata = metadata.with_extra(
        processing_time_ms=100.0,
        attempt_number=2
    )
    ```

    ## Error Handling
    The class implements validation for:
    - Score range (0.0 to 1.0)
    - Required fields
    - Processing time (non-negative)
    - Attempt number (positive integer)

    Attributes:
        score (float): Quality score between 0.0 and 1.0
        feedback (str): Detailed feedback about the text
        issues (List[str]): List of identified issues
        suggestions (List[str]): List of improvement suggestions
        attempt_number (int): Number of improvement attempts
        processing_time_ms (float): Processing time in milliseconds
        extra (Dict[str, Any]): Additional custom metadata
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
        """Create a new metadata with additional extra data.

        This method creates a new CriticMetadata instance with additional
        extra data while preserving other values.

        Args:
            **kwargs: Additional extra data

        Returns:
            New CriticMetadata instance

        Examples:
            ```python
            metadata = CriticMetadata(
                score=0.8,
                feedback="Good text"
            )
            new_metadata = metadata.with_extra(
                processing_time_ms=100.0,
                attempt_number=2
            )
            ```
        """
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
    """Immutable output from a critic.

    This class defines the structure for critic outputs, including
    the result, improved text, and metadata.

    ## Lifecycle Management

    1. **Creation**
       - Set output values
       - Validate parameters
       - Create immutable instance

    2. **Usage**
       - Access output values
       - Process results
       - Handle metadata

    3. **Validation**
       - Check result type
       - Verify text format
       - Validate metadata

    ## Error Handling

    1. **Validation Errors**
       - Invalid result type
       - Empty text
       - Invalid metadata
       - Missing required fields

    2. **Recovery**
       - Default values
       - Error messages
       - State preservation

    Examples:
        ```python
        output = CriticOutput(
            result=CriticResult.SUCCESS,
            improved_text="Improved text",
            metadata=CriticMetadata(
                score=0.8,
                feedback="Good improvement"
            )
        )
        ```
    """

    result: CriticResultEnum
    improved_text: T
    metadata: CriticMetadata[R]


@runtime_checkable
class TextValidator(Protocol[T]):
    """
    Protocol for text validation.

    This protocol defines the interface for components that validate text
    against quality standards, providing a standardized way to implement
    text validation functionality.

    ## Overview
    The protocol requires:
    - A config property to expose configuration
    - A validate() method to check text quality
    - Type-safe implementation with proper error handling

    ## Usage Examples
    ```python
    from sifaka.critics.base import TextValidator
    from sifaka.critics.models import CriticConfig

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
            if not text:
                return False
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

    ## Error Handling
    Implementations should handle:
    - Empty or invalid text inputs
    - Validation failures
    - Resource availability issues
    - Configuration errors

    Type Parameters:
        T: The input type (usually str)
    """

    @property
    def config(self) -> CriticConfig:
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
    based on rule violations, providing a standardized way to implement
    text improvement functionality.

    ## Overview
    The protocol requires:
    - A config property to expose configuration
    - An improve() method to enhance text quality
    - Type-safe implementation with proper error handling
    - Support for handling rule violations

    ## Usage Examples
    ```python
    from sifaka.critics.base import TextImprover
    from sifaka.critics.models import CriticConfig
    from typing import Dict, List, Any

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
            if not text:
                return "Default text content"

            improved = text
            for violation in violations:
                rule_id = violation.get("rule_id", "unknown")
                message = violation.get("message", "")

                if rule_id == "length" and "too_short" in message:
                    improved += " Additional content to increase length."
                elif rule_id == "style" and "capitalization" in message:
                    improved = improved.capitalize()

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

    ## Error Handling
    Implementations should handle:
    - Empty or invalid text inputs
    - Invalid violation formats
    - Improvement failures
    - Resource availability issues
    - Configuration errors

    Type Parameters:
        T: The input type (usually str)
        R: The result type
    """

    @property
    def config(self) -> CriticConfig:
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
    and provide detailed feedback, providing a standardized way to implement
    text analysis functionality.

    ## Overview
    The protocol requires:
    - A config property to expose configuration
    - A critique() method to analyze text quality
    - Type-safe implementation with proper error handling
    - Support for generating detailed feedback

    ## Usage Examples
    ```python
    from sifaka.critics.base import TextCritic, CriticMetadata
    from sifaka.critics.models import CriticConfig

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
            if not text:
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

    ## Error Handling
    Implementations should handle:
    - Empty or invalid text inputs
    - Critique failures
    - Resource availability issues
    - Configuration errors
    - Processing time tracking

    Type Parameters:
        T: The input type (usually str)
        R: The result type
    """

    @property
    def config(self) -> CriticConfig:
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
    Abstract base class for critics.

    This class provides a foundation for implementing critics that can validate,
    improve, and critique text. It implements common functionality and enforces
    a consistent interface for all critic implementations.

    ## Overview
    The class provides:
    - Configuration management
    - State management
    - Resource initialization and cleanup
    - Common validation and error handling
    - Abstract methods for critic-specific functionality

    ## Usage Examples
    ```python
    from sifaka.critics.base import BaseCritic
    from sifaka.critics.models import CriticConfig, CriticMetadata

    class MyCritic(BaseCritic[str, str]):
        def __init__(self, config: CriticConfig):
            super().__init__(config)

        def validate(self, text: str) -> bool:
            return len(text) > 0

        def improve(self, text: str, feedback: Optional[str] = None) -> str:
            return text.upper()

        def critique(self, text: str) -> CriticMetadata[str]:
            return CriticMetadata(
                score=0.8,
                feedback="Good text",
                issues=[],
                suggestions=[]
            )

        def improve_with_feedback(self, text: str, feedback: str) -> str:
            return self.improve(text, feedback)

    # Create and use the critic
    critic = MyCritic(CriticConfig(
        name="my_critic",
        description="A custom critic implementation"
    ))
    text = "This is a test."
    is_valid = critic.validate(text)
    improved = critic.improve(text)
    feedback = critic.critique(text)
    ```

    ## Error Handling
    The class implements:
    - Configuration validation
    - Resource management
    - State validation
    - Type checking
    - Error recovery strategies

    Type Parameters:
        T: The input type (usually str)
        R: The result type
    """

    def __init__(self, config: CriticConfig) -> None:
        """
        Initialize the critic.

        Args:
            config: The critic configuration
        """
        self._config = config
        self._validate_config()
        self.initialize()

    @property
    def config(self) -> CriticConfig:
        """
        Get critic configuration.

        Returns:
            The critic configuration
        """
        return self._config

    @property
    def name(self) -> str:
        """
        Get critic name.

        Returns:
            The critic name
        """
        return self._config.name

    @property
    def description(self) -> str:
        """
        Get critic description.

        Returns:
            The critic description
        """
        return self._config.description

    def update_config(self, config: CriticConfig) -> None:
        """
        Update critic configuration.

        Args:
            config: The new configuration
        """
        self._config = config
        self._validate_config()

    def initialize(self) -> None:
        """
        Initialize critic resources.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up critic resources.
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate critic configuration.
        """
        if not self._config.name:
            raise ValueError("Critic name is required")
        if not self._config.description:
            raise ValueError("Critic description is required")

    def is_valid_text(self, text: Any) -> TypeGuard[T]:
        """
        Check if text is valid.

        Args:
            text: The text to check

        Returns:
            True if text is valid, False otherwise
        """
        return isinstance(text, str) and bool(text)

    @abstractmethod
    def validate(self, text: T) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if text is valid, False otherwise
        """
        pass

    @abstractmethod
    def improve(self, text: T, feedback: Optional[str] = None) -> T:
        """
        Improve text.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            The improved text
        """
        pass

    @abstractmethod
    def critique(self, text: T) -> CriticMetadata[R]:
        """
        Critique text.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        pass

    @abstractmethod
    def improve_with_feedback(self, text: T, feedback: str) -> T:
        """
        Improve text with feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide improvement

        Returns:
            The improved text
        """
        pass

    def process(self, text: T, feedback: Optional[str] = None) -> CriticOutput[T, R]:
        """
        Process text through the critic pipeline.

        Args:
            text: The text to process
            feedback: Optional feedback to guide improvement

        Returns:
            CriticOutput containing the processing results
        """
        if not self.is_valid_text(text):
            return CriticOutput(
                result=CriticResultEnum.FAILURE,
                improved_text=text,
                metadata=CriticMetadata(
                    score=0.0, feedback="Invalid text", issues=["Text must be a non-empty string"]
                ),
            )

        try:
            # Validate text
            is_valid = self.validate(text)
            if not is_valid:
                return CriticOutput(
                    result=CriticResultEnum.NEEDS_IMPROVEMENT,
                    improved_text=text,
                    metadata=CriticMetadata(
                        score=0.0,
                        feedback="Text needs improvement",
                        issues=["Text failed validation"],
                    ),
                )

            # Improve text if feedback provided
            improved_text = text
            if feedback:
                improved_text = self.improve_with_feedback(text, feedback)

            # Get critique
            metadata = self.critique(improved_text)

            return CriticOutput(
                result=CriticResultEnum.SUCCESS, improved_text=improved_text, metadata=metadata
            )
        except Exception as e:
            return CriticOutput(
                result=CriticResultEnum.FAILURE,
                improved_text=text,
                metadata=CriticMetadata(
                    score=0.0, feedback=f"Error: {str(e)}", issues=["Processing failed"]
                ),
            )


@overload
def create_critic(
    critic_class: Type[C],
    name: str,
    description: str,
    config: CriticConfig,
) -> C:
    """
    Create a critic with a configuration.

    This function creates a critic instance using the provided class
    and configuration.

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        config: Configuration for the critic

    Returns:
        A critic instance

    Examples:
        ```python
        from sifaka.critics.base import create_critic
        from sifaka.critics.models import CriticConfig

        config = CriticConfig(
            name="my_critic",
            description="A custom critic"
        )
        critic = create_critic(MyCritic, config=config)
        ```
    """
    ...


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
) -> C:
    """
    Create a critic with parameters.

    This function creates a critic instance using the provided class
    and parameters.

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        **kwargs: Additional keyword arguments

    Returns:
        A critic instance

    Examples:
        ```python
        from sifaka.critics.base import create_critic

        critic = create_critic(
            MyCritic,
            name="my_critic",
            description="A custom critic",
            min_confidence=0.8,
            max_attempts=3
        )
        ```
    """
    ...


def create_critic(
    critic_class: Type[C],
    name: str = "custom_critic",
    description: str = "Custom critic implementation",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_size: int = DEFAULT_CACHE_SIZE,
    priority: int = 1,
    cost: float = 1.0,
    config: Optional[CriticConfig] = None,
    **kwargs: Any,
) -> C:
    """
    Create a critic instance.

    This function provides a factory method for creating critic instances
    with standardized configuration and error handling.

    ## Overview
    The function:
    - Creates a critic instance with the specified configuration
    - Handles both direct config and parameter-based configuration
    - Provides default values for common parameters
    - Validates configuration before creating the critic

    ## Usage Examples
    ```python
    from sifaka.critics.base import create_critic, Critic

    # Create a critic with default settings
    critic = create_critic(Critic)

    # Create a critic with custom settings
    critic = create_critic(
        Critic,
        name="custom_critic",
        description="A custom critic implementation",
        min_confidence=0.8,
        max_attempts=5,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Create a critic with a config object
    from sifaka.critics.models import CriticConfig
    config = CriticConfig(
        name="config_critic",
        description="Critic created with config object",
        min_confidence=0.9,
        max_attempts=3,
        cache_size=100,
        priority=1,
        cost=1.0
    )
    critic = create_critic(Critic, config=config)
    ```

    ## Error Handling
    The function handles:
    - Invalid critic class
    - Invalid configuration
    - Missing required parameters
    - Type validation
    - Configuration validation

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache for memoization
        priority: Priority of the critic
        cost: Cost of running the critic
        config: Optional critic configuration
        **kwargs: Additional configuration parameters

    Returns:
        An instance of the specified critic class

    Raises:
        ValueError: If configuration is invalid
        TypeError: If critic_class is invalid
    """
    # Create config if not provided
    if config is None:
        config = CriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            params=kwargs,
        )

    # Validate critic class
    if not issubclass(critic_class, BaseCritic):
        raise TypeError(f"critic_class must be a subclass of BaseCritic, got {critic_class}")

    # Create and return critic instance
    return critic_class(config)


class Critic(BaseCritic[str, str]):
    """
    Concrete implementation of BaseCritic for string-based text processing.

    This class provides a complete implementation of the critic interface
    for processing string-based text, with built-in validation, improvement,
    and critique functionality.

    ## Overview
    The class provides:
    - String-based text validation
    - Text improvement with feedback
    - Text critique with detailed analysis
    - Error handling and recovery
    - Configuration management

    ## Usage Examples
    ```python
    from sifaka.critics.base import Critic
    from sifaka.critics.models import CriticConfig

    # Create a critic
    critic = Critic(CriticConfig(
        name="text_critic",
        description="Analyzes and improves text quality"
    ))

    # Process text
    text = "This is a test text."
    result = critic.process(text)

    # Check results
    if result.result == CriticResultEnum.SUCCESS:
        print(f"Improved text: {result.improved_text}")
        print(f"Score: {result.metadata.score:.2f}")
        print(f"Feedback: {result.metadata.feedback}")
    ```

    ## Error Handling
    The class implements:
    - Input validation
    - Error recovery
    - State management
    - Resource cleanup
    - Detailed error reporting
    """

    def validate(self, text: str) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if text is valid, False otherwise
        """
        if not self.is_valid_text(text):
            return False

        # Basic validation rules
        if len(text) < 10:
            return False
        if not any(c.isupper() for c in text):
            return False
        if not any(c.islower() for c in text):
            return False

        return True

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        if not self.is_valid_text(text):
            return text

        improved = text

        # Apply improvements based on violations
        for violation in violations:
            rule_id = violation.get("rule_id", "unknown")
            message = violation.get("message", "")

            if rule_id == "length" and "too_short" in message:
                improved += " Additional content to increase length."
            elif rule_id == "style" and "capitalization" in message:
                improved = improved.capitalize()
            elif rule_id == "grammar" and "missing_punctuation" in message:
                if not improved.endswith((".", "!", "?")):
                    improved += "."

        return improved

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text with feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide improvement

        Returns:
            The improved text
        """
        if not self.is_valid_text(text):
            return text

        improved = text

        # Apply improvements based on feedback
        if "too short" in feedback.lower():
            improved += " Additional content to increase length."
        if "capitalization" in feedback.lower():
            improved = improved.capitalize()
        if "punctuation" in feedback.lower():
            if not improved.endswith((".", "!", "?")):
                improved += "."

        return improved

    def critique(self, text: str) -> CriticMetadata[str]:
        """
        Critique text.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        if not self.is_valid_text(text):
            return CriticMetadata(
                score=0.0, feedback="Invalid text", issues=["Text must be a non-empty string"]
            )

        # Analyze text quality
        word_count = len(text.split())
        has_punctuation = any(p in text for p in ".!?")
        has_capitalization = any(c.isupper() for c in text)
        has_lowercase = any(c.islower() for c in text)

        # Calculate score based on metrics
        score = 0.0
        if word_count >= 10:
            score += 0.3
        if has_punctuation:
            score += 0.2
        if has_capitalization:
            score += 0.2
        if has_lowercase:
            score += 0.3

        # Generate feedback
        issues = []
        suggestions = []

        if word_count < 10:
            issues.append("Text is too short")
            suggestions.append("Add more content")
        if not has_punctuation:
            issues.append("Missing punctuation")
            suggestions.append("Add appropriate punctuation")
        if not has_capitalization:
            issues.append("Missing capitalization")
            suggestions.append("Capitalize appropriate words")
        if not has_lowercase:
            issues.append("All uppercase text")
            suggestions.append("Use lowercase where appropriate")

        feedback = "Good text quality" if score >= 0.8 else "Text needs improvement"

        return CriticMetadata(
            score=score, feedback=feedback, issues=issues, suggestions=suggestions
        )


def create_basic_critic(
    name: str = "basic_critic",
    description: str = "Basic text critic",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **kwargs: Any,
) -> Critic:
    """
    Create a basic text critic.

    This function creates a basic critic instance with sensible defaults
    for common text processing tasks.

    ## Overview
    The function:
    - Creates a Critic instance with basic text processing capabilities
    - Provides default values for common parameters
    - Handles basic text validation and improvement
    - Includes simple critique functionality

    ## Usage Examples
    ```python
    from sifaka.critics.base import create_basic_critic

    # Create a basic critic with default settings
    critic = create_basic_critic()

    # Create a basic critic with custom settings
    critic = create_basic_critic(
        name="custom_basic_critic",
        description="A custom basic critic",
        min_confidence=0.8,
        max_attempts=5
    )

    # Use the critic
    text = "This is a test text."
    result = critic.process(text)
    print(f"Score: {result.metadata.score:.2f}")
    print(f"Feedback: {result.metadata.feedback}")
    ```

    ## Error Handling
    The function handles:
    - Invalid configuration
    - Missing required parameters
    - Type validation
    - Configuration validation

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional configuration parameters

    Returns:
        A basic Critic instance

    Raises:
        ValueError: If configuration is invalid
    """
    return create_critic(
        Critic,
        name=name,
        description=description,
        min_confidence=min_confidence,
        max_attempts=max_attempts,
        **kwargs,
    )


__all__ = [
    "Critic",
    "BaseCritic",
    "CriticMetadata",
    "CriticOutput",
    "CriticResultEnum",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "create_critic",
    "create_basic_critic",
]
