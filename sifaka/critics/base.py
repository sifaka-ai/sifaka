"""
Base module for critics that provide feedback and validation on prompts.

This module provides the foundational components for the Sifaka critic framework,
including base classes, protocols, and type definitions for text validation,
improvement, and critiquing. Critics work alongside rules to provide a complete
validation and improvement system.

## Integration with Rules

Critics complement rules in the following ways:
- Rules provide binary validation (pass/fail)
- Critics provide nuanced feedback and improvement suggestions
- Rules focus on specific constraints (length, style, content)
- Critics analyze overall text quality and coherence
- Rules can trigger critic improvements when violations occur

## Component Overview

1. **Protocols**
   - `TextValidator`: Interface for text validation
   - `TextImprover`: Interface for text improvement
   - `TextCritic`: Interface for text critiquing

2. **Base Classes**
   - `BaseCritic`: Abstract base class for critics
   - `Critic`: Concrete implementation of BaseCritic

3. **Data Models**
   - `CriticMetadata`: Metadata for critic results
   - `CriticOutput`: Output from critic operations

## Component Lifecycle

### Protocol Lifecycle

1. **Implementation**
   - Create class implementing protocol
   - Define required methods
   - Add configuration property
   - Implement error handling

2. **Verification**
   - Check protocol compliance
   - Validate method signatures
   - Test error handling
   - Verify type hints

3. **Usage**
   - Create instance
   - Configure settings
   - Process text
   - Handle results

### Base Class Lifecycle

1. **Initialization**
   - Validate configuration
   - Set up resources
   - Initialize state
   - Configure logging

2. **Operation**
   - Process text input
   - Validate content
   - Generate feedback
   - Improve text

3. **Cleanup**
   - Release resources
   - Clear state
   - Log results
   - Handle errors

## Error Handling

1. **Input Validation**
   - Empty text checks
   - Type validation
   - Format verification
   - Content validation

2. **Processing Errors**
   - Validation failures
   - Improvement errors
   - Critique failures
   - Resource errors

3. **Recovery Strategies**
   - Default values
   - Fallback methods
   - State preservation
   - Error logging

## Examples

```python
from sifaka.critics.base import BaseCritic
from sifaka.critics.models import CriticConfig, CriticMetadata

class MyCritic(BaseCritic[str, str]):
    def __init__(self, config: CriticConfig):
        super().__init__(config)

    def validate(self, text: str) -> bool:
        # Implementation
        return True

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        # Implementation
        return "Improved text"

    def critique(self, text: str) -> CriticMetadata[str]:
        # Implementation
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

# Import the Pydantic models
from .models import CriticConfig, CriticMetadata

# Default configuration values
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CACHE_SIZE = 100

# Input and output type variables
T = TypeVar("T")  # Input type (usually str)
R = TypeVar("R")  # Result type
C = TypeVar("C", bound="BaseCritic")  # Critic type


class CriticResult(str, Enum):
    """Enumeration of possible critic results.

    This enum defines the possible outcomes of critic operations:
    - SUCCESS: Operation completed successfully
    - NEEDS_IMPROVEMENT: Text needs improvement
    - FAILURE: Operation failed

    Examples:
        ```python
        result = CriticResult.SUCCESS
        if result == CriticResult.NEEDS_IMPROVEMENT:
            print("Text needs improvement")
        ```
    """

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticMetadata(Generic[R]):
    """Immutable metadata for critic results.

    This class defines the metadata structure for critic results,
    including score, feedback, issues, and suggestions.

    ## Lifecycle Management

    1. **Creation**
       - Set metadata values
       - Validate parameters
       - Create immutable instance

    2. **Usage**
       - Access metadata values
       - Create modified instances
       - Validate data

    3. **Validation**
       - Check score range
       - Verify required fields
       - Ensure immutability

    ## Error Handling

    1. **Validation Errors**
       - Invalid score range
       - Negative processing time
       - Invalid attempt number
       - Missing required fields

    2. **Recovery**
       - Default values
       - Parameter adjustment
       - Error messages

    Examples:
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
    from sifaka.critics.base import TextValidator
    from sifaka.critics.models import CriticConfig
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
            from sifaka.utils.text import is_empty_text
            if is_empty_text(text):
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
    from sifaka.critics.base import TextValidator
    from sifaka.critics.models import CriticConfig
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
    from sifaka.critics.base import TextImprover
    from sifaka.critics.models import CriticConfig
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
    from sifaka.critics.base import TextImprover
    from sifaka.critics.models import CriticConfig
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
    from sifaka.critics.base import TextCritic, CriticMetadata
    from sifaka.critics.models import CriticConfig
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
            from sifaka.utils.text import is_empty_text
            if is_empty_text(text):
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
    from sifaka.critics.base import TextCritic, CriticMetadata
    from sifaka.critics.models import CriticConfig
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

    This class provides a base implementation for critics that validate,
    improve, and critique text. It implements common functionality and
    defines abstract methods that must be implemented by subclasses.

    ## Lifecycle Management

    1. **Initialization**
       - Validate configuration
       - Set up resources
       - Initialize state
       - Configure logging

    2. **Operation**
       - Process text input
       - Validate content
       - Generate feedback
       - Improve text

    3. **Cleanup**
       - Release resources
       - Clear state
       - Log results
       - Handle errors

    ## Error Handling

    1. **Input Validation**
       - Empty text checks
       - Type validation
       - Format verification
       - Content validation

    2. **Processing Errors**
       - Validation failures
       - Improvement errors
       - Critique failures
       - Resource errors

    3. **Recovery Strategies**
       - Default values
       - Fallback methods
       - State preservation
       - Error logging

    ## Examples

    Implementing a custom critic:

    ```python
    from sifaka.critics.base import BaseCritic
    from sifaka.critics.models import CriticConfig
    from sifaka.critics.base import CriticMetadata

    class MyCritic(BaseCritic[str, str]):
        def __init__(self, config: CriticConfig):
            super().__init__(config)

        def validate(self, text: str) -> bool:
            # Implementation
            return True

        def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
            # Implementation
            return "Improved text"

        def critique(self, text: str) -> CriticMetadata[str]:
            # Implementation
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
    """

    def __init__(self, config: CriticConfig) -> None:
        """
        Initialize a BaseCritic instance.

        Args:
            config: Configuration for the critic

        Raises:
            ValueError: If configuration is invalid
        """
        self._config = config
        self._validate_config()

    @property
    def config(self) -> CriticConfig:
        """
        Get critic configuration.

        Returns:
            The critic configuration
        """
        return self._config

    def _validate_config(self) -> None:
        """
        Validate critic configuration.

        This method checks if the configuration is valid and raises
        appropriate exceptions if not.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self._config, CriticConfig):
            raise ValueError("config must be a CriticConfig instance")

    def is_valid_text(self, text: Any) -> TypeGuard[T]:
        """
        Check if text is valid for this critic.

        Args:
            text: The text to check

        Returns:
            True if the text is valid, False otherwise
        """
        if not isinstance(text, str):
            return False

        from sifaka.utils.text import is_empty_text

        return not is_empty_text(text)

    @abstractmethod
    def validate(self, text: T) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is invalid
        """
        ...

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
            ValueError: If text or violations are invalid
        """
        ...

    @abstractmethod
    def critique(self, text: T) -> CriticMetadata[R]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is invalid
        """
        ...

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
            ValueError: If text or feedback is invalid
        """
        ...

    def process(self, text: T, violations: List[Dict[str, Any]]) -> CriticOutput[T, R]:
        """
        Process text and violations.

        This method combines validation, improvement, and critique
        operations into a single workflow.

        Args:
            text: The text to process
            violations: List of rule violations

        Returns:
            CriticOutput containing the results

        Raises:
            ValueError: If text or violations are invalid
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        # Validate text
        is_valid = self.validate(text)

        # Improve text
        improved_text = self.improve(text, violations)

        # Critique text
        metadata = self.critique(improved_text)

        # Determine result
        if is_valid and metadata.score >= self.config.min_confidence:
            result = CriticResult.SUCCESS
        elif is_valid:
            result = CriticResult.NEEDS_IMPROVEMENT
        else:
            result = CriticResult.FAILURE

        return CriticOutput(result=result, improved_text=improved_text, metadata=metadata)


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

    This function creates a critic instance using either a provided
    configuration or parameters to create one.

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        config: Optional critic configuration
        **kwargs: Additional keyword arguments

    Returns:
        A critic instance

    Examples:
        ```python
        from sifaka.critics.base import create_critic
        from sifaka.critics.models import CriticConfig

        # Create with configuration
        critic = create_critic(
            MyCritic,
            config=CriticConfig(
                name="my_critic",
                description="A custom critic"
            )
        )

        # Create with parameters
        critic = create_critic(
            MyCritic,
            name="my_critic",
            description="A custom critic",
            min_confidence=0.8,
            max_attempts=3
        )
        ```
    """
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

    return critic_class(config)


class Critic(BaseCritic[str, str]):
    """
    Concrete implementation of BaseCritic for string text.

    This class provides a concrete implementation of BaseCritic
    specifically for string text input and output.

    ## Lifecycle Management

    1. **Initialization**
       - Validate configuration
       - Set up resources
       - Initialize state
       - Configure logging

    2. **Operation**
       - Process text input
       - Validate content
       - Generate feedback
       - Improve text

    3. **Cleanup**
       - Release resources
       - Clear state
       - Log results
       - Handle errors

    ## Error Handling

    1. **Input Validation**
       - Empty text checks
       - Type validation
       - Format verification
       - Content validation

    2. **Processing Errors**
       - Validation failures
       - Improvement errors
       - Critique failures
       - Resource errors

    3. **Recovery Strategies**
       - Default values
       - Fallback methods
       - State preservation
       - Error logging

    ## Examples

    Using the Critic class:

    ```python
    from sifaka.critics.base import Critic, CriticConfig

    # Create a critic
    critic = Critic(CriticConfig(
        name="text_critic",
        description="Analyzes and improves text"
    ))

    # Use the critic
    text = "This is a test."
    is_valid = critic.validate(text)
    improved = critic.improve(text, [])
    feedback = critic.critique(text)
    ```
    """

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is invalid
        """
        if not self.is_valid_text(text):
            return False

        # Basic validation
        from sifaka.utils.text import is_empty_text

        if is_empty_text(text):
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

        Raises:
            ValueError: If text or violations are invalid
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        # Basic improvement
        improved = text.strip()
        for violation in violations:
            rule_id = violation.get("rule_id", "unknown")
            message = violation.get("message", "")

            if rule_id == "capitalization" and "missing" in message:
                improved = improved.capitalize()
            elif rule_id == "whitespace" and "extra" in message:
                improved = " ".join(improved.split())

        return improved

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text or feedback is invalid
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("feedback must be a non-empty string")

        # Basic improvement based on feedback
        improved = text.strip()
        if "capitalize" in feedback.lower():
            improved = improved.capitalize()
        if "trim" in feedback.lower():
            improved = improved.strip()

        return improved

    def critique(self, text: str) -> CriticMetadata[str]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is invalid
        """
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        # Basic critique
        text_length = len(text)
        has_punctuation = any(p in text for p in ".!?")
        is_capitalized = text[0].isupper() if text else False

        # Calculate score
        score = min(1.0, text_length / 100)  # Higher score for longer text
        if has_punctuation:
            score += 0.1
        if is_capitalized:
            score += 0.1
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
    Create a basic critic instance.

    This function creates a basic critic instance with default settings.

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional keyword arguments

    Returns:
        A basic critic instance

    Examples:
        ```python
        from sifaka.critics.base import create_basic_critic

        # Create a basic critic
        critic = create_basic_critic(
            name="my_critic",
            description="A basic critic"
        )

        # Use the critic
        text = "This is a test."
        is_valid = critic.validate(text)
        improved = critic.improve(text, [])
        feedback = critic.critique(text)
        ```
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
    "CriticResult",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "create_critic",
    "create_basic_critic",
]
