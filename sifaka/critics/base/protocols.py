"""
Protocol definitions for critics.

This module defines the protocol interfaces for text validation, improvement, and critiquing
components in the Sifaka critic framework.

## Overview
The module provides protocol definitions that establish a common contract for critic behavior,
enabling better modularity and extensibility. These protocols define the expected interfaces
for components that validate, improve, and critique text.

## Components
1. **TextValidator**: Protocol for text validation
2. **TextImprover**: Protocol for text improvement
3. **TextCritic**: Protocol for text critiquing

## Usage Examples
```python
from sifaka.critics.base.protocols import TextValidator
from sifaka.utils.config.critics import CriticConfig

class LengthValidator:
    def __init__(self, min_length: int = 10, max_length: int = 1000) -> None:
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
        min_length = self.config.params.get("min_length", 10) if params else ""
        max_length = self.config.params.get("max_length", 1000) if params else ""
        return min_length <= text_length <= max_length

# Check if it adheres to the protocol
validator = LengthValidator(min_length=20, max_length=500)
assert isinstance(validator, TextValidator)
```

## Error Handling
The protocols define interfaces that implementations should follow for proper error handling,
including handling of empty or invalid inputs, validation failures, and resource issues.
"""

from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable

from sifaka.utils.config.critics import CriticConfig
from sifaka.critics.base.metadata import CriticMetadata

# Input and output type variables with variance annotations
T = TypeVar("T", contravariant=True)  # Input type (usually str) - contravariant
R = TypeVar("R", covariant=True)  # Result type - covariant


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
    from sifaka.critics.base.protocols import TextValidator
    from sifaka.utils.config.critics import CriticConfig

    class LengthValidator:
        def __init__(self, min_length: int = 10, max_length: int = 1000) -> None:
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
            min_length = self.config.params.get("min_length", 10) if params else ""
            max_length = self.config.params.get("max_length", 1000) if params else ""
            return min_length <= text_length <= max_length

    # Check if it adheres to the protocol
    validator = LengthValidator(min_length=20, max_length=500)
    assert isinstance(validator, TextValidator)

    # Use the validator
    is_valid = validator.validate("This is a test text") if validator else ""
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
    from sifaka.critics.base.protocols import TextImprover
    from sifaka.utils.config.critics import CriticConfig
    from typing import Dict, List, Any

    class SimpleImprover:
        def __init__(self) -> None:
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
                rule_id = violation.get("rule_id", "unknown") if violation else ""
                message = violation.get("message", "") if violation else ""

                if rule_id == "length" and "too_short" in message:
                    improved += " Additional content to increase length."
                elif rule_id == "style" and "capitalization" in message:
                    improved = improved.capitalize() if improved else ""

            return improved
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

    def improve(self, text: T, violations: List[Dict[str, Any]]) -> R:
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
    from sifaka.critics.base.protocols import TextCritic, CriticMetadata
    from sifaka.utils.config.critics import CriticConfig

    class SimpleCritic:
        def __init__(self) -> None:
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
            word_count = len(text.split() if text else "")
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

    def critique(self, text: T) -> R:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        ...
