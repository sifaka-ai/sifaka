"""
Abstract base class for critics.

This module defines the abstract base class for critics in the Sifaka framework,
providing a foundation for implementing text validation, improvement, and critiquing
functionality.

## Overview
The module provides the BaseCritic abstract base class, which serves as the foundation
for all critic implementations. It defines the core interface and common functionality
that all critics must implement.

## Components
1. **BaseCritic**: Abstract base class for critics

## Usage Examples
```python
from sifaka.critics.base.abstract import BaseCritic
from sifaka.utils.config.critics import CriticConfig
from sifaka.core.base import BaseResult

class MyCritic(BaseCritic[str]):
    def __init__(self, name: str, description: str, config: Optional[CriticConfig] = None) -> None:
        super().__init__(name, description, config)

    def validate(self, text: str) -> bool:
        return len(text) > 0

    def improve(self, text: str, feedback: Optional[str] = None) -> str:
        return text.upper() if text else ""

    def critique(self, text: str) -> BaseResult:
        return BaseResult(
            passed=True,
            message="Good text",
            score=0.8,
            issues=[],
            suggestions=[]
        )

# Create and use the critic
critic = MyCritic(
    name="my_critic",
    description="A custom critic implementation"
)
text = "This is a test."
is_valid = critic.validate(text) if critic else ""
improved = critic.improve(text) if critic else ""
feedback = critic.critique(text) if critic else ""
```

## Error Handling
The class implements comprehensive error handling for:
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
"""

from abc import abstractmethod
import time
from typing import Any, Generic, Optional, TypeVar, cast
from sifaka.core.base import BaseComponent, BaseResult, BaseConfig
from sifaka.utils.config.critics import CriticConfig
from sifaka.utils.errors import safely_execute_component_operation as safely_execute_critic
from sifaka.utils.errors import CriticError

T = TypeVar("T")


class BaseCritic(BaseComponent[T, BaseResult], Generic[T]):
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
    from sifaka.critics.base.abstract import BaseCritic
    from sifaka.utils.config.critics import CriticConfig
    from sifaka.core.base import BaseResult

    class MyCritic(BaseCritic[str]):
        def __init__(self, name: str, description: str, config: Optional[CriticConfig] = None) -> None:
            super().__init__(name, description, config)

        def validate(self, text: str) -> bool:
            return len(text) > 0

        def improve(self, text: str, feedback: Optional[str] = None) -> str:
            return text.upper() if text else ""

        def critique(self, text: str) -> BaseResult:
            return BaseResult(
                passed=True,
                message="Good text",
                score=0.8,
                issues=[],
                suggestions=[]
            )
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
    """

    def __init__(
        self, name: str, description: str, config: Optional[CriticConfig] = None, **kwargs: Any
    ) -> None:
        """
        Initialize the critic.

        Args:
            name: Name of the critic
            description: Description of the critic
            config: Optional critic configuration
            **kwargs: Additional configuration parameters
        """
        # CriticConfig is a subclass of BaseConfig, so this is type-compatible
        # Use explicit cast to satisfy mypy type checking
        base_config = cast(BaseConfig, config or CriticConfig(**kwargs))
        super().__init__(name, description, base_config)

    @abstractmethod
    def validate(self, text: T) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if text is valid, False otherwise
        """
        ...

    @abstractmethod
    def improve(self, text: T, feedback: Optional[Optional[str]] = None) -> T:
        """
        Improve text.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            The improved text
        """
        ...

    @abstractmethod
    def critique(self, text: T) -> BaseResult:
        """
        Critique text.

        Args:
            text: The text to critique

        Returns:
            BaseResult containing the critique details
        """
        ...

    def process(self, text: T) -> BaseResult:
        """
        Process text through the critic pipeline.

        Args:
            text: The text to process

        Returns:
            BaseResult containing the processing results
        """
        start_time = time.time()
        if self and not self.validate_input(text):
            return BaseResult(
                passed=False,
                message="Invalid input",
                metadata={
                    "error_type": "invalid_input",
                    "score": 0.0,
                    "issues": ["Invalid input type"],
                    "suggestions": ["Provide valid input"],
                    "processing_time_ms": (time.time() - start_time),
                },
            )
        # Convert to string for handle_empty_input which expects a string
        empty_result = (
            self.handle_empty_input(str(text) if text is not None else "") if self else None
        )
        if empty_result:
            processing_time = (time.time() - start_time)
            # Ensure we return a BaseResult
            if isinstance(empty_result, BaseResult):
                result = empty_result.with_metadata(processing_time_ms=processing_time)
                return cast(BaseResult[Any], result)  # Explicit cast to satisfy mypy
            else:
                # If empty_result is not a BaseResult, create one
                return BaseResult(
                    passed=False,
                    message="Empty input",
                    metadata={
                        "error_type": "empty_input",
                        "processing_time_ms": processing_time,
                        "original_result": str(empty_result),
                    },
                )

        def process_operation() -> Any:
            if self:
                result = self.critique(text)
                if self:
                    self.update_statistics(result)
                if (
                    result
                    and not result.passed
                    and hasattr(result, "suggestions")
                    and result.suggestions
                ):
                    # Get feedback from metadata if available, otherwise pass None
                    feedback = (
                        result.metadata.get("feedback")
                        if hasattr(result, "metadata") and result.metadata
                        else None
                    )
                    improved_text = self.improve(text, feedback) if self else None
                    if result:
                        processing_time = (time.time() - start_time)
                        result = result.with_metadata(
                            improved_text=improved_text,
                            improvement_applied=True,
                            processing_time_ms=processing_time,
                        )
                return result
            # Return a BaseResult to match the expected return type
            return BaseResult(
                passed=False,
                message="Processing failed",
                metadata={"error_type": "processing_error"},
            )

        result = safely_execute_critic(
            operation=process_operation,
            component_name=self.name,
            component_type="Critic",
            error_class=CriticError,
        )
        if isinstance(result, dict) and result and result.get("error_type"):
            error_message = (
                result.get("error_message", "Unknown error") if result else "Unknown error"
            )
            error_type = result.get("error_type") if result else "unknown"
            processing_time = (time.time() - start_time)
            return BaseResult(
                passed=False,
                message=error_message,
                metadata={
                    "error_type": error_type,
                    "score": 0.0,
                    "issues": [f"Processing error: {error_message}"],
                    "suggestions": ["Retry with different input"],
                    "processing_time_ms": processing_time,
                },
            )
        # Ensure we always return a BaseResult
        if not isinstance(result, BaseResult):
            return BaseResult(
                passed=False,
                message="Invalid result type",
                metadata={
                    "error_type": "invalid_result_type",
                    "actual_result": str(result),
                    "processing_time_ms": (time.time() - start_time),
                },
            )
        return result
