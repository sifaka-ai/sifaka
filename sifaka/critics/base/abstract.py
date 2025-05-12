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
    def __init__(self, name: str, description: str, config: CriticConfig = None):
        super().__init__(name, description, config)

    def validate(self, text: str) -> bool:
        return len(text) > 0

    def improve(self, text: str, feedback: str = None) -> str:
        return text.upper()

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
is_valid = critic.validate(text)
improved = critic.improve(text)
feedback = critic.critique(text)
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
from typing import Any, Generic, Optional, TypeVar

from sifaka.core.base import BaseComponent, BaseResult
from sifaka.utils.config.critics import CriticConfig
from sifaka.utils.errors import safely_execute_component_operation as safely_execute_critic

# Input type variable
T = TypeVar("T")  # Input type (usually str)


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
        def __init__(self, name: str, description: str, config: CriticConfig = None):
            super().__init__(name, description, config)

        def validate(self, text: str) -> bool:
            return len(text) > 0

        def improve(self, text: str, feedback: str = None) -> str:
            return text.upper()

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
        super().__init__(name, description, config or CriticConfig(**kwargs))

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
    def improve(self, text: T, feedback: Optional[str] = None) -> T:
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

        # Validate input
        if not self.validate_input(text):
            return BaseResult(
                passed=False,
                message="Invalid input",
                metadata={"error_type": "invalid_input"},
                score=0.0,
                issues=["Invalid input type"],
                suggestions=["Provide valid input"],
                processing_time_ms=time.time() - start_time,
            )

        # Handle empty input
        empty_result = self.handle_empty_input(text)
        if empty_result:
            return empty_result.with_metadata(processing_time_ms=time.time() - start_time)

        # Define the processing operation
        def process_operation():
            # Run critique
            result = self.critique(text)
            self.update_statistics(result)

            # If improvement needed, try to improve
            if not result.passed and result.suggestions:
                improved_text = self.improve(text, result.feedback)
                result = result.with_metadata(
                    improved_text=improved_text,
                    improvement_applied=True,
                    processing_time_ms=time.time() - start_time,
                )

            return result

        # Use the standardized safely_execute_critic function
        result = safely_execute_critic(
            operation=process_operation,
            critic_name=self.name,
            component_name=self.__class__.__name__,
        )

        # If the result is an ErrorResult, convert it to a BaseResult
        if isinstance(result, dict) and result.get("error_type"):
            return BaseResult(
                passed=False,
                message=result.get("error_message", "Unknown error"),
                metadata={"error_type": result.get("error_type")},
                score=0.0,
                issues=[f"Processing error: {result.get('error_message')}"],
                suggestions=["Retry with different input"],
                processing_time_ms=time.time() - start_time,
            )

        return result
