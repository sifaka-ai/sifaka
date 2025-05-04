"""
Protocol definitions for critics in Sifaka.

This module defines the protocols that critics must implement to be compatible
with the Sifaka framework. These protocols define the interfaces for text validation,
improvement, and critiquing.

## Protocol Overview

The module provides several key protocols:

1. **Text Validation Protocols**
   - `SyncTextValidator`: Synchronous text validation
   - `AsyncTextValidator`: Asynchronous text validation
   - `TextValidator`: Combined synchronous protocol

2. **Text Improvement Protocols**
   - `SyncTextImprover`: Synchronous text improvement
   - `AsyncTextImprover`: Asynchronous text improvement
   - `TextImprover`: Combined synchronous protocol

3. **Text Critiquing Protocols**
   - `SyncTextCritic`: Synchronous text critiquing
   - `AsyncTextCritic`: Asynchronous text critiquing
   - `TextCritic`: Combined synchronous protocol

4. **Language Model Protocols**
   - `SyncLLMProvider`: Synchronous language model provider
   - `AsyncLLMProvider`: Asynchronous language model provider
   - `LLMProvider`: Combined synchronous protocol

5. **Prompt Factory Protocols**
   - `SyncPromptFactory`: Synchronous prompt factory
   - `AsyncPromptFactory`: Asynchronous prompt factory
   - `PromptFactory`: Combined synchronous protocol

## Protocol Lifecycle

Each protocol defines a specific lifecycle for its operations:

1. **Initialization**
   - Protocol implementation
   - Resource setup
   - State initialization

2. **Operation**
   - Method execution
   - Error handling
   - Result processing

3. **Cleanup**
   - Resource release
   - State cleanup
   - Error recovery

## Error Handling

Protocols define error handling requirements:

1. **Input Validation**
   - Parameter validation
   - Type checking
   - Format verification

2. **Operation Errors**
   - Method execution errors
   - Resource errors
   - State errors

3. **Result Validation**
   - Return type checking
   - Format validation
   - Content verification

## Examples

```python
from sifaka.critics.protocols import TextValidator, TextImprover, TextCritic

class MyCritic(TextValidator, TextImprover, TextCritic):
    def validate(self, text: str) -> bool:
        # Implementation
        pass

    def improve(self, text: str, feedback: str) -> str:
        # Implementation
        pass

    def critique(self, text: str) -> dict:
        # Implementation
        pass

# Create and use the critic
critic = MyCritic()
text = "This is a sample text."
is_valid = critic.validate(text)
improved = critic.improve(text, "Add more detail.")
feedback = critic.critique(text)
```
"""

from typing import Any, List, Protocol, TypedDict, runtime_checkable


@runtime_checkable
class SyncTextValidator(Protocol):
    """Protocol for synchronous text validation.

    This protocol defines the interface for synchronous text validation operations.
    Implementations must provide a method to validate text against quality standards.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Result determination

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Result processing errors

    Examples:
        ```python
        class MyValidator(SyncTextValidator):
            def validate(self, text: str) -> bool:
                # Implementation
                return True
        ```
    """

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        ...


@runtime_checkable
class AsyncTextValidator(Protocol):
    """Protocol for asynchronous text validation.

    This protocol defines the interface for asynchronous text validation operations.
    Implementations must provide an async method to validate text against quality standards.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Result determination

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Result processing errors

    Examples:
        ```python
        class MyAsyncValidator(AsyncTextValidator):
            async def validate(self, text: str) -> bool:
                # Implementation
                return True
        ```
    """

    async def validate(self, text: str) -> bool:
        """
        Asynchronously validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            bool: True if the text passes validation, False otherwise

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If validation fails
        """
        ...


@runtime_checkable
class TextValidator(SyncTextValidator, Protocol):
    """Protocol for text validation (sync version).

    This protocol combines the synchronous text validation interface with
    additional functionality for text validation operations.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Result determination

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Result processing errors

    Examples:
        ```python
        class MyValidator(TextValidator):
            def validate(self, text: str) -> bool:
                # Implementation
                return True
        ```
    """

    ...


@runtime_checkable
class SyncTextImprover(Protocol):
    """Protocol for synchronous text improvement.

    This protocol defines the interface for synchronous text improvement operations.
    Implementations must provide a method to improve text based on feedback.

    ## Lifecycle Steps
    1. Input validation
    2. Feedback processing
    3. Text improvement
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Feedback processing errors
    - Improvement errors
    - Formatting errors

    Examples:
        ```python
        class MyImprover(SyncTextImprover):
            def improve(self, text: str, feedback: str) -> str:
                # Implementation
                return "Improved text"
        ```
    """

    def improve(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or feedback is empty or invalid
            RuntimeError: If improvement fails
        """
        ...


@runtime_checkable
class AsyncTextImprover(Protocol):
    """Protocol for asynchronous text improvement.

    This protocol defines the interface for asynchronous text improvement operations.
    Implementations must provide an async method to improve text based on feedback.

    ## Lifecycle Steps
    1. Input validation
    2. Feedback processing
    3. Text improvement
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Feedback processing errors
    - Improvement errors
    - Formatting errors

    Examples:
        ```python
        class MyAsyncImprover(AsyncTextImprover):
            async def improve(self, text: str, feedback: str) -> str:
                # Implementation
                return "Improved text"
        ```
    """

    async def improve(self, text: str, feedback: str) -> str:
        """
        Asynchronously improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improved text

        Raises:
            ValueError: If text or feedback is empty or invalid
            RuntimeError: If improvement fails
        """
        ...


@runtime_checkable
class TextImprover(SyncTextImprover, Protocol):
    """Protocol for text improvement (sync version).

    This protocol combines the synchronous text improvement interface with
    additional functionality for text improvement operations.

    ## Lifecycle Steps
    1. Input validation
    2. Feedback processing
    3. Text improvement
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Feedback processing errors
    - Improvement errors
    - Formatting errors

    Examples:
        ```python
        class MyImprover(TextImprover):
            def improve(self, text: str, feedback: str) -> str:
                # Implementation
                return "Improved text"
        ```
    """

    ...


class CritiqueResult(TypedDict):
    """Type definition for critique results.

    This class defines the structure of critique results, including
    score, feedback, issues, and suggestions.

    Attributes:
        score: A float between 0 and 1 indicating the quality score
        feedback: General feedback about the text
        issues: List of specific issues found
        suggestions: List of improvement suggestions

    Examples:
        ```python
        result: CritiqueResult = {
            "score": 0.8,
            "feedback": "Good overall structure",
            "issues": ["Missing examples", "Unclear terminology"],
            "suggestions": ["Add more examples", "Define technical terms"]
        }
        ```
    """

    score: float
    feedback: str
    issues: List[str]
    suggestions: List[str]


@runtime_checkable
class SyncTextCritic(Protocol):
    """Protocol for synchronous text critiquing.

    This protocol defines the interface for synchronous text critiquing operations.
    Implementations must provide a method to critique text and provide feedback.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Feedback generation
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Feedback generation errors
    - Formatting errors

    Examples:
        ```python
        class MyCritic(SyncTextCritic):
            def critique(self, text: str) -> CritiqueResult:
                # Implementation
                return {
                    "score": 0.8,
                    "feedback": "Good overall structure",
                    "issues": ["Missing examples"],
                    "suggestions": ["Add more examples"]
                }
        ```
    """

    def critique(self, text: str) -> CritiqueResult:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        ...


@runtime_checkable
class AsyncTextCritic(Protocol):
    """Protocol for asynchronous text critiquing.

    This protocol defines the interface for asynchronous text critiquing operations.
    Implementations must provide an async method to critique text and provide feedback.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Feedback generation
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Feedback generation errors
    - Formatting errors

    Examples:
        ```python
        class MyAsyncCritic(AsyncTextCritic):
            async def critique(self, text: str) -> CritiqueResult:
                # Implementation
                return {
                    "score": 0.8,
                    "feedback": "Good overall structure",
                    "issues": ["Missing examples"],
                    "suggestions": ["Add more examples"]
                }
        ```
    """

    async def critique(self, text: str) -> CritiqueResult:
        """
        Asynchronously critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CritiqueResult: A dictionary containing critique information

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If critique fails
        """
        ...


@runtime_checkable
class TextCritic(SyncTextCritic, Protocol):
    """Protocol for text critiquing (sync version).

    This protocol combines the synchronous text critiquing interface with
    additional functionality for text critiquing operations.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Feedback generation
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Feedback generation errors
    - Formatting errors

    Examples:
        ```python
        class MyCritic(TextCritic):
            def critique(self, text: str) -> CritiqueResult:
                # Implementation
                return {
                    "score": 0.8,
                    "feedback": "Good overall structure",
                    "issues": ["Missing examples"],
                    "suggestions": ["Add more examples"]
                }
        ```
    """

    ...


@runtime_checkable
class SyncLLMProvider(Protocol):
    """Protocol for synchronous language model providers.

    This protocol defines the interface for synchronous language model providers.
    Implementations must provide a method to generate text from prompts.

    ## Lifecycle Steps
    1. Input validation
    2. Prompt processing
    3. Model interaction
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Model interaction errors
    - Response processing errors
    - Formatting errors

    Examples:
        ```python
        class MyProvider(SyncLLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                # Implementation
                return "Generated text"
        ```
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments for the model

        Returns:
            str: The generated text

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        ...


@runtime_checkable
class AsyncLLMProvider(Protocol):
    """Protocol for asynchronous language model providers.

    This protocol defines the interface for asynchronous language model providers.
    Implementations must provide an async method to generate text from prompts.

    ## Lifecycle Steps
    1. Input validation
    2. Prompt processing
    3. Model interaction
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Model interaction errors
    - Response processing errors
    - Formatting errors

    Examples:
        ```python
        class MyAsyncProvider(AsyncLLMProvider):
            async def generate(self, prompt: str, **kwargs: Any) -> str:
                # Implementation
                return "Generated text"
        ```
    """

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Asynchronously generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional arguments for the model

        Returns:
            str: The generated text

        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        ...


@runtime_checkable
class LLMProvider(SyncLLMProvider, Protocol):
    """Protocol for language model providers (sync version).

    This protocol combines the synchronous language model provider interface with
    additional functionality for text generation operations.

    ## Lifecycle Steps
    1. Input validation
    2. Prompt processing
    3. Model interaction
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Model interaction errors
    - Response processing errors
    - Formatting errors

    Examples:
        ```python
        class MyProvider(LLMProvider):
            def generate(self, prompt: str, **kwargs: Any) -> str:
                # Implementation
                return "Generated text"
        ```
    """

    ...


@runtime_checkable
class SyncPromptFactory(Protocol):
    """Protocol for synchronous prompt factories.

    This protocol defines the interface for synchronous prompt factories.
    Implementations must provide a method to create prompts for language models.

    ## Lifecycle Steps
    1. Input validation
    2. Template processing
    3. Variable substitution
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Template processing errors
    - Variable substitution errors
    - Formatting errors

    Examples:
        ```python
        class MyFactory(SyncPromptFactory):
            def create_prompt(self, text: str, **kwargs: Any) -> str:
                # Implementation
                return "Formatted prompt"
        ```
    """

    def create_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If prompt creation fails
        """
        ...


@runtime_checkable
class AsyncPromptFactory(Protocol):
    """Protocol for asynchronous prompt factories.

    This protocol defines the interface for asynchronous prompt factories.
    Implementations must provide an async method to create prompts for language models.

    ## Lifecycle Steps
    1. Input validation
    2. Template processing
    3. Variable substitution
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Template processing errors
    - Variable substitution errors
    - Formatting errors

    Examples:
        ```python
        class MyAsyncFactory(AsyncPromptFactory):
            async def create_prompt(self, text: str, **kwargs: Any) -> str:
                # Implementation
                return "Formatted prompt"
        ```
    """

    async def create_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Asynchronously create a prompt for a language model.

        Args:
            text: The text to create a prompt for
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The created prompt

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If prompt creation fails
        """
        ...


@runtime_checkable
class PromptFactory(SyncPromptFactory, Protocol):
    """Protocol for prompt factories (sync version).

    This protocol combines the synchronous prompt factory interface with
    additional functionality for prompt creation operations.

    ## Lifecycle Steps
    1. Input validation
    2. Template processing
    3. Variable substitution
    4. Result formatting

    ## Error Handling
    - Input validation errors
    - Template processing errors
    - Variable substitution errors
    - Formatting errors

    Examples:
        ```python
        class MyFactory(PromptFactory):
            def create_prompt(self, text: str, **kwargs: Any) -> str:
                # Implementation
                return "Formatted prompt"
        ```
    """

    ...


# Export public protocols
__all__ = [
    # Synchronous protocols
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "LLMProvider",
    "PromptFactory",
    # Synchronous explicit protocols
    "SyncTextValidator",
    "SyncTextImprover",
    "SyncTextCritic",
    "SyncLLMProvider",
    "SyncPromptFactory",
    # Asynchronous protocols
    "AsyncTextValidator",
    "AsyncTextImprover",
    "AsyncTextCritic",
    "AsyncLLMProvider",
    "AsyncPromptFactory",
    # Type definitions
    "CritiqueResult",
]
