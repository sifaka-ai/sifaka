"""
Critic interfaces for Sifaka.

This module defines the interfaces for critics in the Sifaka framework.
These interfaces establish a common contract for critic behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Critic**: Base interface for all critics
   - **TextValidator**: Interface for text validators
   - **TextImprover**: Interface for text improvers
   - **TextCritic**: Interface for text critics
   - **LLMProvider**: Interface for language model providers
   - **PromptFactory**: Interface for prompt factories

2. **Text Validation Protocols**
   - `TextValidator`: Synchronous text validation

3. **Text Improvement Protocols**
   - `TextImprover`: Synchronous text improvement

4. **Text Critiquing Protocols**
   - `TextCritic`: Synchronous text critiquing

5. **Language Model Protocols**
   - `LLMProvider`: Synchronous language model provider

6. **Prompt Factory Protocols**
   - `PromptFactory`: Synchronous prompt factory

## Usage Examples

```python
from sifaka.interfaces.critic import Critic, TextValidator, TextImprover

# Create a critic implementation
class MyCritic(Critic[str, str, dict]):
    def validate(self, text: str) -> bool:
        return len(text) > 0

    def critique(self, text: str) -> dict:
        return {
            "score": 0.8,
            "feedback": "Good text",
            "issues": [],
            "suggestions": []
        }

    def improve(self, text: str, feedback: Optional[Optional[str]] = None) -> str:
        return text.strip() if text else ""
```

## Error Handling

- ValueError: Raised for invalid inputs
- RuntimeError: Raised for execution failures
- TypeError: Raised for type mismatches
"""

from abc import abstractmethod
from typing import Any, List, Optional, Protocol, TypeVar, TypedDict, runtime_checkable

from sifaka.core.interfaces import Configurable, Identifiable

# Type variables
ConfigType = TypeVar("ConfigType")
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)
ResultType = TypeVar("ResultType", covariant=True)


@runtime_checkable
class TextValidator(Protocol):
    """Protocol for text validation.

    This protocol defines the interface for text validation operations.
    Implementations must provide a method to validate text against quality standards.

    ## Lifecycle Steps
    1. Input validation
    2. Text analysis
    3. Result determination

    ## Error Handling
    - Input validation errors
    - Analysis errors
    - Result processing errors
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
class TextImprover(Protocol):
    """Protocol for text improvement.

    This protocol defines the interface for text improvement operations.
    Implementations must provide a method to improve text based on feedback.
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


class CritiqueResult(TypedDict):
    """Type definition for critique results.

    This class defines the structure of critique results, including
    score, feedback, issues, and suggestions.

    Attributes:
        score: A float between 0 and 1 indicating the quality score
        feedback: General feedback about the text
        issues: List of specific issues found
        suggestions: List of improvement suggestions
    """

    score: float
    feedback: str
    issues: List[str]
    suggestions: List[str]


@runtime_checkable
class TextCritic(Protocol):
    """Protocol for text critiquing.

    This protocol defines the interface for text critiquing operations.
    Implementations must provide a method to critique text and provide feedback.
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
class LLMProvider(Protocol):
    """Protocol for language model providers.

    This protocol defines the interface for language model providers.
    Implementations must provide a method to generate text from prompts.
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
class PromptFactory(Protocol):
    """Protocol for prompt factories.

    This protocol defines the interface for prompt factories.
    Implementations must provide a method to create prompts for language models.
    """

    def create_prompt(self, text: str, **kwargs: Any) -> str:
        """
        Create a prompt for a language model.

        Args:
            text: The text to include in the prompt
            **kwargs: Additional arguments for prompt creation

        Returns:
            str: The formatted prompt

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If prompt creation fails
        """
        ...


@runtime_checkable
class Critic(
    Identifiable, Configurable[ConfigType], Protocol[ConfigType, InputType, OutputType, ResultType]
):
    """
    Interface for critics.

    This interface defines the contract for components that critique and improve text.
    It ensures that critics can validate, critique, and improve text, and expose
    critic metadata.

    ## Lifecycle

    1. **Initialization**: Set up critic resources and configuration
    2. **Validation**: Validate text
    3. **Critique**: Critique text
    4. **Improvement**: Improve text
    5. **Configuration Management**: Manage critic configuration
    6. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide validate, critique, and improve methods
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the critic configuration
    - Provide an update_config method to update the critic configuration
    """

    @abstractmethod
    def validate(self, text: InputType) -> bool:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid, False otherwise

        Raises:
            ValueError: If the text is invalid
        """
        pass

    @abstractmethod
    def critique(self, text: InputType) -> ResultType:
        """
        Critique text.

        Args:
            text: The text to critique

        Returns:
            A critique result

        Raises:
            ValueError: If the text is invalid
        """
        pass

    @abstractmethod
    def improve(self, text: InputType, feedback: Optional[Optional[str]] = None) -> OutputType:
        """
        Improve text.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If the text is invalid
        """
        pass
