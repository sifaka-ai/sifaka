"""
Manager protocol interfaces for Sifaka.

This module defines the interfaces for managers in the Sifaka framework.
These interfaces establish a common contract for manager behavior, enabling better
modularity and extensibility.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Protocol, TypeVar, runtime_checkable

# Type variables
PromptType = TypeVar("PromptType")
InputType = TypeVar("InputType", contravariant=True)
ValidationResultType = TypeVar("ValidationResultType", covariant=True)


@runtime_checkable
class PromptManagerProtocol(Protocol[PromptType]):
    """
    Interface for prompt managers.

    This interface defines the contract for components that manage prompts.
    It ensures that prompt managers can create, format, and validate prompts.

    ## Lifecycle

    1. **Initialization**: Set up prompt management resources
    2. **Prompt Creation**: Create prompts from inputs
    3. **Prompt Formatting**: Format prompts for models
    4. **Prompt Validation**: Validate prompts
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a create_prompt method to create prompts from inputs
    - Provide a format_prompt method to format prompts for models
    - Provide a validate_prompt method to validate prompts
    """

    @abstractmethod
    def create_prompt(self, input_value: Any, **kwargs: Any) -> PromptType:
        """
        Create a prompt from an input value.

        Args:
            input_value: The input value to create a prompt from
            **kwargs: Additional prompt creation parameters

        Returns:
            A prompt

        Raises:
            ValueError: If the input value is invalid
        """
        pass

    @abstractmethod
    def format_prompt(self, prompt: PromptType, **kwargs: Any) -> Any:
        """
        Format a prompt for a model.

        Args:
            prompt: The prompt to format
            **kwargs: Additional prompt formatting parameters

        Returns:
            A formatted prompt

        Raises:
            ValueError: If the prompt is invalid
        """
        pass

    @abstractmethod
    def validate_prompt(self, prompt: PromptType) -> bool:
        """
        Validate a prompt.

        Args:
            prompt: The prompt to validate

        Returns:
            True if the prompt is valid, False otherwise

        Raises:
            ValueError: If the prompt is invalid
        """
        pass


@runtime_checkable
class ValidationManagerProtocol(Protocol[InputType, ValidationResultType]):
    """
    Interface for validation managers.

    This interface defines the contract for components that manage validation.
    It ensures that validation managers can validate inputs against rules and
    aggregate validation results.

    ## Lifecycle

    1. **Initialization**: Set up validation management resources
    2. **Rule Registration**: Register rules for validation
    3. **Input Validation**: Validate inputs against rules
    4. **Result Aggregation**: Aggregate validation results
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a validate method to validate inputs against rules
    - Provide methods to register and manage rules
    - Aggregate validation results
    """

    @abstractmethod
    def validate(self, input_value: InputType) -> ValidationResultType:
        """
        Validate an input value against registered rules.

        Args:
            input_value: The input value to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the input value is invalid
        """
        pass

    @abstractmethod
    def add_rule(self, rule: Any) -> None:
        """
        Add a rule for validation.

        Args:
            rule: The rule to add

        Raises:
            ValueError: If the rule is invalid
        """
        pass

    @abstractmethod
    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from validation.

        Args:
            rule_name: The name of the rule to remove

        Raises:
            ValueError: If the rule is not found
        """
        pass

    @abstractmethod
    def get_rules(self) -> List[Any]:
        """
        Get all registered rules.

        Returns:
            A list of registered rules
        """
        pass
