"""
Chain interfaces for Sifaka.

This module defines the interfaces for chains in the Sifaka framework.
These interfaces establish a common contract for chain behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Base interface for all chains
   - **PromptManager**: Interface for prompt managers
   - **ValidationManager**: Interface for validation managers
   - **RetryStrategy**: Interface for retry strategies
   - **ResultFormatter**: Interface for result formatters

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.interfaces.core import Component, Configurable, Identifiable, Stateful

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)
ResultType = TypeVar("ResultType", covariant=True)
PromptType = TypeVar("PromptType")
ValidationResultType = TypeVar("ValidationResultType")


@runtime_checkable
class PromptManager(Protocol[PromptType]):
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
class ValidationManager(Protocol[InputType, ValidationResultType]):
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


@runtime_checkable
class RetryStrategy(Protocol):
    """
    Interface for retry strategies.

    This interface defines the contract for components that manage retry logic.
    It ensures that retry strategies can execute operations with retries and
    handle retry conditions.

    ## Lifecycle

    1. **Initialization**: Set up retry strategy resources
    2. **Execution**: Execute operations with retries
    3. **Condition Handling**: Handle retry conditions
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an execute method to execute operations with retries
    - Handle retry conditions appropriately
    """

    @abstractmethod
    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an operation with retries.

        Args:
            operation: The operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            RuntimeError: If the operation fails after all retries
        """
        pass

    @abstractmethod
    def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
        """
        Check if an operation should be retried.

        Args:
            attempt: The current attempt number
            result: The result of the operation
            error: The error that occurred, if any

        Returns:
            True if the operation should be retried, False otherwise
        """
        pass


@runtime_checkable
class ResultFormatter(Protocol[ResultType]):
    """
    Interface for result formatters.

    This interface defines the contract for components that format results.
    It ensures that result formatters can format results for different outputs.

    ## Lifecycle

    1. **Initialization**: Set up result formatting resources
    2. **Formatting**: Format results for different outputs
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a format method to format results
    """

    @abstractmethod
    def format(self, result: Any) -> ResultType:
        """
        Format a result.

        Args:
            result: The result to format

        Returns:
            A formatted result

        Raises:
            ValueError: If the result is invalid
        """
        pass


@runtime_checkable
class Chain(Identifiable, Configurable[ConfigType], Protocol[InputType, OutputType]):
    """
    Interface for chains.

    This interface defines the contract for components that orchestrate the
    validation and improvement flow between models, rules, and critics. It ensures
    that chains can run inputs through the flow and return standardized results.

    ## Lifecycle

    1. **Initialization**: Set up chain resources and configuration
    2. **Execution**: Run inputs through the flow
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage chain configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    """

    @abstractmethod
    def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain.

        Args:
            input_value: The input value to run
            **kwargs: Additional run parameters

        Returns:
            A chain result

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails
        """
        pass


@runtime_checkable
class AsyncChain(Protocol[InputType, OutputType]):
    """
    Interface for asynchronous chains.

    This interface defines the contract for components that orchestrate the
    validation and improvement flow between models, rules, and critics asynchronously.
    It ensures that chains can run inputs through the flow asynchronously and return
    standardized results.

    ## Lifecycle

    1. **Initialization**: Set up chain resources and configuration
    2. **Execution**: Run inputs through the flow asynchronously
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage chain configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the chain name.

        Returns:
            The chain name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the chain description.

        Returns:
            The chain description
        """
        pass

    @abstractmethod
    async def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain asynchronously.

        Args:
            input_value: The input value to run
            **kwargs: Additional run parameters

        Returns:
            A chain result

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails
        """
        pass
