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
   - **Model**: Interface for text generation models
   - **Validator**: Interface for output validators
   - **Improver**: Interface for output improvers
   - **Formatter**: Interface for result formatters
   - **ChainComponent**: Base interface for all chain components

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## State Management

The interfaces support standardized state management:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager methods
- Clear separation of configuration and state

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails
- Detailed error tracking and reporting

## Execution Tracking

The interfaces support execution tracking:
- Execution count tracking
- Execution time tracking
- Success/failure tracking
- Performance statistics
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel, Field

from sifaka.core.interfaces import (
    Component,
    Configurable,
    Identifiable,
    Stateful,
    Plugin as CorePlugin,
)

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
    5. **State Management**: Manage chain state
    6. **Error Handling**: Handle and track errors
    7. **Execution Tracking**: Track execution statistics
    8. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    - Implement state management using _state_manager
    - Implement error handling and tracking
    - Implement execution tracking and statistics
    """

    @property
    @abstractmethod
    def _state_manager(self) -> Any:
        """
        Get the state manager.

        Returns:
            The state manager
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the chain.

        This method should be called after the chain is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up the chain.

        This method should be called when the chain is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            The current state
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state.

        Args:
            state: The new state

        Raises:
            ValueError: If the state is invalid
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the state to its initial values.

        Raises:
            RuntimeError: If state reset fails
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            A dictionary of execution statistics
        """
        pass

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
    5. **State Management**: Manage chain state
    6. **Error Handling**: Handle and track errors
    7. **Execution Tracking**: Track execution statistics
    8. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    - Implement state management using _state_manager
    - Implement error handling and tracking
    - Implement execution tracking and statistics
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

    @property
    @abstractmethod
    def config(self) -> Any:
        """
        Get the chain configuration.

        Returns:
            The chain configuration
        """
        pass

    @abstractmethod
    def update_config(self, config: Any) -> None:
        """
        Update the chain configuration.

        Args:
            config: The new configuration

        Raises:
            ValueError: If the configuration is invalid
        """
        pass

    @property
    @abstractmethod
    def _state_manager(self) -> Any:
        """
        Get the state manager.

        Returns:
            The state manager
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the chain asynchronously.

        This method should be called after the chain is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up the chain asynchronously.

        This method should be called when the chain is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """
        pass

    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """
        Get the current state asynchronously.

        Returns:
            The current state
        """
        pass

    @abstractmethod
    async def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state asynchronously.

        Args:
            state: The new state

        Raises:
            ValueError: If the state is invalid
        """
        pass

    @abstractmethod
    async def reset_state(self) -> None:
        """
        Reset the state to its initial values asynchronously.

        Raises:
            RuntimeError: If state reset fails
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics asynchronously.

        Returns:
            A dictionary of execution statistics
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


@runtime_checkable
class ChainComponent(Protocol):
    """
    Base interface for all chain components.

    This interface defines the common properties that all chain components
    must implement. It serves as the foundation for the component hierarchy
    in the chain system, ensuring consistent identification and description
    of components.

    ## Architecture
    The ChainComponent interface uses Python's Protocol class to define
    a structural interface that components can implement without explicit
    inheritance. This enables better flexibility and composition in the
    component architecture.

    ## Lifecycle
    Components implementing this interface should maintain consistent
    name and description properties throughout their lifecycle.

    ## Examples
    ```python
    class MyComponent:
        @property
        def name(self) -> str:
            return "my_component"

        @property
        def description(self) -> str:
            return "A custom chain component"
    ```
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            str: A unique identifier for the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            str: A human-readable description of the component
        """
        ...


class ValidationResult(BaseModel):
    """
    Result of a validation operation.

    This class represents the result of validating an output against a rule or
    validator. It includes information about whether the validation passed,
    a descriptive message, a numerical score, and optional lists of issues
    and improvement suggestions.

    ## Architecture
    The ValidationResult class is designed as an immutable Pydantic model
    with fields for storing validation results. It follows a value object
    pattern, where instances represent the outcome of a validation operation.

    ## Lifecycle
    1. Creation: Instantiated with validation outcome
    2. Usage: Accessed to retrieve validation status and details

    ## Examples
    ```python
    # Create a validation result
    result = ValidationResult(
        passed=True,
        message="Length validation passed",
        score=1.0
    )

    # Create a failed validation with issues and suggestions
    failed_result = ValidationResult(
        passed=False,
        message="Content validation failed",
        score=0.3,
        issues=["Contains inappropriate content"],
        suggestions=["Remove inappropriate content"]
    )

    # Check validation status
    if result.passed:
        print("Validation passed!")
    else:
        print(f"Validation failed: {result.message}")
        print(f"Issues: {result.issues}")
        print(f"Suggestions: {result.suggestions}")
    ```

    Attributes:
        passed (bool): Whether the validation passed
        message (str): Validation message
        score (float): Validation score (0.0 to 1.0)
        issues (List[str]): List of issues found
        suggestions (List[str]): List of improvement suggestions
        metadata (Dict[str, Any]): Additional metadata
    """

    passed: bool = Field(description="Whether the validation passed")
    message: str = Field(description="Validation message")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation score")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@runtime_checkable
class Model(ChainComponent, Protocol):
    """Interface for text generation models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        ...

    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.

        This method has a default implementation that calls the synchronous
        generate method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


@runtime_checkable
class Validator(ChainComponent, Protocol):
    """Interface for output validators."""

    @abstractmethod
    def validate(self, output: str) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: The output to validate

        Returns:
            The validation result

        Raises:
            ValidationError: If validation fails
        """
        ...

    async def validate_async(self, output: str) -> ValidationResult:
        """
        Validate an output asynchronously.

        This method has a default implementation that calls the synchronous
        validate method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to validate

        Returns:
            The validation result

        Raises:
            ValidationError: If validation fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate, output)


@runtime_checkable
class Improver(ChainComponent, Protocol):
    """Interface for output improvers."""

    @abstractmethod
    def improve(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output based on validation results.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        ...

    async def improve_async(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output asynchronously.

        This method has a default implementation that calls the synchronous
        improve method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.improve(output, validation_results))


@runtime_checkable
class ChainFormatter(ChainComponent, Protocol):
    """Interface for result formatters."""

    @abstractmethod
    def format(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        ...

    async def format_async(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result asynchronously.

        This method has a default implementation that calls the synchronous
        format method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.format(output, validation_results))


@runtime_checkable
class ChainPlugin(ChainComponent, CorePlugin, Protocol):
    """
    Interface for chain plugins.

    This interface extends the core Plugin interface with chain-specific
    functionality. It ensures that chain plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """

    pass
