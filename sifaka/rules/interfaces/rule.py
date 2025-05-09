"""
Rule interfaces for Sifaka.

This module defines the interfaces for rules and validators in the Sifaka framework.
These interfaces establish a common contract for rule behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Rule**: Base interface for all rules
   - **Validator**: Interface for validators
   - **RuleResultHandler**: Interface for rule result handlers
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.core.interfaces import Configurable, Identifiable

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
InputType = TypeVar("InputType")
ResultType = TypeVar("ResultType")

from ..config import RuleConfig
from ..result import RuleResult


@runtime_checkable
class Validatable(Protocol[ResultType]):
    """
    Interface for objects that can be validated.

    This interface defines the contract for components that can be validated
    against rules. It ensures that validatable objects can be checked for
    validity and expose validation errors.

    ## Lifecycle

    1. **Validation**: Check if the object is valid
    2. **Error Reporting**: Get validation errors

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an is_valid method to check if the object is valid
    - Provide a get_errors method to get validation errors
    """

    @abstractmethod
    def is_valid(self) -> bool:
        """
        Check if the object is valid.

        Returns:
            True if the object is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_errors(self) -> List[ResultType]:
        """
        Get validation errors.

        Returns:
            A list of validation errors
        """
        pass


@runtime_checkable
class RuleResultHandler(Protocol[ResultType]):
    """
    Interface for rule result handlers.

    This interface defines the contract for components that handle rule validation
    results. It ensures that result handlers can process validation results and
    take appropriate actions.

    ## Lifecycle

    1. **Initialization**: Set up result handling resources
    2. **Result Processing**: Process validation results
    3. **Action Execution**: Take actions based on results
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a handle_result method to process validation results
    - Take appropriate actions based on results
    """

    @abstractmethod
    def handle_result(self, result: ResultType) -> None:
        """
        Handle a validation result.

        Args:
            result: The validation result to handle

        Raises:
            ValueError: If the result is invalid
            RuntimeError: If result handling fails
        """
        pass


@runtime_checkable
class Rule(Identifiable, Configurable[ConfigType], Protocol[InputType, ResultType]):
    """
    Interface for rules.

    This interface defines the contract for components that validate inputs
    against rules. It ensures that rules can validate inputs, return standardized
    validation results, and expose rule metadata.

    ## Lifecycle

    1. **Initialization**: Set up rule resources and configuration
    2. **Validation**: Validate inputs against the rule
    3. **Result Handling**: Process validation results
    4. **Configuration Management**: Manage rule configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a validate method to validate inputs
    - Return standardized validation results
    - Provide name and description properties
    - Provide a config property to access the rule configuration
    - Provide an update_config method to update the rule configuration
    """

    @abstractmethod
    def validate(self, input_value: InputType) -> ResultType:
        """
        Validate an input value against the rule.

        Args:
            input_value: The input value to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the input value is invalid
        """
        pass


@runtime_checkable
class AsyncRule(Protocol[InputType, ResultType]):
    """
    Interface for asynchronous rules.

    This interface defines the contract for components that validate inputs
    asynchronously against rules. It ensures that rules can validate inputs
    asynchronously, return standardized validation results, and expose rule metadata.

    ## Lifecycle

    1. **Initialization**: Set up rule resources and configuration
    2. **Validation**: Validate inputs against the rule asynchronously
    3. **Result Handling**: Process validation results
    4. **Configuration Management**: Manage rule configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async validate method to validate inputs
    - Return standardized validation results
    - Provide name and description properties
    - Provide a config property to access the rule configuration
    - Provide an update_config method to update the rule configuration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the rule name.

        Returns:
            The rule name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the rule description.

        Returns:
            The rule description
        """
        pass

    @property
    @abstractmethod
    def config(self) -> ConfigType:
        """
        Get the rule configuration.

        Returns:
            The rule configuration
        """
        pass

    @abstractmethod
    def update_config(self, config: ConfigType) -> None:
        """
        Update the rule configuration.

        Args:
            config: The new configuration object or values to update

        Raises:
            ValueError: If the configuration is invalid
        """
        pass

    @abstractmethod
    async def validate(self, input_value: InputType) -> ResultType:
        """
        Validate an input value against the rule asynchronously.

        Args:
            input_value: The input value to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the input value is invalid
        """
        pass


@runtime_checkable
class RuleProtocol(Protocol):
    """
    Protocol defining the interface for rules.

    This protocol is useful for type checking code that works with rules
    without requiring a specific rule implementation. It defines the minimum
    interface that all rule-like objects must implement to be used in contexts
    that expect rules.

    Lifecycle:
        1. Access: Get rule properties (name, description, config)
        2. Validation: Validate input against rule
        3. Result Handling: Process validation results
    """

    @property
    def name(self) -> str:
        """
        Get the rule name.

        Returns:
            The name of the rule
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the rule description.

        Returns:
            The description of the rule
        """
        ...

    @property
    def config(self) -> RuleConfig:
        """
        Get the rule configuration.

        Returns:
            The configuration of the rule
        """
        ...

    def validate(self, text: str, **kwargs: Any) -> RuleResult:
        """
        Validate text against the rule.

        Args:
            text: The text to validate
            **kwargs: Additional validation options

        Returns:
            The validation result
        """
        ...
