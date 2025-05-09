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


@runtime_checkable
class TextValidator(Protocol[InputType, ResultType]):
    """
    Interface for text validators.

    This interface defines the contract for components that validate text.
    It ensures that text validators can validate text and return standardized
    validation results.

    ## Lifecycle

    1. **Initialization**: Set up validation resources
    2. **Validation**: Validate text
    3. **Result Handling**: Process validation results
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a validate method to validate text
    - Return standardized validation results
    """

    @abstractmethod
    def validate(self, text: InputType) -> ResultType:
        """
        Validate text.

        Args:
            text: The text to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the text is invalid
        """
        pass


@runtime_checkable
class TextImprover(Protocol[InputType, OutputType]):
    """
    Interface for text improvers.

    This interface defines the contract for components that improve text.
    It ensures that text improvers can improve text and return improved text.

    ## Lifecycle

    1. **Initialization**: Set up improvement resources
    2. **Improvement**: Improve text
    3. **Result Handling**: Process improvement results
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an improve method to improve text
    - Return improved text
    """

    @abstractmethod
    def improve(self, text: InputType, feedback: Optional[str] = None) -> OutputType:
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


@runtime_checkable
class TextCritic(Protocol[InputType, ResultType]):
    """
    Interface for text critics.

    This interface defines the contract for components that critique text.
    It ensures that text critics can critique text and return critique results.

    ## Lifecycle

    1. **Initialization**: Set up critique resources
    2. **Critique**: Critique text
    3. **Result Handling**: Process critique results
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a critique method to critique text
    - Return critique results
    """

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


@runtime_checkable
class Critic(Identifiable, Configurable[ConfigType], Protocol[InputType, OutputType, ResultType]):
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
    def improve(self, text: InputType, feedback: Optional[str] = None) -> OutputType:
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


@runtime_checkable
class AsyncCritic(Protocol[InputType, OutputType, ResultType]):
    """
    Interface for asynchronous critics.

    This interface defines the contract for components that critique and improve text
    asynchronously. It ensures that critics can validate, critique, and improve text
    asynchronously, and expose critic metadata.

    ## Lifecycle

    1. **Initialization**: Set up critic resources and configuration
    2. **Validation**: Validate text asynchronously
    3. **Critique**: Critique text asynchronously
    4. **Improvement**: Improve text asynchronously
    5. **Configuration Management**: Manage critic configuration
    6. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide async validate, critique, and improve methods
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the critic configuration
    - Provide an update_config method to update the critic configuration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the critic name.

        Returns:
            The critic name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the critic description.

        Returns:
            The critic description
        """
        pass

    @abstractmethod
    async def validate(self, text: InputType) -> bool:
        """
        Validate text asynchronously.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid, False otherwise

        Raises:
            ValueError: If the text is invalid
        """
        pass

    @abstractmethod
    async def critique(self, text: InputType) -> ResultType:
        """
        Critique text asynchronously.

        Args:
            text: The text to critique

        Returns:
            A critique result

        Raises:
            ValueError: If the text is invalid
        """
        pass

    @abstractmethod
    async def improve(self, text: InputType, feedback: Optional[str] = None) -> OutputType:
        """
        Improve text asynchronously.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement

        Returns:
            Improved text

        Raises:
            ValueError: If the text is invalid
        """
        pass
