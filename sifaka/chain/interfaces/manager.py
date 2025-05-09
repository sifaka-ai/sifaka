"""
Manager Interface Module

Protocol interfaces for Sifaka's manager system.

## Overview
This module defines the interfaces for managers in the Sifaka framework.
These interfaces establish a common contract for manager behavior, enabling better
modularity and extensibility.

## Components
1. **PromptManagerProtocol**: Interface for prompt management
   - Prompt creation
   - Prompt formatting
   - Prompt validation
2. **ValidationManagerProtocol**: Interface for validation management
   - Input validation
   - Rule management
   - Result aggregation

## Usage Examples
```python
from typing import Any, Dict, List
from sifaka.chain.interfaces.manager import PromptManagerProtocol, ValidationManagerProtocol

class SimplePromptManager(PromptManagerProtocol[str]):
    def create_prompt(self, input_value: Any, **kwargs: Any) -> str:
        return f"Process this input: {input_value}"

    def format_prompt(self, prompt: str, **kwargs: Any) -> Any:
        return {"text": prompt, "format": "plain"}

    def validate_prompt(self, prompt: str) -> bool:
        return len(prompt) > 0

class SimpleValidationManager(ValidationManagerProtocol[str, Dict[str, Any]]):
    def validate(self, input_value: str) -> Dict[str, Any]:
        return {"valid": True, "message": "Input is valid"}

    def add_rule(self, rule: Any) -> None:
        pass

    def remove_rule(self, rule_name: str) -> None:
        pass

    def get_rules(self) -> List[Any]:
        return []
```

## Error Handling
- ValueError: Raised when input values or rules are invalid
- RuntimeError: Raised when manager operations fail

## Configuration
- input_value: The input value to process
- prompt: The prompt to format/validate
- rule: The rule to add/remove
- rule_name: The name of the rule to remove
- kwargs: Additional parameters for operations
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

    Detailed description of what the class does, including:
    - Defines the contract for components that manage prompts
    - Ensures prompt managers can create, format, and validate prompts
    - Handles prompt lifecycle from creation to validation
    - Maintains consistent behavior across different prompt manager implementations

    Type parameters:
        PromptType: The type of prompt managed by this protocol

    Example:
        ```python
        class SimplePromptManager(PromptManagerProtocol[str]):
            def create_prompt(self, input_value: Any, **kwargs: Any) -> str:
                return f"Process this input: {input_value}"

            def format_prompt(self, prompt: str, **kwargs: Any) -> Any:
                return {"text": prompt, "format": "plain"}

            def validate_prompt(self, prompt: str) -> bool:
                return len(prompt) > 0
        ```
    """

    @abstractmethod
    def create_prompt(self, input_value: Any, **kwargs: Any) -> PromptType:
        """
        Create a prompt from an input value.

        Detailed description of what the method does, including:
        - Transforms an input value into a prompt
        - Handles input validation and preprocessing
        - Applies any necessary formatting or structure
        - Returns a prompt ready for use

        Args:
            input_value: The input value to create a prompt from
            **kwargs: Additional prompt creation parameters

        Returns:
            A prompt

        Raises:
            ValueError: If the input value is invalid

        Example:
            ```python
            # Create a prompt from an input
            prompt = manager.create_prompt(
                input_value="Write a story",
                format="markdown"
            )
            print(f"Created prompt: {prompt}")
            ```
        """
        pass

    @abstractmethod
    def format_prompt(self, prompt: PromptType, **kwargs: Any) -> Any:
        """
        Format a prompt for a model.

        Detailed description of what the method does, including:
        - Formats a prompt according to model requirements
        - Applies any necessary transformations
        - Ensures the prompt is in the correct format
        - Returns a formatted prompt ready for the model

        Args:
            prompt: The prompt to format
            **kwargs: Additional prompt formatting parameters

        Returns:
            A formatted prompt

        Raises:
            ValueError: If the prompt is invalid

        Example:
            ```python
            # Format a prompt for a model
            formatted_prompt = manager.format_prompt(
                prompt="Write a story",
                model="gpt-4"
            )
            print(f"Formatted prompt: {formatted_prompt}")
            ```
        """
        pass

    @abstractmethod
    def validate_prompt(self, prompt: PromptType) -> bool:
        """
        Validate a prompt.

        Detailed description of what the method does, including:
        - Validates a prompt against requirements
        - Checks for completeness and correctness
        - Ensures the prompt meets quality standards
        - Returns a boolean indicating validity

        Args:
            prompt: The prompt to validate

        Returns:
            True if the prompt is valid, False otherwise

        Raises:
            ValueError: If the prompt is invalid

        Example:
            ```python
            # Validate a prompt
            is_valid = manager.validate_prompt(
                prompt="Write a story"
            )
            print(f"Prompt is valid: {is_valid}")
            ```
        """
        pass


@runtime_checkable
class ValidationManagerProtocol(Protocol[InputType, ValidationResultType]):
    """
    Interface for validation managers.

    Detailed description of what the class does, including:
    - Defines the contract for components that manage validation
    - Ensures validation managers can validate inputs against rules
    - Handles rule registration and management
    - Aggregates validation results consistently

    Type parameters:
        InputType: The type of input validated by this protocol
        ValidationResultType: The type of validation result produced

    Example:
        ```python
        class SimpleValidationManager(ValidationManagerProtocol[str, Dict[str, Any]]):
            def validate(self, input_value: str) -> Dict[str, Any]:
                return {"valid": True, "message": "Input is valid"}

            def add_rule(self, rule: Any) -> None:
                pass

            def remove_rule(self, rule_name: str) -> None:
                pass

            def get_rules(self) -> List[Any]:
                return []
        ```
    """

    @abstractmethod
    def validate(self, input_value: InputType) -> ValidationResultType:
        """
        Validate an input value against registered rules.

        Detailed description of what the method does, including:
        - Validates an input value against all registered rules
        - Aggregates validation results from multiple rules
        - Provides detailed feedback on validation failures
        - Returns a structured validation result

        Args:
            input_value: The input value to validate

        Returns:
            A validation result

        Raises:
            ValueError: If the input value is invalid

        Example:
            ```python
            # Validate an input
            result = manager.validate(
                input_value="test input"
            )
            print(f"Validation result: {result}")
            ```
        """
        pass

    @abstractmethod
    def add_rule(self, rule: Any) -> None:
        """
        Add a rule for validation.

        Detailed description of what the method does, including:
        - Registers a new validation rule
        - Validates the rule before adding
        - Ensures no duplicate rules
        - Updates validation configuration

        Args:
            rule: The rule to add

        Raises:
            ValueError: If the rule is invalid

        Example:
            ```python
            # Add a validation rule
            manager.add_rule(
                rule={"name": "length", "min": 5, "max": 100}
            )
            ```
        """
        pass

    @abstractmethod
    def remove_rule(self, rule_name: str) -> None:
        """
        Remove a rule from validation.

        Detailed description of what the method does, including:
        - Removes a registered validation rule
        - Handles rule lookup and removal
        - Updates validation configuration
        - Ensures clean removal of rule

        Args:
            rule_name: The name of the rule to remove

        Raises:
            ValueError: If the rule is not found

        Example:
            ```python
            # Remove a validation rule
            manager.remove_rule(
                rule_name="length"
            )
            ```
        """
        pass

    @abstractmethod
    def get_rules(self) -> List[Any]:
        """
        Get all registered rules.

        Detailed description of what the method does, including:
        - Retrieves all registered validation rules
        - Returns rules in a consistent format
        - Provides access to current validation configuration
        - Maintains rule order and structure

        Returns:
            A list of registered rules

        Example:
            ```python
            # Get all validation rules
            rules = manager.get_rules()
            print(f"Registered rules: {rules}")
            ```
        """
        pass
