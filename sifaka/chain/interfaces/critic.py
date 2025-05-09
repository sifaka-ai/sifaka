"""
Critic Interface Module

Protocol interfaces for Sifaka's critic system.

## Overview
This module defines the interfaces for critics in the Sifaka framework.
These interfaces establish a common contract for critic behavior, enabling better
modularity and extensibility.

## Components
1. **CriticProtocol**: Base interface for critics
   - Output evaluation
   - Feedback generation
   - Improvement suggestions

## Usage Examples
```python
from typing import Any, Dict
from sifaka.chain.interfaces.critic import CriticProtocol

class SimpleCritic(CriticProtocol[str, str, Dict[str, Any]]):
    def evaluate(self, input_value: str, output_value: str, **kwargs: Any) -> bool:
        # Evaluate the output
        return len(output_value) > 10

    def get_feedback(self, input_value: str, output_value: str, **kwargs: Any) -> Dict[str, Any]:
        # Get feedback on the output
        return {
            "feedback": "Output is too short",
            "suggestions": ["Add more details", "Expand on key points"]
        }

    def improve(self, input_value: str, output_value: str, feedback: Dict[str, Any], **kwargs: Any) -> str:
        # Improve the output based on feedback
        return f"{output_value}\nAdditional details..."
```

## Error Handling
- ValueError: Raised when input or output values are invalid
- RuntimeError: Raised when critic operations fail

## Configuration
- input_value: The original input value
- output_value: The output value to evaluate/improve
- feedback: Feedback on the output value
- kwargs: Additional parameters for operations
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)
FeedbackType = TypeVar("FeedbackType", covariant=True)


@runtime_checkable
class CriticProtocol(Protocol[InputType, OutputType, FeedbackType]):
    """
    Interface for critics.

    Detailed description of what the class does, including:
    - Defines the contract for components that provide feedback on outputs
    - Ensures critics can evaluate outputs and provide standardized feedback
    - Handles output evaluation, feedback generation, and improvement suggestions
    - Maintains consistent behavior across different critic implementations

    Type parameters:
        InputType: The type of input accepted by the critic
        OutputType: The type of output evaluated by the critic
        FeedbackType: The type of feedback provided by the critic

    Example:
        ```python
        class SimpleCritic(CriticProtocol[str, str, Dict[str, Any]]):
            def evaluate(self, input_value: str, output_value: str, **kwargs: Any) -> bool:
                # Evaluate the output
                return len(output_value) > 10

            def get_feedback(self, input_value: str, output_value: str, **kwargs: Any) -> Dict[str, Any]:
                # Get feedback on the output
                return {
                    "feedback": "Output is too short",
                    "suggestions": ["Add more details", "Expand on key points"]
                }

            def improve(self, input_value: str, output_value: str, feedback: Dict[str, Any], **kwargs: Any) -> str:
                # Improve the output based on feedback
                return f"{output_value}\nAdditional details..."
        ```
    """

    @abstractmethod
    def evaluate(self, input_value: InputType, output_value: OutputType, **kwargs: Any) -> bool:
        """
        Evaluate an output value.

        Detailed description of what the method does, including:
        - Evaluates whether an output value meets quality criteria
        - Considers the original input value for context
        - Returns a boolean indicating acceptability
        - Handles validation of input and output values

        Args:
            input_value: The original input value
            output_value: The output value to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            True if the output is acceptable, False otherwise

        Raises:
            ValueError: If the input or output value is invalid

        Example:
            ```python
            # Evaluate an output
            is_acceptable = critic.evaluate(
                input_value="Write a story",
                output_value="Once upon a time..."
            )
            print(f"Output is acceptable: {is_acceptable}")
            ```
        """
        pass

    @abstractmethod
    def get_feedback(
        self, input_value: InputType, output_value: OutputType, **kwargs: Any
    ) -> FeedbackType:
        """
        Get feedback on an output value.

        Detailed description of what the method does, including:
        - Analyzes an output value and generates feedback
        - Considers the original input value for context
        - Provides specific suggestions for improvement
        - Returns structured feedback data

        Args:
            input_value: The original input value
            output_value: The output value to get feedback on
            **kwargs: Additional feedback parameters

        Returns:
            Feedback on the output value

        Raises:
            ValueError: If the input or output value is invalid

        Example:
            ```python
            # Get feedback on an output
            feedback = critic.get_feedback(
                input_value="Write a story",
                output_value="Once upon a time..."
            )
            print(f"Feedback: {feedback}")
            ```
        """
        pass

    @abstractmethod
    def improve(
        self,
        input_value: InputType,
        output_value: OutputType,
        feedback: FeedbackType,
        **kwargs: Any,
    ) -> OutputType:
        """
        Suggest improvements for an output value.

        Detailed description of what the method does, including:
        - Takes an output value and feedback to generate improvements
        - Considers the original input value for context
        - Applies suggested improvements to the output
        - Returns an improved version of the output

        Args:
            input_value: The original input value
            output_value: The output value to improve
            feedback: Feedback on the output value
            **kwargs: Additional improvement parameters

        Returns:
            An improved output value

        Raises:
            ValueError: If the input or output value is invalid

        Example:
            ```python
            # Improve an output based on feedback
            improved_output = critic.improve(
                input_value="Write a story",
                output_value="Once upon a time...",
                feedback={"suggestions": ["Add more details"]}
            )
            print(f"Improved output: {improved_output}")
            ```
        """
        pass
