"""
Critic protocol interfaces for Sifaka.

This module defines the interfaces for critics in the Sifaka framework.
These interfaces establish a common contract for critic behavior, enabling better
modularity and extensibility.
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

    This interface defines the contract for components that provide feedback
    on outputs and suggest improvements. It ensures that critics can evaluate
    outputs and provide standardized feedback.

    ## Lifecycle

    1. **Initialization**: Set up critic resources
    2. **Evaluation**: Evaluate outputs and provide feedback
    3. **Improvement**: Suggest improvements for outputs
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an evaluate method to evaluate outputs
    - Provide a get_feedback method to get feedback on outputs
    - Provide an improve method to suggest improvements for outputs
    """

    @abstractmethod
    def evaluate(self, input_value: InputType, output_value: OutputType, **kwargs: Any) -> bool:
        """
        Evaluate an output value.

        Args:
            input_value: The original input value
            output_value: The output value to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            True if the output is acceptable, False otherwise

        Raises:
            ValueError: If the input or output value is invalid
        """
        pass

    @abstractmethod
    def get_feedback(self, input_value: InputType, output_value: OutputType, **kwargs: Any) -> FeedbackType:
        """
        Get feedback on an output value.

        Args:
            input_value: The original input value
            output_value: The output value to get feedback on
            **kwargs: Additional feedback parameters

        Returns:
            Feedback on the output value

        Raises:
            ValueError: If the input or output value is invalid
        """
        pass

    @abstractmethod
    def improve(self, input_value: InputType, output_value: OutputType, feedback: FeedbackType, **kwargs: Any) -> OutputType:
        """
        Suggest improvements for an output value.

        Args:
            input_value: The original input value
            output_value: The output value to improve
            feedback: Feedback on the output value
            **kwargs: Additional improvement parameters

        Returns:
            An improved output value

        Raises:
            ValueError: If the input or output value is invalid
        """
        pass
