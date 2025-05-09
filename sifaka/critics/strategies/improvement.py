"""
Improvement strategies for critics.

This module provides strategies for improving text based on feedback or violations.
These strategies implement different approaches to text improvement, such as
direct improvement, iterative improvement, and reflection-based improvement.

## Strategy Overview

The module provides several key strategies:

1. **ImprovementStrategy**: Base protocol for improvement strategies
2. **DefaultImprovementStrategy**: Default implementation of improvement strategy
3. **IterativeImprovementStrategy**: Strategy that improves text iteratively
4. **ReflectionImprovementStrategy**: Strategy that uses reflections for improvement

## Strategy Lifecycle

Each strategy defines a specific lifecycle for its operations:

1. **Initialization**
   - Strategy implementation
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
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from ..interfaces.critic import LLMProvider


@runtime_checkable
class ImprovementStrategy(Protocol):
    """Protocol for improvement strategies.

    This protocol defines the interface for improvement strategies.
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
    """

    def improve(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
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


class DefaultImprovementStrategy:
    """Default implementation of improvement strategy.

    This class provides a default implementation of the improvement strategy
    protocol. It uses a language model to improve text based on feedback.

    ## Lifecycle Management

    The DefaultImprovementStrategy manages its lifecycle through three main phases:

    1. **Initialization**
       - Validates model provider
       - Sets up resources
       - Initializes state
       - Configures error handling

    2. **Operation**
       - Processes text and feedback
       - Generates improved text
       - Handles errors
       - Returns results

    3. **Cleanup**
       - Releases resources
       - Clears state
       - Logs results
       - Handles errors
    """

    def __init__(self, model_provider: LLMProvider):
        """
        Initialize the improvement strategy.

        Args:
            model_provider: The language model provider to use

        Raises:
            ValueError: If model_provider is None
            TypeError: If model_provider is not a valid provider
        """
        if model_provider is None:
            raise ValueError("model_provider cannot be None")
        self._model = model_provider

    def improve(self, text: str, feedback: Union[str, List[Dict[str, Any]]]) -> str:
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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        # Convert violations to feedback if needed
        if isinstance(feedback, list):
            feedback = self._violations_to_feedback(feedback)
        elif not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Create improvement prompt
        improvement_prompt = self._create_improvement_prompt(text, feedback)

        try:
            # Get response from the model
            response = self._model.generate(improvement_prompt)

            # Parse the response
            improved_text = self._parse_improvement_response(response)

            return improved_text
        except Exception as e:
            raise RuntimeError(f"Failed to improve text: {str(e)}")

    def _violations_to_feedback(self, violations: List[Dict[str, Any]]) -> str:
        """
        Convert violations to feedback.

        Args:
            violations: List of rule violations

        Returns:
            str: Feedback string
        """
        if not violations:
            return "No specific issues to address."

        feedback = "Please address the following issues:\n"
        for i, violation in enumerate(violations, 1):
            rule = violation.get("rule", f"Issue {i}")
            description = violation.get("description", "No description provided")
            feedback += f"{i}. {rule}: {description}\n"

        return feedback

    def _create_improvement_prompt(self, text: str, feedback: str) -> str:
        """
        Create an improvement prompt.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            str: The improvement prompt
        """
        return (
            "Please improve the following text based on the feedback provided.\n\n"
            f"TEXT:\n{text}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            "IMPROVED_TEXT:"
        )

    def _parse_improvement_response(self, response: str) -> str:
        """
        Parse the improvement response.

        Args:
            response: The response from the model

        Returns:
            str: The improved text
        """
        if "IMPROVED_TEXT:" in response:
            parts = response.split("IMPROVED_TEXT:")
            if len(parts) > 1:
                return parts[1].strip()
        return response.strip()
