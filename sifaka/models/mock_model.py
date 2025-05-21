"""
Mock model implementation for Sifaka.

This module provides a mock implementation of the Model interface for testing.
"""

import logging
from typing import Any, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought

logger = logging.getLogger(__name__)


class MockModel(Model):
    """
    Mock model implementation for testing.

    This model returns predefined responses for testing without making API calls.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        response_template: Optional[str] = None,
        **options: Any,
    ):
        """
        Initialize the mock model.

        Args:
            model_name: Name of the mock model.
            response_template: Template for the response. If None, a default template is used.
            **options: Additional options (ignored).
        """
        self.model_name = model_name
        self.response_template = (
            response_template or "Mock response from {model_name} for prompt: {prompt}"
        )
        self.options = options

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"mock-{self.model_name}"

    def generate(self, thought: Thought) -> str:
        """
        Generate text based on the prompt and context in the thought.

        Args:
            thought: The thought containing the prompt and context.

        Returns:
            The generated text.
        """
        logger.info(f"Generating mock response for prompt: {thought.prompt[:50]}...")

        # Format the response template
        response = self.response_template.format(model_name=self.model_name, prompt=thought.prompt)

        # Add information about context if available
        if thought.retrieved_context:
            context_info = "\n\nContext information:"
            for i, context in enumerate(thought.retrieved_context):
                context_info += (
                    f"\n- Context {i+1} from {context.source}: {context.content[:50]}..."
                )
            response += context_info

        # Add information about validation results if available
        if thought.validation_results:
            validation_info = "\n\nValidation results:"
            for i, result in enumerate(thought.validation_results):
                status = "passed" if result.passed else "failed"
                validation_info += f"\n- Validation {i+1} ({result.validator_name}): {status}"
            response += validation_info

        # Add information about critic feedback if available
        if thought.critic_feedback:
            feedback_info = "\n\nCritic feedback:"
            for i, feedback in enumerate(thought.critic_feedback):
                feedback_info += (
                    f"\n- Feedback {i+1} from {feedback.critic_name}: {feedback.feedback[:50]}..."
                )
            response += feedback_info

        return response
