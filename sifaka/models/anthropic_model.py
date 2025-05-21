"""
Anthropic model implementation for Sifaka.

This module provides an implementation of the Model interface using Anthropic's API.
"""

import logging
from typing import Optional

import anthropic

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought

logger = logging.getLogger(__name__)


class AnthropicModel(Model):
    """
    Anthropic model implementation.

    Uses Anthropic's API to generate text based on prompts.
    """

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the Anthropic model.

        Args:
            model_name: Name of the Anthropic model to use.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            api_key: Anthropic API key. If None, uses the ANTHROPIC_API_KEY environment variable.
            system_prompt: Optional system prompt to use for all generations.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt or "You are a helpful assistant."

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"anthropic-{self.model_name}"

    def generate(self, thought: Thought) -> str:
        """
        Generate text based on the prompt and context in the thought.

        Args:
            thought: The thought containing the prompt and context.

        Returns:
            The generated text.
        """
        # Prepare messages
        messages = []

        # Add retrieved context if available
        context_parts = []
        if thought.retrieved_context:
            for context in thought.retrieved_context:
                context_parts.append(f"--- {context.source} ---\n{context.content}")

        # Add validation results if available
        validation_parts = []
        if thought.validation_results:
            for result in thought.validation_results:
                if not result.passed:
                    validation_parts.append(f"- {result.validator_name}: {result.message}")

        # Add critic feedback if available
        feedback_parts = []
        if thought.critic_feedback:
            for feedback in thought.critic_feedback:
                feedback_text = f"--- {feedback.critic_name} ---\n{feedback.feedback}"
                if feedback.suggestions:
                    feedback_text += "\nSuggestions:\n"
                    for suggestion in feedback.suggestions:
                        feedback_text += f"- {suggestion}\n"
                feedback_parts.append(feedback_text)

        # Build system prompt with context, validation results, and feedback
        full_system_prompt = self.system_prompt

        if context_parts:
            full_system_prompt += "\n\nHere is some relevant context:\n\n"
            full_system_prompt += "\n\n".join(context_parts)

        if validation_parts:
            full_system_prompt += (
                "\n\nThe previous response failed validation for the following reasons:\n\n"
            )
            full_system_prompt += "\n".join(validation_parts)

        if feedback_parts:
            full_system_prompt += "\n\nHere is feedback on how to improve the response:\n\n"
            full_system_prompt += "\n\n".join(feedback_parts)

        # Add user prompt
        messages.append({"role": "user", "content": thought.prompt})

        # Add previous response if available
        if thought.text and thought.history:
            messages.append({"role": "assistant", "content": thought.text})

            # If we have a previous response and validation failed, add the user prompt again
            if validation_parts:
                messages.append(
                    {
                        "role": "user",
                        "content": "Please revise your response based on the feedback.",
                    }
                )

        try:
            # Generate response
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                system=full_system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract generated text
            generated_text = response.content[0].text

            # Log token usage
            if hasattr(response, "usage"):
                logger.info(
                    f"Token usage: {response.usage.input_tokens} input, "
                    f"{response.usage.output_tokens} output, "
                    f"{response.usage.total_tokens} total"
                )

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise
