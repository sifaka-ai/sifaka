"""
OpenAI model implementation for Sifaka.

This module provides an implementation of the Model interface using OpenAI's API.
"""

import logging
from typing import Optional

from openai import OpenAI

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    OpenAI model implementation.

    Uses OpenAI's API to generate text based on prompts.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the OpenAI model.

        Args:
            model_name: Name of the OpenAI model to use.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
            system_prompt: Optional system prompt to use for all generations.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt or "You are a helpful assistant."

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"openai-{self.model_name}"

    def generate(self, thought: Thought) -> str:
        """
        Generate text based on the prompt and context in the thought.

        Args:
            thought: The thought containing the prompt and context.

        Returns:
            The generated text.
        """
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add retrieved context if available
        if thought.retrieved_context:
            context_message = "Here is some relevant context:\n\n"
            for context in thought.retrieved_context:
                context_message += f"--- {context.source} ---\n{context.content}\n\n"

            messages.append({"role": "system", "content": context_message})

        # Add validation results if available
        if thought.validation_results:
            validation_message = (
                "IMPORTANT: The previous response failed validation for the following reasons:\n\n"
            )
            for result in thought.validation_results:
                if not result.passed:
                    validation_message += f"- {result.validator_name}: {result.message}\n"

            validation_message += "\nYou MUST address ALL validation issues in your new response."

            messages.append({"role": "system", "content": validation_message})

        # Add critic feedback if available
        if thought.critic_feedback:
            feedback_message = "Here is feedback on how to improve the response:\n\n"
            for feedback in thought.critic_feedback:
                feedback_message += f"--- {feedback.critic_name} ---\n{feedback.feedback}\n\n"
                if feedback.suggestions:
                    feedback_message += "Suggestions:\n"
                    for suggestion in feedback.suggestions:
                        feedback_message += f"- {suggestion}\n"
                    feedback_message += "\n"

            messages.append({"role": "system", "content": feedback_message})

        # Add user prompt
        messages.append({"role": "user", "content": thought.prompt})

        # Add previous response if available
        if thought.text and thought.history:
            messages.append({"role": "assistant", "content": thought.text})

        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Log token usage
            if hasattr(response, "usage"):
                logger.info(
                    f"Token usage: {response.usage.prompt_tokens} prompt, "
                    f"{response.usage.completion_tokens} completion, "
                    f"{response.usage.total_tokens} total"
                )

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
