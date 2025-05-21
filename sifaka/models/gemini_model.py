"""
Google Gemini model implementation for Sifaka.

This module provides an implementation of the Model interface using Google's Gemini API.
"""

import logging
import os
from typing import Optional

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from core.interfaces import Model
from core.thought import Thought
from models.factory import ModelConfigurationError, ModelError

logger = logging.getLogger(__name__)


class GeminiModel(Model):
    """
    Google Gemini model implementation.

    Uses Google's Gemini API to generate text based on prompts.
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the Gemini model.

        Args:
            model_name: Name of the Gemini model to use.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum number of tokens to generate.
            api_key: Google API key. If None, uses the GOOGLE_API_KEY environment variable.
            system_prompt: Optional system prompt to use for all generations.
        """
        # Check if Gemini package is available
        if not GEMINI_AVAILABLE:
            raise ModelConfigurationError(
                "Google Generative AI package not installed. Install it with 'pip install google-generativeai'."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise ModelError(
                "Google API key not provided. Either pass it as an argument or set the GOOGLE_API_KEY environment variable."
            )

        self.system_prompt = system_prompt or "You are a helpful assistant."

        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)

            # Get the model
            self.model = genai.GenerativeModel(model_name=self.model_name)

            logger.info(f"Successfully initialized Gemini model '{model_name}'")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise ModelError(f"Error initializing Gemini model: {str(e)}")

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"gemini-{self.model_name}"

    def generate(self, thought: Thought) -> str:
        """
        Generate text based on the prompt and context in the thought.

        Args:
            thought: The thought containing the prompt and context.

        Returns:
            The generated text.
        """
        # Prepare system prompt with context, validation results, and feedback
        full_system_prompt = self.system_prompt

        # Add retrieved context if available
        if thought.retrieved_context:
            full_system_prompt += "\n\nHere is some relevant context:\n\n"
            for context in thought.retrieved_context:
                full_system_prompt += f"--- {context.source} ---\n{context.content}\n\n"

        # Add validation results if available
        if thought.validation_results:
            full_system_prompt += (
                "\n\nThe previous response failed validation for the following reasons:\n\n"
            )
            for result in thought.validation_results:
                if not result.passed:
                    full_system_prompt += f"- {result.validator_name}: {result.message}\n"

        # Add critic feedback if available
        if thought.critic_feedback:
            full_system_prompt += "\n\nHere is feedback on how to improve the response:\n\n"
            for feedback in thought.critic_feedback:
                full_system_prompt += f"--- {feedback.critic_name} ---\n{feedback.feedback}\n\n"
                if feedback.suggestions:
                    full_system_prompt += "Suggestions:\n"
                    for suggestion in feedback.suggestions:
                        full_system_prompt += f"- {suggestion}\n"
                    full_system_prompt += "\n"

        # Prepare the prompt with system instructions
        prompt = f"{full_system_prompt}\n\nUser: {thought.prompt}"

        # Add previous response if available
        if thought.text and thought.history:
            prompt += f"\n\nYour previous response: {thought.text}"

            # If we have validation results, add a request to revise
            if any(not result.passed for result in thought.validation_results):
                prompt += "\n\nPlease revise your response based on the feedback."

        try:
            # Set up generation config
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }

            # Generate response
            response = self.model.generate_content(
                prompt, generation_config=genai.types.GenerationConfig(**generation_config)
            )

            # Extract generated text
            generated_text = response.text

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise ModelError(f"Error generating text with Gemini: {str(e)}")
