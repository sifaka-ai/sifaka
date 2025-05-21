"""
Base critic class for Sifaka.

This module provides the base critic class that all critics should inherit from.
It includes consistent error handling and improvement patterns for all critics.
"""

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, Optional

from sifaka.core.interfaces import Critic, Model
from sifaka.core.thought import Thought

# Configure logger
logger = logging.getLogger(__name__)


class BaseCritic(Critic):
    """
    Base class for critics that improve text.

    Critics use LLMs to critique and improve text based on specific criteria.
    This class implements the Critic interface.

    Attributes:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        name: Optional[str] = None,
        **options: Any,
    ):
        """
        Initialize the critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            name: Optional name for the critic.
            **options: Additional options for the critic.

        Raises:
            ValueError: If the model is not provided.
        """
        if not model:
            logger.error("Model not provided to critic")
            raise ValueError("Model not provided")

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._name = name or self.__class__.__name__
        self.options = options

        # Log initialization
        logger.debug(
            f"Initialized {self.name} critic with model={model.name}, "
            f"temperature={temperature}, options={options}"
        )

    @property
    def name(self) -> str:
        """Return the name of the critic."""
        return self._name

    def critique(self, thought: Thought) -> str:
        """
        Critique the text in the thought and provide feedback.

        This method handles the critique process, including:
        - Setting up error handling
        - Calling the _critique method to perform specific critique logic
        - Adding the feedback to the thought

        Subclasses should not override this method. Instead, they should
        override the _critique method to implement specific critique logic.

        Args:
            thought: The thought containing the text to critique.

        Returns:
            Feedback on how to improve the text.
        """
        start_time = time.time()
        text = thought.text

        try:
            # Handle empty text
            if not text or not text.strip():
                logger.debug(f"{self.name}: Empty text provided, returning without critique")
                feedback = "Empty text cannot be critiqued."
                thought.add_critic_feedback(
                    critic_name=self.name,
                    feedback=feedback,
                    suggestions=[],
                    details={
                        "critic_name": self.name,
                        "error_type": "EmptyText",
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                )
                return feedback

            # Perform critique
            logger.debug(f"{self.name}: Critiquing text of length {len(text)}")
            critique_result = self._critique(thought)

            # Extract feedback and suggestions
            feedback = critique_result["feedback"]
            suggestions = critique_result.get("suggestions", [])
            details = critique_result.get("details", {})

            # Add processing time to details
            details["processing_time_ms"] = (time.time() - start_time) * 1000

            # Add feedback to thought
            thought.add_critic_feedback(
                critic_name=self.name,
                feedback=feedback,
                suggestions=suggestions,
                details=details,
            )

            return feedback

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")

            # Add error feedback to thought
            error_feedback = f"Error critiquing text: {str(e)}"
            thought.add_critic_feedback(
                critic_name=self.name,
                feedback=error_feedback,
                suggestions=[],
                details={
                    "critic_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return error_feedback

    @abstractmethod
    def _critique(self, thought: Thought) -> Dict[str, Any]:
        """
        Implement specific critique logic.

        This method should be overridden by subclasses to implement
        specific critique logic. It is called by the critique method
        after handling empty text and setting up error handling.

        Args:
            thought: The thought containing the text to critique.

        Returns:
            A dictionary with critique results, including at least:
            - feedback: The critique feedback
            - suggestions: A list of suggestions for improvement (optional)
            - details: Additional details about the critique (optional)

        Raises:
            NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _critique method")

    def _generate_with_model(
        self, prompt: str, system_prompt: Optional[str] = None, **options: Any
    ) -> str:
        """
        Generate text using the model.

        This is a helper method for critics to generate text using the model.

        Args:
            prompt: The prompt to use for generation.
            system_prompt: The system prompt to use for the model.
            **options: Additional options to pass to the model.

        Returns:
            The generated text.

        Raises:
            Exception: If there is an error generating text.
        """
        try:
            # Merge options with defaults
            merged_options = {
                "temperature": self.temperature,
                **self.options,
                **options,
            }

            # Use provided system prompt or default
            if system_prompt:
                merged_options["system_prompt"] = system_prompt
            elif self.system_prompt:
                merged_options["system_prompt"] = self.system_prompt

            # Log generation attempt
            temp_value = merged_options.get("temperature")
            logger.debug(
                f"{self.name}: Generating text with prompt length={len(prompt)}, temperature={temp_value}"
            )

            # Create a temporary thought for generation
            generation_thought = Thought(prompt=prompt)

            # Generate text
            start_time = time.time()
            result = self.model.generate(generation_thought)
            generation_time = (time.time() - start_time) * 1000

            # Log generation success
            logger.debug(
                f"{self.name}: Generated text in {generation_time:.2f}ms, result length={len(result)}"
            )

            return result

        except Exception as e:
            # Log the error
            logger.error(f"Error generating text in {self.name}: {str(e)}")

            # Re-raise the exception
            raise
