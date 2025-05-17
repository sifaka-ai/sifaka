"""
Base critic class for Sifaka.

This module provides the base critic class that all critics should inherit from.
"""

from typing import Optional, Dict, Any, List, Union

from sifaka.models.base import Model
from sifaka.results import ImprovementResult
from sifaka.errors import ImproverError, ModelError


class Critic:
    """Base class for critics that improve text.

    Critics use LLMs to critique and improve text based on specific criteria.

    Attributes:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided.
        """
        if not model:
            raise ImproverError("Model not provided")

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.options = options

    def improve(self, text: str) -> tuple[str, ImprovementResult]:
        """Improve text based on specific criteria.

        Args:
            text: The text to improve.

        Returns:
            A tuple of (improved_text, improvement_result).

        Raises:
            ImproverError: If the text cannot be improved.
        """
        if not text:
            return "", ImprovementResult(
                original_text="",
                improved_text="",
                changes_made=False,
                message="Empty text cannot be improved",
                details={"critic_type": self.__class__.__name__},
            )

        try:
            # Get critique
            critique = self._critique(text)

            # Check if improvement is needed
            if not critique["needs_improvement"]:
                return text, ImprovementResult(
                    original_text=text,
                    improved_text=text,
                    changes_made=False,
                    message=critique["message"],
                    details={
                        "critic_type": self.__class__.__name__,
                        "critique": critique,
                    },
                )

            # Improve text
            improved_text = self._improve(text, critique)

            # Check if text was actually improved
            if improved_text == text:
                return text, ImprovementResult(
                    original_text=text,
                    improved_text=text,
                    changes_made=False,
                    message="No changes were made",
                    details={
                        "critic_type": self.__class__.__name__,
                        "critique": critique,
                    },
                )

            return improved_text, ImprovementResult(
                original_text=text,
                improved_text=improved_text,
                changes_made=True,
                message=critique["message"],
                details={
                    "critic_type": self.__class__.__name__,
                    "critique": critique,
                },
            )
        except ModelError as e:
            raise ImproverError(f"Error improving text: {str(e)}")
        except Exception as e:
            raise ImproverError(f"Unexpected error improving text: {str(e)}")

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text based on specific criteria.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("_critique method must be implemented by subclasses")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text based on critique.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("_improve method must be implemented by subclasses")

    def _generate(self, prompt: str, **options: Any) -> str:
        """Generate text using the model.

        Args:
            prompt: The prompt to use.
            **options: Additional options to pass to the model.

        Returns:
            The generated text.

        Raises:
            ModelError: If the text cannot be generated.
        """
        # Merge default options with provided options
        merged_options = {
            "temperature": self.temperature,
            **self.options,
            **options,
        }

        # Generate text
        return self.model.generate(prompt, **merged_options)
