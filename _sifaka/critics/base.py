"""Base critic class for Sifaka.

This module provides the base critic class that all critics should inherit from.
It includes consistent error handling and improvement patterns for all critics.

The Critic class implements the Improver protocol and provides a standard
two-step process for improving text:
1. Critique the text to identify areas for improvement
2. Improve the text based on the critique

Example:
    ```python
    from sifaka.critics.base import Critic
    from sifaka.models.openai import OpenAIModel

    class MyCritic(Critic):
        def _critique(self, text: str) -> Dict[str, Any]:
            # Implement critique logic
            return {
                "needs_improvement": True,
                "message": "Text needs improvement",
                "issues": ["Issue 1", "Issue 2"],
                "suggestions": ["Suggestion 1", "Suggestion 2"]
            }

        def _improve(self, text: str, critique: Dict[str, Any]) -> str:
            # Implement improvement logic
            return "Improved text"

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a critic
    critic = MyCritic(model=model)

    # Improve text
    improved_text, result = critic.improve("Text to improve")
    ```
"""

import logging
import time
from typing import Any, Dict, Optional

from sifaka.errors import ImproverError, ModelError
from sifaka.interfaces import Improver
from sifaka.models.base import Model
from sifaka.results import ImprovementResult as SifakaImprovementResult
from sifaka.utils.error_handling import improvement_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class Critic(Improver):
    """Base class for critics that improve text.

    Critics use LLMs to critique and improve text based on specific criteria.
    This class implements the Improver protocol.

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
            raise ImproverError(
                message="Model not provided",
                component="Critic",
                operation="initialization",
                suggestions=["Provide a valid model instance"],
            )

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.options = options
        self._name = options.get("name", self.__class__.__name__)

        # Log initialization
        logger.debug(
            f"Initialized {self._name} with model={model.__class__.__name__}, temperature={temperature}"
        )

    @property
    def name(self) -> str:
        """
        Get the name of the critic.

        Returns:
            The name of the critic
        """
        return str(self._name)

    def configure(self, **options: Any) -> None:
        """Configure the critic with new options.

        Args:
            **options: Configuration options to apply to the critic.
                Supported options include:
                - system_prompt: The system prompt to use for the model.
                - temperature: The temperature to use for the model.
                - refinement_rounds: Number of refinement rounds (for self-refine critics).
                - num_critics: Number of critics (for n-critics).
                - principles: List of principles (for constitutional critics).
                - max_passages: Maximum number of passages (for retrieval-enhanced critics).
                - include_passages_in_critique: Whether to include passages in critique.
                - include_passages_in_improve: Whether to include passages in improve.

        Raises:
            ImproverError: If there is an error configuring the critic.
        """
        logger.debug(f"Configuring {self.name} with options: {options}")

        # Update system prompt if provided
        if "system_prompt" in options:
            self.system_prompt = options["system_prompt"]
            logger.debug(f"Updated system prompt for {self.name}")

        # Update temperature if provided
        if "temperature" in options:
            self.temperature = options["temperature"]
            logger.debug(f"Updated temperature for {self.name} to {self.temperature}")

        # Update other options
        for key, value in options.items():
            if key not in ["system_prompt", "temperature"]:
                self.options[key] = value
                logger.debug(f"Updated option '{key}' for {self.name}")

        logger.debug(f"Successfully configured {self.name}")

    def improve(self, text: str) -> tuple[str, SifakaImprovementResult]:
        """Improve text using this critic.

        This method analyzes the provided text and generates an improved version
        based on the critic's improvement strategy. It follows a two-step process:
        1. Critique the text to identify areas for improvement
        2. Improve the text based on the critique

        Args:
            text (str): The text to improve.

        Returns:
            tuple[str, SifakaImprovementResult]: A tuple containing the improved text
                and a result object with details about the improvement process.

        Raises:
            ImproverError: If the improvement process encounters an error.

        Example:
            ```python
            improved_text, result = critic.improve("Text to improve")
            print(f"Original: {result.original_text}")
            print(f"Improved: {result.improved_text}")
            print(f"Changes made: {result.changes_made}")
            ```
        """
        start_time = time.time()

        if not text:
            logger.debug(f"{self.name}: Empty text provided, returning without improvement")
            return "", SifakaImprovementResult(
                _original_text="",
                _improved_text="",
                _changes_made=False,
                message="Empty text cannot be improved",
                _details={"critic_type": self.name},
            )

        try:
            # Get critique with error handling
            with improvement_context(
                improver_name=self.name,
                operation="critique",
                message_prefix="Failed to critique text",
                suggestions=["Check the model and input text"],
            ):
                critique = self._critique(text)
                logger.debug(
                    f"{self.name}: Critiqued text, needs_improvement={critique.get('needs_improvement', True)}"
                )

            # Check if improvement is needed
            if not critique.get("needs_improvement", True):
                logger.debug(f"{self.name}: No improvement needed")
                return text, SifakaImprovementResult(
                    _original_text=text,
                    _improved_text=text,
                    _changes_made=False,
                    message=critique.get("message", "No improvement needed"),
                    _details={
                        "critic_type": self.name,
                        "critique": critique,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                )

            # Improve text with error handling
            with improvement_context(
                improver_name=self.name,
                operation="improve",
                message_prefix="Failed to improve text",
                suggestions=["Check the model and critique"],
            ):
                improved_text = self._improve(text, critique)
                logger.debug(
                    f"{self.name}: Improved text, length before={len(text)}, length after={len(improved_text)}"
                )

            # Check if text was actually improved
            if improved_text == text:
                logger.debug(f"{self.name}: No changes were made to the text")
                return text, SifakaImprovementResult(
                    _original_text=text,
                    _improved_text=text,
                    _changes_made=False,
                    message="No changes were made",
                    _details={
                        "critic_type": self.name,
                        "critique": critique,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                )

            # Return improved text with result
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"{self.name}: Successfully improved text in {processing_time:.2f}ms")
            return improved_text, SifakaImprovementResult(
                _original_text=text,
                _improved_text=improved_text,
                _changes_made=True,
                message=critique.get("message", "Text improved"),
                _details={
                    "critic_type": self.name,
                    "critique": critique,
                    "processing_time_ms": processing_time,
                },
            )
        except ModelError as e:
            # Log the error
            log_error(e, logger, component="Critic", operation=f"{self.name}.improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="improve",
                suggestions=[
                    "Check the model configuration",
                    "Verify API keys and quotas",
                ],
                metadata={
                    "critic_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        except Exception as e:
            # Log the error
            log_error(e, logger, component="Critic", operation=f"{self.name}.improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Unexpected error improving text: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="improve",
                suggestions=[
                    "Check the input text format",
                    "Verify critic implementation",
                ],
                metadata={
                    "critic_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    def _critique(self, text: str) -> Dict[str, Any]:  # noqa
        """Critique text based on specific criteria.

        This method analyzes the provided text and identifies areas for improvement.
        It should be implemented by subclasses to provide specific critique logic.

        Args:
            text (str): The text to critique.

        Returns:
            Dict[str, Any]: A dictionary with critique information, typically including:
                - needs_improvement (bool): Whether the text needs improvement
                - message (str): A summary of the critique
                - issues (List[str]): Specific issues identified
                - suggestions (List[str]): Suggestions for improvement

        Raises:
            ImproverError: If the text cannot be critiqued.
            NotImplementedError: If the method is not implemented by a subclass.
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("_critique method must be implemented by subclasses")

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:  # noqa
        """Improve text based on critique.

        This method generates an improved version of the text based on the critique
        information. It should be implemented by subclasses to provide specific
        improvement logic.

        Args:
            text (str): The text to improve.
            critique (Dict[str, Any]): The critique information from the _critique method.

        Returns:
            str: The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
            NotImplementedError: If the method is not implemented by a subclass.
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
            ImproverError: If there is an unexpected error during generation.
        """
        try:
            # Merge default options with provided options
            merged_options = {
                "temperature": self.temperature,
                **self.options,
                **options,
            }

            # Add system prompt if provided and not already in options
            if self.system_prompt and "system_prompt" not in merged_options:
                merged_options["system_prompt"] = self.system_prompt

            # Convert stop_sequences to stop for OpenAI compatibility
            if "stop_sequences" in merged_options:
                merged_options["stop"] = merged_options.pop("stop_sequences")

            # Log generation attempt
            temp_value = merged_options.get("temperature")
            logger.debug(
                f"{self.name}: Generating text with prompt length={len(prompt)}, temperature={temp_value}"
            )

            # Generate text
            start_time = time.time()
            result = self.model.generate(prompt, **merged_options)
            generation_time = (time.time() - start_time) * 1000

            # Log generation success
            logger.debug(
                f"{self.name}: Generated text in {generation_time:.2f}ms, result length={len(result)}"
            )

            return result

        except ModelError as e:
            # Log the error
            log_error(e, logger, component="Critic", operation=f"{self.name}.generate")

            # Re-raise ModelError (will be caught by improve method)
            raise

        except Exception as e:
            # Log the error
            log_error(e, logger, component="Critic", operation=f"{self.name}.generate")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Unexpected error generating text: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="generate",
                suggestions=["Check the model configuration", "Verify prompt format"],
                metadata={
                    "critic_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "prompt_length": len(prompt),
                },
            )
