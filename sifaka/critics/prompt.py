"""
Prompt-based critic for Sifaka.

This module provides a critic that uses custom prompts to improve text.

This is a general-purpose critic that can be customized with different prompts
to implement various critique and improvement strategies. It serves as a flexible
foundation for more specialized critics.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver
from sifaka.utils.error_handling import critic_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class PromptCritic(Critic):
    """Critic that uses custom prompts to improve text.

    This critic uses custom prompts to critique and improve text. It allows
    for flexible customization of the critique and improvement process.

    Attributes:
        model: The model to use for critiquing and improving text.
        critique_prompt_template: The template to use for critiquing text.
        improvement_prompt_template: The template to use for improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        critique_prompt_template: Optional[str] = None,
        improvement_prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the prompt critic.

        Args:
            model: The model to use for critiquing and improving text.
            critique_prompt_template: The template to use for critiquing text.
            improvement_prompt_template: The template to use for improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        start_time = time.time()

        # Log initialization attempt
        logger.debug(
            f"Initializing PromptCritic with model={model.__class__.__name__ if model else None}, "
            f"temperature={temperature}"
        )

        try:
            # Validate parameters
            if not model:
                logger.error("No model provided to PromptCritic")
                raise ImproverError(
                    message="Model must be provided",
                    component="PromptCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid model instance",
                        "Check that the model implements the Model protocol",
                    ],
                    metadata={"temperature": temperature},
                )

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in improving text. "
                    "Your goal is to provide detailed feedback and suggestions for improvement."
                )
                logger.debug("Using default system prompt for PromptCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="PromptCritic",
                operation="initialization",
                message_prefix="Failed to initialize PromptCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "temperature": temperature,
                    "has_critique_template": critique_prompt_template is not None,
                    "has_improvement_template": improvement_prompt_template is not None,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Use default critique prompt template if not provided
            self.critique_prompt_template = critique_prompt_template or (
                "Please analyze the following text and provide a critique:\n\n"
                "```\n{text}\n```\n\n"
                "Provide your analysis in JSON format with the following fields:\n"
                '- "needs_improvement": boolean indicating whether the text needs improvement\n'
                '- "message": a brief summary of your analysis\n'
                '- "issues": a list of specific issues found\n'
                '- "suggestions": a list of suggestions for improvement\n\n'
                "JSON response:"
            )
            logger.debug(
                f"Using {'custom' if critique_prompt_template else 'default'} critique prompt template"
            )

            # Use default improvement prompt template if not provided
            self.improvement_prompt_template = improvement_prompt_template or (
                "Please improve the following text based on the critique:\n\n"
                "Text:\n```\n{text}\n```\n\n"
                "Critique:\n{critique}\n\n"
                "Improved text:"
            )
            logger.debug(
                f"Using {'custom' if improvement_prompt_template else 'default'} improvement prompt template"
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful initialization
            logger.debug(f"Successfully initialized PromptCritic in {processing_time:.2f}ms")

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="PromptCritic", operation="initialization")

            # Re-raise as ImproverError with more context if not already an ImproverError
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize PromptCritic: {str(e)}",
                    component="PromptCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the templates are valid",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "temperature": temperature,
                        "has_critique_template": critique_prompt_template is not None,
                        "has_improvement_template": improvement_prompt_template is not None,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "processing_time_ms": processing_time,
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the critique prompt template.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        start_time = time.time()

        logger.debug(f"PromptCritic: Critiquing text of length {len(text)}")

        try:
            # Format the prompt
            prompt = self.critique_prompt_template.format(text=text)

            # Generate critique using the model
            with critic_context(
                critic_name="PromptCritic",
                operation="critique",
                message_prefix="Failed to critique text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the critique prompt template is valid",
                    "Try with a different model or temperature",
                ],
                metadata={"text_length": len(text), "temperature": self.temperature},
            ):
                response = self._generate(prompt)

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        "PromptCritic: No JSON found in critique response, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    return {
                        "needs_improvement": True,
                        "message": "Unable to parse critique response, but proceeding with improvement",
                        "issues": ["Unable to identify specific issues"],
                        "suggestions": ["General improvement"],
                        "processing_time_ms": processing_time,
                    }

                # Parse JSON
                json_str = response[json_start:json_end]
                critique = json.loads(json_str)

                # Ensure all required fields are present
                critique.setdefault("needs_improvement", True)
                critique.setdefault("message", "Text needs improvement")
                critique.setdefault("issues", [])
                critique.setdefault("suggestions", [])

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                critique["processing_time_ms"] = processing_time

                # Log successful critique
                logger.debug(
                    f"PromptCritic: Successfully critiqued text in {processing_time:.2f}ms, "
                    f"found {len(critique['issues'])} issues"
                )

                return critique

        except json.JSONDecodeError as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="PromptCritic", operation="critique_json_parse")

            # Failed to parse JSON, create a default response
            logger.warning(f"PromptCritic: Failed to parse JSON in critique response: {str(e)}")
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
                "processing_time_ms": processing_time,
                "error": str(e),
            }

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="PromptCritic", operation="critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text: {str(e)}",
                component="PromptCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the critique prompt template is valid",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

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
        start_time = time.time()

        logger.debug(
            f"PromptCritic: Improving text of length {len(text)} based on critique with "
            f"{len(critique.get('issues', []))} issues and {len(critique.get('suggestions', []))} suggestions"
        )

        try:
            # Format critique as a string
            critique_str = f"Issues:\n"
            for issue in critique.get("issues", []):
                critique_str += f"- {issue}\n"

            critique_str += f"\nSuggestions:\n"
            for suggestion in critique.get("suggestions", []):
                critique_str += f"- {suggestion}\n"

            # Create improvement prompt
            prompt = self.improvement_prompt_template.format(
                text=text,
                critique=critique_str,
            )

            # Generate improved text using the model
            with critic_context(
                critic_name="PromptCritic",
                operation="improve",
                message_prefix="Failed to improve text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the improvement prompt template is valid",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "issues_count": len(critique.get("issues", [])),
                    "suggestions_count": len(critique.get("suggestions", [])),
                    "temperature": self.temperature,
                },
            ):
                response = self._generate(prompt)

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful improvement
                logger.debug(
                    f"PromptCritic: Successfully improved text in {processing_time:.2f}ms, "
                    f"original length: {len(text)}, improved length: {len(improved_text)}"
                )

                return improved_text

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="PromptCritic", operation="improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                component="PromptCritic",
                operation="improve",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the improvement prompt template is valid",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "issues_count": len(critique.get("issues", [])),
                    "suggestions_count": len(critique.get("suggestions", [])),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )


@register_improver("prompt")
def create_prompt_critic(
    model: Model,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> PromptCritic:
    """Create a prompt critic.

    This factory function creates a PromptCritic with the specified parameters.
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        critique_prompt_template: The template to use for critiquing text.
        improvement_prompt_template: The template to use for improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the PromptCritic.

    Returns:
        A PromptCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    start_time = time.time()

    try:
        # Log factory function call
        logger.debug(
            f"Creating PromptCritic with model={model.__class__.__name__ if model else None}, "
            f"temperature={temperature}"
        )

        # Validate parameters
        if not model:
            logger.error("No model provided to create_prompt_critic")
            raise ImproverError(
                message="Model must be provided",
                component="PromptCriticFactory",
                operation="creation",
                suggestions=[
                    "Provide a valid model instance",
                    "Check that the model implements the Model protocol",
                ],
                metadata={"temperature": temperature},
            )

        # Create the critic
        critic = PromptCritic(
            model=model,
            critique_prompt_template=critique_prompt_template,
            improvement_prompt_template=improvement_prompt_template,
            system_prompt=system_prompt,
            temperature=temperature,
            **options,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(f"Successfully created PromptCritic in {processing_time:.2f}ms")

        return critic

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="PromptCriticFactory", operation="creation")

        # Raise as ImproverError with more context if not already an ImproverError
        if not isinstance(e, ImproverError):
            raise ImproverError(
                message=f"Failed to create PromptCritic: {str(e)}",
                component="PromptCriticFactory",
                operation="creation",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the templates are valid",
                    "Check the error message for details",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "temperature": temperature,
                    "has_critique_template": critique_prompt_template is not None,
                    "has_improvement_template": improvement_prompt_template is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
