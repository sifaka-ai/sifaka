"""
Constitutional critic for Sifaka.

This module provides a critic that evaluates text against a set of principles.

Based on the paper:
"Constitutional AI: Harmlessness from AI Feedback"
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson,
Christopher Olah, Circulation, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Hendrycks,
Ethan Perez, Jamie Kerr, Jared Kaplan, Jeremie M. Harris, Joseph Gonzalez, Josh Landau,
Liane Lovitt, Michael Sellitto, Miles Brundage, Pamela Mishkin, Paul Christiano, Rachel Hao,
Raphael Millière, Sam Bowman, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Scott Sievert,
Sheer El-Showk, Stanislav Fort, Timothy Telleen-Lawton, Thomas Langlois, Tyna Eloundou,
Varun Sundar, Yuntao Bai, Zac Hatfield-Dodds
arXiv:2212.08073 [cs.CL]
https://arxiv.org/abs/2212.08073
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.models.base import Model
from sifaka.registry import register_improver
from sifaka.utils.error_handling import critic_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class ConstitutionalCritic(Critic):
    """Critic that evaluates text against a set of principles.

    This critic evaluates text against a set of principles (a "constitution")
    and provides feedback when violations are detected.

    Attributes:
        model: The model to use for critiquing and improving text.
        principles: The principles to evaluate text against.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        principles: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the constitutional critic.

        Args:
            model: The model to use for critiquing and improving text.
            principles: The principles to evaluate text against.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        start_time = time.time()

        # Log initialization attempt
        logger.debug(
            f"Initializing ConstitutionalCritic with model={model.__class__.__name__ if model else None}, "
            f"principles_count={len(principles) if principles else 'default'}, "
            f"temperature={temperature}"
        )

        try:
            # Validate parameters
            if not model:
                logger.error("No model provided to ConstitutionalCritic")
                raise ImproverError(
                    message="Model must be provided",
                    component="ConstitutionalCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid model instance",
                        "Check that the model implements the Model protocol",
                    ],
                    metadata={
                        "principles_count": len(principles) if principles else 0,
                        "temperature": temperature,
                    },
                )

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in evaluating text against principles. "
                    "Your goal is to identify violations of principles and provide feedback for improvement."
                )
                logger.debug("Using default system prompt for ConstitutionalCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="ConstitutionalCritic",
                operation="initialization",
                message_prefix="Failed to initialize ConstitutionalCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "principles_count": len(principles) if principles else 0,
                    "temperature": temperature,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Use default principles if not provided
            self.principles = principles or [
                "The text should be clear and concise.",
                "The text should be grammatically correct.",
                "The text should be well-structured.",
                "The text should be factually accurate.",
                "The text should be appropriate for the intended audience.",
            ]

            # Validate principles
            if not self.principles:
                logger.warning("Empty principles list provided, using default principles")
                self.principles = [
                    "The text should be clear and concise.",
                    "The text should be grammatically correct.",
                    "The text should be well-structured.",
                    "The text should be factually accurate.",
                    "The text should be appropriate for the intended audience.",
                ]

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful initialization
            logger.debug(
                f"Successfully initialized ConstitutionalCritic with {len(self.principles)} principles "
                f"in {processing_time:.2f}ms"
            )

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="ConstitutionalCritic", operation="initialization")

            # Re-raise as ImproverError with more context if not already an ImproverError
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize ConstitutionalCritic: {str(e)}",
                    component="ConstitutionalCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the principles are well-defined",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "principles_count": len(principles) if principles else 0,
                        "temperature": temperature,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "processing_time_ms": processing_time,
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text against the principles.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        start_time = time.time()

        # Log critique attempt
        logger.debug(
            f"ConstitutionalCritic: Critiquing text of length {len(text)} against {len(self.principles)} principles"
        )

        # Format principles as a bulleted list
        principles_str = "\n".join(f"- {p}" for p in self.principles)

        prompt = f"""
        Please evaluate the following text against these principles:

        Principles:
        {principles_str}

        Text:
        ```
        {text}
        ```

        Provide your evaluation in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text violates any principles
        - "message": a brief summary of your evaluation
        - "violations": a list of specific principle violations
        - "suggestions": a list of suggestions for improvement

        JSON response:
        """

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="ConstitutionalCritic",
                operation="critique",
                message_prefix="Failed to critique text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the principles are well-defined",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "principles_count": len(self.principles),
                    "temperature": self.temperature,
                },
            ):
                # Generate critique
                response = self._generate(prompt)

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        "ConstitutionalCritic: No JSON found in critique response, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    # Create a Dict[str, Any] to return
                    default_critique: Dict[str, Any] = {
                        "needs_improvement": True,
                        "message": "Unable to parse critique response, but proceeding with improvement",
                        "violations": ["Unable to identify specific violations"],
                        "suggestions": ["General improvement"],
                        "processing_time_ms": processing_time,
                    }
                    return default_critique

                # Parse JSON
                json_str = response[json_start:json_end]
                critique_data = json.loads(json_str)

                # Ensure all required fields are present
                critique_data.setdefault("needs_improvement", True)
                critique_data.setdefault("message", "Text needs improvement")
                critique_data.setdefault("violations", [])
                critique_data.setdefault("suggestions", [])

                # Add issues field for compatibility with base Critic
                critique_data["issues"] = critique_data.get("violations", [])

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                critique_data["processing_time_ms"] = processing_time

                # Log successful critique
                logger.debug(
                    f"ConstitutionalCritic: Successfully critiqued text in {processing_time:.2f}ms, "
                    f"found {len(critique_data['violations'])} violations"
                )

                # Explicitly create a Dict[str, Any] to return
                critique_result: Dict[str, Any] = critique_data
                return critique_result

        except json.JSONDecodeError as e:
            # Log the error
            log_error(
                e,
                logger,
                component="ConstitutionalCritic",
                operation="critique_json_parse",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Failed to parse JSON, create a default response
            logger.warning(
                f"ConstitutionalCritic: Failed to parse JSON in critique response: {str(e)}"
            )
            # Create a Dict[str, Any] to return
            json_error_critique: Dict[str, Any] = {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "violations": ["Unable to identify specific violations"],
                "suggestions": ["General improvement"],
                "issues": ["Unable to identify specific violations"],
                "processing_time_ms": processing_time,
                "error": str(e),
            }
            return json_error_critique

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ConstitutionalCritic", operation="critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text: {str(e)}",
                component="ConstitutionalCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the principles are well-defined",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "principles_count": len(self.principles),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
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

        # Log improvement attempt
        logger.debug(
            f"ConstitutionalCritic: Improving text of length {len(text)} based on critique with "
            f"{len(critique.get('violations', []))} violations and {len(critique.get('suggestions', []))} suggestions"
        )

        # Format principles as a bulleted list
        principles_str = "\n".join(f"- {p}" for p in self.principles)

        # Format violations and suggestions
        violations = critique.get("violations", [])
        suggestions = critique.get("suggestions", [])

        violations_str = "\n".join(f"- {v}" for v in violations)
        suggestions_str = "\n".join(f"- {s}" for s in suggestions)

        prompt = f"""
        Please improve the following text to address the violations of principles:

        Principles:
        {principles_str}

        Text:
        ```
        {text}
        ```

        Violations:
        {violations_str}

        Suggestions:
        {suggestions_str}

        Improved text:
        """

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="ConstitutionalCritic",
                operation="improve",
                message_prefix="Failed to improve text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the critique information is valid",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "principles_count": len(self.principles),
                    "violations_count": len(violations),
                    "suggestions_count": len(suggestions),
                    "temperature": self.temperature,
                },
            ):
                # Generate improved text
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
                    f"ConstitutionalCritic: Successfully improved text in {processing_time:.2f}ms, "
                    f"original length: {len(text)}, improved length: {len(improved_text)}"
                )

                return improved_text

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ConstitutionalCritic", operation="improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                component="ConstitutionalCritic",
                operation="improve",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the critique information is valid",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "principles_count": len(self.principles),
                    "violations_count": len(violations),
                    "suggestions_count": len(suggestions),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )


@register_improver("constitutional")
def create_constitutional_critic(
    model: Model,
    principles: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> ConstitutionalCritic:
    """Create a constitutional critic.

    This factory function creates a ConstitutionalCritic based on the paper
    "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        principles: The principles to evaluate text against.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the ConstitutionalCritic.

    Returns:
        A ConstitutionalCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    start_time = time.time()

    try:
        # Log factory function call
        logger.debug(
            f"Creating ConstitutionalCritic with model {model.__class__.__name__ if model else None}, "
            f"principles_count={len(principles) if principles else 'default'}, "
            f"temperature={temperature}"
        )

        # Validate parameters
        if not model:
            logger.error("No model provided to create_constitutional_critic")
            raise ImproverError(
                message="Model must be provided",
                component="ConstitutionalCriticFactory",
                operation="create_critic",
                suggestions=[
                    "Provide a valid model instance",
                    "Check that the model implements the Model protocol",
                ],
                metadata={
                    "principles_count": len(principles) if principles else 0,
                    "temperature": temperature,
                },
            )

        # Create the critic
        critic = ConstitutionalCritic(
            model=model,
            principles=principles,
            system_prompt=system_prompt,
            temperature=temperature,
            **options,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created ConstitutionalCritic with {len(critic.principles)} principles "
            f"in {processing_time:.2f}ms"
        )

        return critic

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(
            e,
            logger,
            component="ConstitutionalCriticFactory",
            operation="create_critic",
        )

        # Raise as ImproverError with more context if not already an ImproverError
        if not isinstance(e, ImproverError):
            raise ImproverError(
                message=f"Failed to create ConstitutionalCritic: {str(e)}",
                component="ConstitutionalCriticFactory",
                operation="create_critic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the principles are well-defined",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "principles_count": len(principles) if principles else 0,
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
