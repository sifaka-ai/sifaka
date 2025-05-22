"""Reflexion critic for Sifaka.

This module provides a critic that uses self-reflection to improve text quality.
The ReflexionCritic implements the Reflexion approach for improving text through
self-reflection and iterative refinement.

References:
    Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023).
    Reflexion: Language Agents with Verbal Reinforcement Learning.
    arXiv preprint arXiv:2303.11366.
    https://arxiv.org/abs/2303.11366

Example:
    ```python
    from sifaka.critics.reflexion import create_reflexion_critic
    from sifaka.models.openai import OpenAIModel

    # Create a model
    model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a reflexion critic
    critic = create_reflexion_critic(
        model=model,
        reflection_rounds=2
    )

    # Improve text
    improved_text, result = critic.improve("Text to improve")
    ```
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


class ReflexionCritic(Critic):
    """Critic that uses self-reflection to improve text quality.

    This critic implements the Reflexion approach for improving text through
    self-reflection. It performs a single improvement step without internal iterations.

    The process involves:
    1. Generating an initial critique of the text
    2. Improving the text based on the critique

    Attributes:
        model (Model): The model used for critiquing and improving text.
        system_prompt (str): The system prompt used for the model.
        temperature (float): The temperature used for the model.

    Example:
        ```python
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.models.openai import OpenAIModel

        # Create a model
        model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

        # Create a reflexion critic
        critic = ReflexionCritic(
            model=model,
            temperature=0.7
        )

        # Improve text
        improved_text, result = critic.improve("Text to improve")
        print(f"Original: {result.original_text}")
        print(f"Improved: {result.improved_text}")
        ```
    """

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the reflexion critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        # Log initialization attempt
        logger.debug(
            f"Initializing ReflexionCritic with model={model.__class__.__name__}, "
            f"temperature={temperature}"
        )

        try:
            # Validate parameters
            if not model:
                logger.error("No model provided to ReflexionCritic")
                raise ImproverError(
                    message="Model must be provided",
                    component="ReflexionCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid model instance",
                        "Check that the model implements the Model protocol",
                    ],
                    metadata={
                        "temperature": temperature,
                    },
                )

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in self-reflection and improvement. "
                    "Your goal is to reflect on text and improve it."
                )
                logger.debug("Using default system prompt for ReflexionCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="ReflexionCritic",
                operation="initialization",
                message_prefix="Failed to initialize ReflexionCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "temperature": temperature,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Log successful initialization
            logger.debug(
                f"Successfully initialized ReflexionCritic with model={model.__class__.__name__}, "
                f"temperature={temperature}"
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ReflexionCritic", operation="initialization")

            # Re-raise as ImproverError with more context if not already an ImproverError
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize ReflexionCritic: {str(e)}",
                    component="ReflexionCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the reflection_rounds parameter is a positive integer",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "temperature": temperature,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text through self-reflection.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        start_time = time.time()

        # Log critique attempt
        logger.debug(f"ReflexionCritic: Critiquing text of length {len(text)}")

        prompt = f"""
        Please reflect on the following text and identify areas for improvement:

        ```
        {text}
        ```

        Provide your reflection in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text needs improvement
        - "message": a brief summary of your reflection
        - "issues": a list of specific issues identified
        - "suggestions": a list of suggestions for improvement
        - "reflections": your detailed thoughts on the text

        JSON response:
        """

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="ReflexionCritic",
                operation="reflection_generation",
                message_prefix="Failed to generate reflection",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                ],
                metadata={"text_length": len(text), "temperature": self.temperature},
            ):
                # Generate reflection
                response = self._generate(prompt)
                logger.debug(f"ReflexionCritic: Generated response of length {len(response)}")

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        "ReflexionCritic: No JSON found in response, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    return {
                        "needs_improvement": True,
                        "message": "Unable to parse critique response, but proceeding with improvement",
                        "issues": ["Unable to identify specific issues"],
                        "suggestions": ["General improvement"],
                        "reflections": ["Unable to generate reflections"],
                        "processing_time_ms": processing_time,
                    }

                # Parse JSON
                with critic_context(
                    critic_name="ReflexionCritic",
                    operation="json_parsing",
                    message_prefix="Failed to parse JSON response",
                    suggestions=[
                        "Check if the model is generating valid JSON",
                        "Try adjusting the temperature to get more consistent output",
                    ],
                    metadata={
                        "response_length": len(response),
                        "json_length": json_end - json_start,
                        "temperature": self.temperature,
                    },
                ):
                    json_str = response[json_start:json_end]
                    critique = json.loads(json_str)
                    logger.debug("ReflexionCritic: Successfully parsed JSON response")

                # Ensure all required fields are present
                critique.setdefault("needs_improvement", True)
                critique.setdefault("message", "Text needs improvement")
                critique.setdefault("issues", [])
                critique.setdefault("suggestions", [])
                critique.setdefault("reflections", [])

                # Log critique details
                logger.debug(
                    f"ReflexionCritic: Generated critique with {len(critique['issues'])} issues, "
                    f"{len(critique['suggestions'])} suggestions, "
                    f"and {len(critique['reflections']) if isinstance(critique['reflections'], list) else 1} reflections"
                )

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                critique["processing_time_ms"] = processing_time

                # Log successful critique
                logger.debug(
                    f"ReflexionCritic: Successfully critiqued text in {processing_time:.2f}ms"
                )

                # Explicitly create a Dict[str, Any] to return
                critique_result: Dict[str, Any] = critique
                return critique_result

        except json.JSONDecodeError as e:
            # Log the error
            log_error(e, logger, component="ReflexionCritic", operation="json_parsing")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Failed to parse JSON, create a default response
            logger.warning(f"ReflexionCritic: Failed to parse JSON in response: {str(e)}")
            # Create a Dict[str, Any] to return
            json_error_critique: Dict[str, Any] = {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
                "reflections": ["Unable to generate reflections"],
                "processing_time_ms": processing_time,
                "error": str(e),
            }
            return json_error_critique

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ReflexionCritic", operation="critique")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            logger.error(f"ReflexionCritic: Error critiquing text: {str(e)}")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text: {str(e)}",
                component="ReflexionCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Check the error message for details",
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
        """Improve text using self-reflection.

        This method performs a single improvement step without internal iterations.

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
        logger.debug(f"ReflexionCritic: Improving text of length {len(text)}")

        # Get reflections from critique
        reflections = []
        if isinstance(critique.get("reflections"), list):
            reflections.extend(critique.get("reflections", []))
            logger.debug(
                f"ReflexionCritic: Using {len(critique.get('reflections', []))} reflections from critique"
            )
        elif isinstance(critique.get("reflections"), str):
            reflections.append(critique.get("reflections"))
            logger.debug("ReflexionCritic: Using 1 reflection from critique")

        # Format issues and suggestions
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])

        issues_str = "\n".join(f"- {i}" for i in issues)
        suggestions_str = "\n".join(f"- {s}" for s in suggestions)
        reflections_str = self._format_list(reflections)

        logger.debug(
            f"ReflexionCritic: Improving text with {len(issues)} issues and {len(suggestions)} suggestions"
        )

        try:
            # Perform improvement
            with critic_context(
                critic_name="ReflexionCritic",
                operation="improvement",
                message_prefix="Failed to improve text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                ],
                metadata={
                    "text_length": len(text),
                    "issues_count": len(issues),
                    "suggestions_count": len(suggestions),
                    "reflections_count": len(reflections),
                    "temperature": self.temperature,
                },
            ):
                # Create improvement prompt
                prompt = f"""
                Please improve the following text based on the issues, suggestions, and reflections:

                Text:
                ```
                {text}
                ```

                Issues:
                {issues_str}

                Suggestions:
                {suggestions_str}

                Reflections:
                {reflections_str}

                Improved text:
                """

                # Generate improved text
                response = self._generate(prompt)
                logger.debug(f"ReflexionCritic: Generated improvement of length {len(response)}")

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()
                    logger.debug(
                        "ReflexionCritic: Removed markdown code block markers from response"
                    )

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful improvement
                logger.debug(
                    f"ReflexionCritic: Successfully improved text in {processing_time:.2f}ms, "
                    f"original length: {len(text)}, improved length: {len(improved_text)}"
                )

                return improved_text

        except Exception as e:
            # Log the error
            log_error(e, logger, component="ReflexionCritic", operation="improvement")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            logger.error(f"ReflexionCritic: Error improving text: {str(e)}")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                component="ReflexionCritic",
                operation="improvement",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Check the error message for details",
                ],
                metadata={
                    "text_length": len(text),
                    "temperature": self.temperature,
                    "issues_count": len(issues),
                    "suggestions_count": len(suggestions),
                    "reflections_count": len(reflections),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a numbered list.

        Args:
            items: The list of items to format.

        Returns:
            A string with the items formatted as a numbered list.
        """
        if not items:
            return "None"

        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))


@register_improver("reflexion")
def create_reflexion_critic(
    model: Model,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> ReflexionCritic:
    """Create a reflexion critic for improving text through self-reflection.

    This factory function creates a ReflexionCritic that implements the Reflexion approach
    for improving text through self-reflection. The critic analyzes text, identifies areas
    for improvement, and provides feedback to improve the text.

    The function is registered with the registry system for dependency injection,
    allowing it to be used with the Chain class.

    Args:
        model (Model): The model to use for critiquing and improving text.
        system_prompt (Optional[str]): The system prompt to use for the model.
        temperature (float): The temperature to use for the model.
        **options (Any): Additional options to pass to the ReflexionCritic.

    Returns:
        ReflexionCritic: A configured ReflexionCritic instance.

    Raises:
        ImproverError: If the model is not provided or if initialization fails.

    Example:
        ```python
        from sifaka.critics.reflexion import create_reflexion_critic
        from sifaka.models.openai import OpenAIModel

        # Create a model
        model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

        # Create a reflexion critic
        critic = create_reflexion_critic(
            model=model,
            temperature=0.7
        )

        # Improve text
        improved_text, result = critic.improve("Text to improve")
        ```

    References:
        Shinn, N., Cassano, F., Labash, B., Gopinath, A., Narasimhan, K., & Yao, S. (2023).
        Reflexion: Language Agents with Verbal Reinforcement Learning.
        arXiv preprint arXiv:2303.11366.
        https://arxiv.org/abs/2303.11366
    """
    start_time = time.time()

    # Log factory function call
    logger.debug(
        f"Creating ReflexionCritic with model={model.__class__.__name__ if model else None}, "
        f"temperature={temperature}"
    )

    try:
        # Validate parameters
        if not model:
            logger.error("No model provided to create_reflexion_critic")
            raise ImproverError(
                message="Model must be provided",
                component="ReflexionCriticFactory",
                operation="creation",
                suggestions=[
                    "Provide a valid model instance",
                    "Check that the model implements the Model protocol",
                ],
                metadata={
                    "temperature": temperature,
                },
            )

        # Create the critic
        critic = ReflexionCritic(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            **options,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(f"Successfully created ReflexionCritic in {processing_time:.2f}ms")

        return critic

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="ReflexionCriticFactory", operation="creation")

        # Raise as ImproverError with more context if not already an ImproverError
        if not isinstance(e, ImproverError):
            raise ImproverError(
                message=f"Failed to create ReflexionCritic: {str(e)}",
                component="ReflexionCriticFactory",
                operation="creation",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                    "Check the error message for details",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
