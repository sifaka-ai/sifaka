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
from typing import Dict, Any, Optional, List

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver
from sifaka.utils.error_handling import critic_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class ReflexionCritic(Critic):
    """Critic that uses self-reflection to improve text quality.

    This critic implements the Reflexion approach for improving text through
    self-reflection and iterative refinement. It uses a multi-step process where
    the model reflects on its own output and iteratively improves it based on
    those reflections.

    The process involves:
    1. Generating an initial critique of the text
    2. Improving the text based on the critique
    3. Performing additional reflection rounds to further refine the text

    Attributes:
        model (Model): The model used for critiquing and improving text.
        reflection_rounds (int): The number of reflection rounds to perform.
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
            reflection_rounds=2,
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
        reflection_rounds: int = 1,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the reflexion critic.

        Args:
            model: The model to use for critiquing and improving text.
            reflection_rounds: The number of reflection rounds to perform.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        # Log initialization attempt
        logger.debug(
            f"Initializing ReflexionCritic with model={model.__class__.__name__}, "
            f"reflection_rounds={reflection_rounds}, temperature={temperature}"
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
                    metadata={"reflection_rounds": reflection_rounds, "temperature": temperature},
                )

            if reflection_rounds < 1:
                logger.warning(
                    f"Invalid reflection_rounds value: {reflection_rounds}, using default value of 1"
                )
                reflection_rounds = 1

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in self-reflection and improvement. "
                    "Your goal is to reflect on text and iteratively improve it."
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
                    "reflection_rounds": reflection_rounds,
                    "temperature": temperature,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Store configuration
            self.reflection_rounds = max(1, reflection_rounds)

            # Log successful initialization
            logger.debug(
                f"Successfully initialized ReflexionCritic with model={model.__class__.__name__}, "
                f"reflection_rounds={self.reflection_rounds}, temperature={temperature}"
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
                        "reflection_rounds": reflection_rounds,
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
        logger.debug(
            f"ReflexionCritic: Critiquing text of length {len(text)}, "
            f"reflection_rounds={self.reflection_rounds}"
        )

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
                    logger.debug(f"ReflexionCritic: Successfully parsed JSON response")

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
                    "reflection_rounds": self.reflection_rounds,
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text through multiple rounds of reflection.

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
            f"ReflexionCritic: Improving text of length {len(text)}, "
            f"reflection_rounds={self.reflection_rounds}"
        )

        current_text = text
        reflections = []

        # Add initial reflection from critique
        if isinstance(critique.get("reflections"), list):
            reflections.extend(critique.get("reflections", []))
            logger.debug(
                f"ReflexionCritic: Added {len(critique.get('reflections', []))} reflections from critique"
            )
        elif isinstance(critique.get("reflections"), str):
            reflections.append(critique.get("reflections"))
            logger.debug("ReflexionCritic: Added 1 reflection from critique")

        # Format issues and suggestions
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])

        issues_str = "\n".join(f"- {i}" for i in issues)
        suggestions_str = "\n".join(f"- {s}" for s in suggestions)

        logger.debug(
            f"ReflexionCritic: Improving text with {len(issues)} issues and {len(suggestions)} suggestions"
        )

        try:
            # Perform initial improvement
            with critic_context(
                critic_name="ReflexionCritic",
                operation="initial_improvement",
                message_prefix="Failed to perform initial improvement",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                ],
                metadata={
                    "text_length": len(text),
                    "issues_count": len(issues),
                    "suggestions_count": len(suggestions),
                    "temperature": self.temperature,
                },
            ):
                # Create improvement prompt
                prompt = f"""
                Please improve the following text based on the issues and suggestions:

                Text:
                ```
                {current_text}
                ```

                Issues:
                {issues_str}

                Suggestions:
                {suggestions_str}

                Improved text:
                """

                # Generate improved text
                response = self._generate(prompt)
                logger.debug(
                    f"ReflexionCritic: Generated initial improvement of length {len(response)}"
                )

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()
                    logger.debug(
                        "ReflexionCritic: Removed markdown code block markers from response"
                    )

                current_text = improved_text
                logger.debug(
                    f"ReflexionCritic: Completed initial improvement, new text length: {len(current_text)}"
                )

            # Perform additional reflection rounds
            for round_num in range(1, self.reflection_rounds):
                logger.debug(f"ReflexionCritic: Starting reflection round {round_num}")

                # Generate reflection on current text
                with critic_context(
                    critic_name="ReflexionCritic",
                    operation=f"reflection_round_{round_num}",
                    message_prefix=f"Failed to generate reflection in round {round_num}",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the text is not too long for the model",
                    ],
                    metadata={
                        "text_length": len(current_text),
                        "round": round_num,
                        "reflections_count": len(reflections),
                        "temperature": self.temperature,
                    },
                ):
                    reflection_prompt = f"""
                    Please reflect on the following text and identify areas for further improvement:

                    ```
                    {current_text}
                    ```

                    Previous reflections:
                    {self._format_list(reflections)}

                    Provide your reflection:
                    """

                    reflection = self._generate(reflection_prompt)
                    reflections.append(reflection)
                    logger.debug(
                        f"ReflexionCritic: Generated reflection of length {len(reflection)} in round {round_num}"
                    )

                # Improve text based on reflection
                with critic_context(
                    critic_name="ReflexionCritic",
                    operation=f"improvement_round_{round_num}",
                    message_prefix=f"Failed to improve text in round {round_num}",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the text is not too long for the model",
                    ],
                    metadata={
                        "text_length": len(current_text),
                        "reflection_length": len(reflection),
                        "round": round_num,
                        "temperature": self.temperature,
                    },
                ):
                    improvement_prompt = f"""
                    Please improve the following text based on the reflection:

                    Text:
                    ```
                    {current_text}
                    ```

                    Reflection:
                    {reflection}

                    Improved text:
                    """

                    response = self._generate(improvement_prompt)
                    logger.debug(
                        f"ReflexionCritic: Generated improvement of length {len(response)} in round {round_num}"
                    )

                    # Extract improved text from response
                    improved_text = response.strip()

                    # Remove any markdown code block markers
                    if improved_text.startswith("```") and improved_text.endswith("```"):
                        improved_text = improved_text[3:-3].strip()
                        logger.debug(
                            "ReflexionCritic: Removed markdown code block markers from response"
                        )

                    current_text = improved_text
                    logger.debug(
                        f"ReflexionCritic: Completed improvement round {round_num}, new text length: {len(current_text)}"
                    )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful improvement
            logger.debug(
                f"ReflexionCritic: Successfully improved text in {processing_time:.2f}ms, "
                f"original length: {len(text)}, improved length: {len(current_text)}, "
                f"rounds: {self.reflection_rounds}"
            )

            return current_text

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
                    "Check if the reflection rounds parameter is valid",
                    "Check the error message for details",
                ],
                metadata={
                    "text_length": len(text),
                    "reflection_rounds": self.reflection_rounds,
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
    reflection_rounds: int = 1,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> ReflexionCritic:
    """Create a reflexion critic for improving text through self-reflection.

    This factory function creates a ReflexionCritic that implements the Reflexion approach
    for improving text through self-reflection and iterative refinement. The critic
    analyzes text, identifies areas for improvement, and iteratively refines the text
    through multiple rounds of reflection.

    The function is registered with the registry system for dependency injection,
    allowing it to be used with the Chain class.

    Args:
        model (Model): The model to use for critiquing and improving text.
        reflection_rounds (int): The number of reflection rounds to perform.
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
            reflection_rounds=2,
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
        f"reflection_rounds={reflection_rounds}, temperature={temperature}"
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
                metadata={"reflection_rounds": reflection_rounds, "temperature": temperature},
            )

        if reflection_rounds < 1:
            logger.warning(
                f"Invalid reflection_rounds value: {reflection_rounds}, using default value of 1"
            )
            reflection_rounds = 1

        # Create the critic
        critic = ReflexionCritic(
            model=model,
            reflection_rounds=reflection_rounds,
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
                    "Verify that the reflection_rounds parameter is a positive integer",
                    "Check the error message for details",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "reflection_rounds": reflection_rounds,
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
