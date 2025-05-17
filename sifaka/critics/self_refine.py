"""
Self-Refine critic for Sifaka.

This module provides a critic that uses the Self-Refine technique.

Based on the paper:
"Self-Refine: Iterative Refinement with Self-Feedback"
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe,
Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder,
Shashank Gupta, Amir Yazdanbakhsh, Peter Clark
arXiv:2303.17651 [cs.CL]
https://arxiv.org/abs/2303.17651
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError
from sifaka.registry import register_improver
from sifaka.utils.error_handling import critic_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class SelfRefineCritic(Critic):
    """Critic that uses the Self-Refine technique.

    This critic implements the Self-Refine technique, which uses a multi-step
    process of feedback and refinement to iteratively improve text.

    Attributes:
        model: The model to use for critiquing and improving text.
        refinement_rounds: The number of refinement rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
    """

    def __init__(
        self,
        model: Model,
        refinement_rounds: int = 2,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **options: Any,
    ):
        """Initialize the Self-Refine critic.

        Args:
            model: The model to use for critiquing and improving text.
            refinement_rounds: The number of refinement rounds to perform.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        start_time = time.time()

        # Log initialization attempt
        logger.debug(
            f"Initializing SelfRefineCritic with model={model.__class__.__name__ if model else None}, "
            f"refinement_rounds={refinement_rounds}, temperature={temperature}"
        )

        try:
            # Validate parameters
            if not model:
                logger.error("No model provided to SelfRefineCritic")
                raise ImproverError(
                    message="Model must be provided",
                    component="SelfRefineCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid model instance",
                        "Check that the model implements the Model protocol",
                    ],
                    metadata={"refinement_rounds": refinement_rounds, "temperature": temperature},
                )

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in iterative refinement. "
                    "Your goal is to provide detailed feedback and iteratively improve text."
                )
                logger.debug("Using default system prompt for SelfRefineCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="SelfRefineCritic",
                operation="initialization",
                message_prefix="Failed to initialize SelfRefineCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "refinement_rounds": refinement_rounds,
                    "temperature": temperature,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Validate and set refinement rounds
            original_refinement_rounds = refinement_rounds
            self.refinement_rounds = max(1, refinement_rounds)
            if (
                self.refinement_rounds != original_refinement_rounds
                and original_refinement_rounds < 1
            ):
                logger.warning(
                    f"Adjusted refinement_rounds from {original_refinement_rounds} to "
                    f"{self.refinement_rounds} (minimum: 1)"
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful initialization
            logger.debug(
                f"Successfully initialized SelfRefineCritic with {self.refinement_rounds} refinement rounds "
                f"in {processing_time:.2f}ms"
            )

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="SelfRefineCritic", operation="initialization")

            # Re-raise as ImproverError with more context if not already an ImproverError
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize SelfRefineCritic: {str(e)}",
                    component="SelfRefineCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the parameters are valid",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "refinement_rounds": refinement_rounds,
                        "temperature": temperature,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "processing_time_ms": processing_time,
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the Self-Refine technique.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        start_time = time.time()

        logger.debug(f"SelfRefineCritic: Critiquing text of length {len(text)}")

        prompt = f"""
        Please evaluate the following text and provide detailed feedback for improvement:

        ```
        {text}
        ```

        Provide your evaluation in JSON format with the following fields:
        - "needs_improvement": boolean indicating whether the text needs improvement
        - "message": a brief summary of your evaluation
        - "issues": a list of specific issues identified
        - "suggestions": a list of suggestions for improvement
        - "evaluation_criteria": a list of criteria you used to evaluate the text

        JSON response:
        """

        try:
            # Generate critique using the model
            with critic_context(
                critic_name="SelfRefineCritic",
                operation="critique",
                message_prefix="Failed to critique text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={"text_length": len(text), "temperature": self.temperature},
            ):
                response = self._generate(prompt)
                logger.debug(f"SelfRefineCritic: Generated response of length {len(response)}")

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        f"SelfRefineCritic: No JSON found in response, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    return {
                        "needs_improvement": True,
                        "message": "Unable to parse critique response, but proceeding with improvement",
                        "issues": ["Unable to identify specific issues"],
                        "suggestions": ["General improvement"],
                        "evaluation_criteria": ["Clarity", "Coherence", "Correctness"],
                        "refinement_history": [],
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
                critique.setdefault("evaluation_criteria", ["Clarity", "Coherence", "Correctness"])
                critique["refinement_history"] = []

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                critique["processing_time_ms"] = processing_time

                # Log successful critique
                logger.debug(
                    f"SelfRefineCritic: Successfully critiqued text in {processing_time:.2f}ms, "
                    f"needs_improvement={critique.get('needs_improvement', True)}, "
                    f"issues_count={len(critique.get('issues', []))}, "
                    f"suggestions_count={len(critique.get('suggestions', []))}"
                )

                return critique

        except json.JSONDecodeError as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="SelfRefineCritic", operation="parse_json")

            # Failed to parse JSON, create a default response
            logger.warning(f"SelfRefineCritic: Failed to parse JSON in response: {str(e)}")
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "issues": ["Unable to identify specific issues"],
                "suggestions": ["General improvement"],
                "evaluation_criteria": ["Clarity", "Coherence", "Correctness"],
                "refinement_history": [],
                "processing_time_ms": processing_time,
                "error": str(e),
            }

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="SelfRefineCritic", operation="critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text: {str(e)}",
                component="SelfRefineCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
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
        """Improve text using the Self-Refine technique.

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
            f"SelfRefineCritic: Improving text of length {len(text)} with max {self.refinement_rounds} rounds"
        )

        current_text = text
        refinement_history = critique.get("refinement_history", [])

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="SelfRefineCritic",
                operation="improve",
                message_prefix="Failed to improve text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "refinement_rounds": self.refinement_rounds,
                    "temperature": self.temperature,
                },
            ):
                # Format issues and suggestions
                issues = critique.get("issues", [])
                suggestions = critique.get("suggestions", [])

                issues_str = "\n".join(f"- {i}" for i in issues)
                suggestions_str = "\n".join(f"- {s}" for s in suggestions)

                # Perform initial improvement
                initial_prompt = f"""
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

                logger.debug(f"SelfRefineCritic: Generating initial improvement")
                response = self._generate(initial_prompt)
                logger.debug(
                    f"SelfRefineCritic: Generated initial improvement of length {len(response)}"
                )

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()
                    logger.debug("SelfRefineCritic: Removed code block markers from response")

                current_text = improved_text

                # Add initial improvement to refinement history
                refinement_history.append(
                    {
                        "round": 1,
                        "text": current_text,
                        "feedback": {
                            "issues": issues,
                            "suggestions": suggestions,
                        },
                    }
                )

                logger.debug(
                    f"SelfRefineCritic: Completed initial improvement, new length: {len(current_text)}"
                )

                # Perform additional refinement rounds
                for round_num in range(2, self.refinement_rounds + 1):
                    round_start_time = time.time()

                    logger.debug(
                        f"SelfRefineCritic: Starting refinement round {round_num}/{self.refinement_rounds}"
                    )

                    try:
                        # Generate feedback on current text
                        feedback_prompt = f"""
                        Please evaluate the following text and provide detailed feedback for further improvement:

                        ```
                        {current_text}
                        ```

                        Previous feedback:
                        {self._format_feedback_history(refinement_history)}

                        Provide your evaluation in JSON format with the following fields:
                        - "issues": a list of specific issues that still need to be addressed
                        - "suggestions": a list of suggestions for further improvement

                        JSON response:
                        """

                        logger.debug(f"SelfRefineCritic: Generating feedback for round {round_num}")
                        feedback_response = self._generate(feedback_prompt)
                        logger.debug(
                            f"SelfRefineCritic: Generated feedback of length {len(feedback_response)}"
                        )

                        # Extract JSON from response
                        json_start = feedback_response.find("{")
                        json_end = feedback_response.rfind("}") + 1

                        if json_start == -1 or json_end == 0:
                            # No JSON found, create a default response
                            logger.warning(
                                f"SelfRefineCritic: No JSON found in feedback response for round {round_num}, using default"
                            )
                            feedback = {
                                "issues": ["Further refinement needed"],
                                "suggestions": ["Continue improving the text"],
                            }
                        else:
                            try:
                                json_str = feedback_response[json_start:json_end]
                                feedback = json.loads(json_str)

                                # Ensure all required fields are present
                                feedback.setdefault("issues", [])
                                feedback.setdefault("suggestions", [])

                                logger.debug(
                                    f"SelfRefineCritic: Parsed feedback with {len(feedback.get('issues', []))} issues and "
                                    f"{len(feedback.get('suggestions', []))} suggestions"
                                )
                            except json.JSONDecodeError as json_error:
                                logger.warning(
                                    f"SelfRefineCritic: Failed to parse JSON in feedback response: {str(json_error)}"
                                )
                                feedback = {
                                    "issues": ["Further refinement needed"],
                                    "suggestions": ["Continue improving the text"],
                                    "error": [str(json_error)],
                                }

                        # Format issues and suggestions
                        issues_str = "\n".join(f"- {i}" for i in feedback.get("issues", []))
                        suggestions_str = "\n".join(
                            f"- {s}" for s in feedback.get("suggestions", [])
                        )

                        # Improve text based on feedback
                        refinement_prompt = f"""
                        Please further improve the following text based on the issues and suggestions:

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

                        logger.debug(
                            f"SelfRefineCritic: Generating refinement for round {round_num}"
                        )
                        refinement_response = self._generate(refinement_prompt)
                        logger.debug(
                            f"SelfRefineCritic: Generated refinement of length {len(refinement_response)}"
                        )

                        # Extract improved text from response
                        improved_text = refinement_response.strip()

                        # Remove any markdown code block markers
                        if improved_text.startswith("```") and improved_text.endswith("```"):
                            improved_text = improved_text[3:-3].strip()
                            logger.debug(
                                "SelfRefineCritic: Removed code block markers from refinement response"
                            )

                        current_text = improved_text

                        # Add refinement to history
                        refinement_history.append(
                            {
                                "round": round_num,
                                "text": current_text,
                                "feedback": feedback,
                            }
                        )

                        # Calculate round time
                        round_time = (
                            time.time() - round_start_time
                        ) * 1000  # Convert to milliseconds

                        logger.debug(
                            f"SelfRefineCritic: Completed refinement round {round_num} in {round_time:.2f}ms, "
                            f"new length: {len(current_text)}"
                        )

                    except Exception as round_error:
                        # Log the error but continue with the current text
                        log_error(
                            round_error,
                            logger,
                            component="SelfRefineCritic",
                            operation=f"refinement_round_{round_num}",
                        )
                        logger.warning(
                            f"SelfRefineCritic: Error in refinement round {round_num}, stopping refinement: {str(round_error)}"
                        )
                        break

                # Calculate total processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful improvement
                logger.debug(
                    f"SelfRefineCritic: Successfully improved text in {processing_time:.2f}ms, "
                    f"original length: {len(text)}, final length: {len(current_text)}, "
                    f"completed rounds: {len(refinement_history)}"
                )

                return current_text

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="SelfRefineCritic", operation="improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                component="SelfRefineCritic",
                operation="improve",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "refinement_rounds": self.refinement_rounds,
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                    "completed_rounds": len(refinement_history),
                },
            )

    def _format_feedback_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the feedback history as a string.

        Args:
            history: The feedback history.

        Returns:
            A string representation of the feedback history.
        """
        start_time = time.time()

        logger.debug(f"SelfRefineCritic: Formatting feedback history with {len(history)} entries")

        try:
            if not history:
                logger.debug("SelfRefineCritic: No feedback history to format")
                return "No previous feedback"

            result = []

            for entry in history:
                round_num = entry.get("round", 0)
                feedback = entry.get("feedback", {})

                issues = feedback.get("issues", [])
                suggestions = feedback.get("suggestions", [])

                result.append(f"Round {round_num} feedback:")

                if issues:
                    result.append("Issues:")
                    for issue in issues:
                        result.append(f"- {issue}")

                if suggestions:
                    result.append("Suggestions:")
                    for suggestion in suggestions:
                        result.append(f"- {suggestion}")

                result.append("")

            formatted_result = "\n".join(result)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful formatting
            logger.debug(
                f"SelfRefineCritic: Successfully formatted feedback history in {processing_time:.2f}ms, "
                f"result length: {len(formatted_result)}"
            )

            return formatted_result

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="SelfRefineCritic", operation="format_feedback_history")

            # Return a default string in case of error
            logger.warning(
                f"SelfRefineCritic: Error formatting feedback history: {str(e)}, using default"
            )
            return "Previous feedback available but could not be formatted."


@register_improver("self_refine")
def create_self_refine_critic(
    model: Model,
    refinement_rounds: int = 2,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **options: Any,
) -> SelfRefineCritic:
    """Create a Self-Refine critic.

    This factory function creates a SelfRefineCritic based on the paper
    "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        refinement_rounds: The number of refinement rounds to perform.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        **options: Additional options to pass to the SelfRefineCritic.

    Returns:
        A SelfRefineCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    start_time = time.time()

    logger.debug(
        f"Creating SelfRefineCritic with model={model.__class__.__name__ if model else None}, "
        f"refinement_rounds={refinement_rounds}, temperature={temperature}"
    )

    try:
        # Create the critic with error handling
        with critic_context(
            critic_name="SelfRefineCritic",
            operation="creation",
            message_prefix="Failed to create SelfRefineCritic",
            suggestions=[
                "Check if the model is properly configured",
                "Verify that the parameters are valid",
            ],
            metadata={
                "model_type": model.__class__.__name__ if model else None,
                "refinement_rounds": refinement_rounds,
                "temperature": temperature,
            },
        ):
            critic = SelfRefineCritic(
                model=model,
                refinement_rounds=refinement_rounds,
                system_prompt=system_prompt,
                temperature=temperature,
                **options,
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful creation
            logger.debug(f"Successfully created SelfRefineCritic in {processing_time:.2f}ms")

            return critic

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="SelfRefineCritic", operation="creation")

        # Re-raise as ImproverError with more context if not already an ImproverError
        if not isinstance(e, ImproverError):
            raise ImproverError(
                message=f"Failed to create SelfRefineCritic: {str(e)}",
                component="SelfRefineCritic",
                operation="creation",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                    "Check the error message for details",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "refinement_rounds": refinement_rounds,
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
