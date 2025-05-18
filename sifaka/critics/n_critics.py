"""
N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics.

This module provides a critic that uses an ensemble of critics for self-refinement.

Based on the paper:
"N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics"
Shahriar Mousavi, Roxana Leontie Rios Gutierrez, Deepak Rengarajan, Vishal Gundecha,
Anand Raju Babu, Avisek Naug, Srinivas Chappidi
arXiv:2310.18679 [cs.CL]
https://arxiv.org/abs/2310.18679
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


class NCriticsCritic(Critic):
    """Critic that uses an ensemble of critics for self-refinement.

    This critic implements the N-Critics technique from the paper
    "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).

    N-Critics leverages an ensemble of specialized critics, each focusing on different aspects
    of the text, to provide comprehensive feedback and guide the refinement process.

    Attributes:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        num_critics: Number of specialized critics to use.
        max_refinement_iterations: Maximum number of refinement iterations.
    """

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        num_critics: int = 3,
        max_refinement_iterations: int = 2,
        **options: Any,
    ):
        """Initialize the N-Critics critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            num_critics: Number of specialized critics to use.
            max_refinement_iterations: Maximum number of refinement iterations.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        start_time = time.time()

        # Log initialization attempt
        logger.debug(
            f"Initializing NCriticsCritic with model={model.__class__.__name__ if model else None}, "
            f"num_critics={num_critics}, max_refinement_iterations={max_refinement_iterations}, "
            f"temperature={temperature}"
        )

        try:
            # Validate parameters
            if not model:
                logger.error("No model provided to NCriticsCritic")
                raise ImproverError(
                    message="Model must be provided",
                    component="NCriticsCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a valid model instance",
                        "Check that the model implements the Model protocol",
                    ],
                    metadata={
                        "num_critics": num_critics,
                        "max_refinement_iterations": max_refinement_iterations,
                        "temperature": temperature,
                    },
                )

            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert language model that uses an ensemble of specialized critics "
                    "to provide comprehensive feedback and guide the refinement process. "
                    "You follow the N-Critics approach to provide structured guidance."
                )
                logger.debug("Using default system prompt for NCriticsCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="NCriticsCritic",
                operation="initialization",
                message_prefix="Failed to initialize NCriticsCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "num_critics": num_critics,
                    "max_refinement_iterations": max_refinement_iterations,
                    "temperature": temperature,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Validate and clamp parameters
            original_num_critics = num_critics
            self.num_critics = max(1, min(5, num_critics))  # Clamp between 1 and 5
            if self.num_critics != original_num_critics:
                logger.warning(
                    f"Adjusted num_critics from {original_num_critics} to {self.num_critics} (valid range: 1-5)"
                )

            original_max_iterations = max_refinement_iterations
            self.max_refinement_iterations = max(1, max_refinement_iterations)
            if self.max_refinement_iterations != original_max_iterations:
                logger.warning(
                    f"Adjusted max_refinement_iterations from {original_max_iterations} to "
                    f"{self.max_refinement_iterations} (minimum: 1)"
                )

            # Define the critic roles
            self.critic_roles = [
                "Factual Accuracy Critic: Focus on identifying factual errors and inaccuracies.",
                "Coherence and Clarity Critic: Focus on improving the logical flow and clarity of the text.",
                "Completeness Critic: Focus on identifying missing information or incomplete explanations.",
                "Style and Tone Critic: Focus on improving the writing style, tone, and language usage.",
                "Relevance Critic: Focus on ensuring the content is relevant to the intended purpose.",
            ][
                : self.num_critics
            ]  # Use only the specified number of critics

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful initialization
            logger.debug(
                f"Successfully initialized NCriticsCritic with {len(self.critic_roles)} critics "
                f"in {processing_time:.2f}ms"
            )

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="initialization")

            # Re-raise as ImproverError with more context if not already an ImproverError
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize NCriticsCritic: {str(e)}",
                    component="NCriticsCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the parameters are valid",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "num_critics": num_critics,
                        "max_refinement_iterations": max_refinement_iterations,
                        "temperature": temperature,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "processing_time_ms": processing_time,
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the N-Critics technique.

        This method implements the ensemble critic approach from the N-Critics paper:
        1. Generate critiques from multiple specialized critics
        2. Aggregate the critiques into a comprehensive assessment

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        start_time = time.time()

        logger.debug(
            f"NCriticsCritic: Critiquing text of length {len(text)} with {len(self.critic_roles)} critics"
        )

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="NCriticsCritic",
                operation="critique",
                message_prefix="Failed to critique text with N-Critics",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "num_critics": len(self.critic_roles),
                    "temperature": self.temperature,
                },
            ):
                # Generate critiques from each specialized critic
                critic_critiques = []
                for i, role in enumerate(self.critic_roles):
                    logger.debug(
                        f"NCriticsCritic: Generating critique from critic {i+1}/{len(self.critic_roles)}: {role[:30]}..."
                    )
                    critique = self._generate_critic_critique(text, role)
                    critic_critiques.append(critique)
                    logger.debug(
                        f"NCriticsCritic: Critic {i+1} score: {critique.get('score', 'N/A')}"
                    )

                # Aggregate critiques
                logger.debug(f"NCriticsCritic: Aggregating {len(critic_critiques)} critiques")
                aggregated_critique = self._aggregate_critiques(critic_critiques)

                # Determine if improvement is needed
                needs_improvement = any(
                    critique.get("needs_improvement", True) for critique in critic_critiques
                )

                # Prepare the final critique
                final_critique = {
                    "needs_improvement": needs_improvement,
                    "message": aggregated_critique["summary"],
                    "critic_critiques": critic_critiques,
                    "aggregated_critique": aggregated_critique,
                    # For compatibility with base Critic
                    "issues": aggregated_critique["issues"],
                    "suggestions": aggregated_critique["suggestions"],
                    "processing_time_ms": (time.time() - start_time)
                    * 1000,  # Convert to milliseconds
                }

                # Log successful critique
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                logger.debug(
                    f"NCriticsCritic: Successfully critiqued text in {processing_time:.2f}ms, "
                    f"needs_improvement={needs_improvement}, "
                    f"issues_count={len(aggregated_critique['issues'])}, "
                    f"suggestions_count={len(aggregated_critique['suggestions'])}"
                )

                return final_critique

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text with N-Critics: {str(e)}",
                component="NCriticsCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "num_critics": len(self.critic_roles),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _generate_critic_critique(self, text: str, role: str) -> Dict[str, Any]:
        """Generate a critique from a specialized critic.

        Args:
            text: The text to critique.
            role: The role of the specialized critic.

        Returns:
            A dictionary with the critique from the specialized critic.

        Raises:
            ImproverError: If the critique cannot be generated.
        """
        start_time = time.time()

        logger.debug(f"NCriticsCritic: Generating critique for role: {role[:30]}...")

        prompt = f"""
        You are a specialized critic with the following role:
        {role}

        Your task is to critique the following text based on your specialized role:

        ```
        {text}
        ```

        Please provide a detailed critique that:
        1. Identifies specific issues related to your specialized role
        2. Explains why these issues are problematic
        3. Suggests concrete improvements
        4. Rates the text on a scale of 1-10 for your specific area of focus

        Format your response as JSON with the following fields:
        - "role": your specialized role
        - "needs_improvement": boolean indicating whether the text needs improvement in your area
        - "score": your rating of the text on a scale of 1-10
        - "issues": a list of specific issues you identified
        - "suggestions": a list of specific suggestions for improvement
        - "explanation": a brief explanation of your overall assessment

        JSON response:
        """

        try:
            # Generate critique using the model
            with critic_context(
                critic_name="NCriticsCritic",
                operation="generate_critique",
                message_prefix=f"Failed to generate critique for role: {role[:30]}",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "role": role[:50],  # Include only first 50 chars of role
                    "temperature": self.temperature,
                },
            ):
                response = self._generate(prompt)
                logger.debug(f"NCriticsCritic: Generated response of length {len(response)}")

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        f"NCriticsCritic: No JSON found in response for role: {role[:30]}, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    return {
                        "role": role,
                        "needs_improvement": True,
                        "score": 5,
                        "issues": ["Unable to parse critique response"],
                        "suggestions": ["General improvement needed"],
                        "explanation": "Unable to parse critique response, but proceeding with improvement",
                        "processing_time_ms": processing_time,
                    }

                # Parse JSON
                json_str = response[json_start:json_end]
                critique = json.loads(json_str)

                # Ensure all required fields are present
                critique.setdefault("role", role)
                critique.setdefault("needs_improvement", True)
                critique.setdefault("score", 5)
                critique.setdefault("issues", ["General improvement needed"])
                critique.setdefault("suggestions", ["Improve based on the feedback provided"])
                critique.setdefault("explanation", "Text needs improvement")

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                critique["processing_time_ms"] = processing_time

                # Log successful critique generation
                logger.debug(
                    f"NCriticsCritic: Successfully generated critique for role: {role[:30]} "
                    f"in {processing_time:.2f}ms, score: {critique.get('score', 'N/A')}"
                )

                # Explicitly create a Dict[str, Any] to return
                critique_result: Dict[str, Any] = critique
                return critique_result

        except json.JSONDecodeError as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="parse_json")

            # Failed to parse JSON, create a default response
            logger.warning(
                f"NCriticsCritic: Failed to parse JSON in response for role: {role[:30]}: {str(e)}"
            )
            # Create a Dict[str, Any] to return
            json_error_critique: Dict[str, Any] = {
                "role": role,
                "needs_improvement": True,
                "score": 5,
                "issues": ["Unable to parse critique response"],
                "suggestions": ["General improvement needed"],
                "explanation": "Unable to parse critique response, but proceeding with improvement",
                "processing_time_ms": processing_time,
                "error": str(e),
            }
            return json_error_critique

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="generate_critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error generating critique from specialized critic: {str(e)}",
                component="NCriticsCritic",
                operation="generate_critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "role": role[:50],  # Include only first 50 chars of role
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _aggregate_critiques(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate critiques from multiple specialized critics.

        Args:
            critiques: List of critiques from specialized critics.

        Returns:
            A dictionary with the aggregated critique.
        """
        start_time = time.time()

        logger.debug(f"NCriticsCritic: Aggregating {len(critiques)} critiques")

        try:
            # Extract all issues and suggestions
            all_issues = []
            all_suggestions = []
            average_score = 0.0

            for critique in critiques:
                all_issues.extend(critique.get("issues", []))
                all_suggestions.extend(critique.get("suggestions", []))
                average_score += critique.get("score", 5)

            # Calculate average score
            if critiques:
                average_score /= len(critiques)

            logger.debug(
                f"NCriticsCritic: Extracted {len(all_issues)} issues, {len(all_suggestions)} suggestions, "
                f"average score: {average_score:.1f}"
            )

            # Generate a summary of the critiques
            critiques_summary = "\n\n".join(
                [
                    f"Critic: {critique.get('role', 'Unknown')}\n"
                    f"Score: {critique.get('score', 5)}/10\n"
                    f"Explanation: {critique.get('explanation', 'No explanation provided')}"
                    for critique in critiques
                ]
            )

            prompt = f"""
            You are an expert at aggregating feedback from multiple critics. Please synthesize the following critiques
            into a coherent summary that captures the key issues and suggestions:

            {critiques_summary}

            Your summary should:
            1. Identify the most important issues across all critiques
            2. Highlight the most valuable suggestions
            3. Provide a balanced assessment of the text's strengths and weaknesses
            4. Be concise but comprehensive

            Summary:
            """

            # Generate summary using the model
            with critic_context(
                critic_name="NCriticsCritic",
                operation="aggregate_critiques",
                message_prefix="Failed to aggregate critiques",
                suggestions=[
                    "Check if the model is properly configured",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "critiques_count": len(critiques),
                    "issues_count": len(all_issues),
                    "suggestions_count": len(all_suggestions),
                    "temperature": self.temperature,
                },
            ):
                summary = self._generate(prompt)
                logger.debug(f"NCriticsCritic: Generated summary of length {len(summary)}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful aggregation
            logger.debug(
                f"NCriticsCritic: Successfully aggregated critiques in {processing_time:.2f}ms"
            )

            return {
                "summary": summary,
                "issues": all_issues,
                "suggestions": all_suggestions,
                "average_score": average_score,
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="aggregate_critiques")

            # If generation fails, create a simple summary
            logger.warning(
                f"NCriticsCritic: Failed to aggregate critiques: {str(e)}, using default summary"
            )

            # Extract basic information even in case of failure
            all_issues = []
            all_suggestions = []
            average_score = 0.0

            try:
                for critique in critiques:
                    all_issues.extend(critique.get("issues", []))
                    all_suggestions.extend(critique.get("suggestions", []))
                    average_score += critique.get("score", 5)

                # Calculate average score
                if critiques:
                    average_score /= len(critiques)
            except Exception:
                # If even basic extraction fails, use empty lists
                all_issues = ["Unable to extract issues from critiques"]
                all_suggestions = ["General improvement needed"]
                average_score = 5.0

            summary = "Multiple issues were identified by the critics. The text needs improvement in several areas."

            return {
                "summary": summary,
                "issues": all_issues,
                "suggestions": all_suggestions,
                "average_score": average_score,
                "processing_time_ms": processing_time,
                "error": str(e),
            }

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the N-Critics technique.

        This method implements the iterative refinement process from the N-Critics paper,
        using feedback from multiple critics to guide the improvement.

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
            f"NCriticsCritic: Improving text of length {len(text)} with max {self.max_refinement_iterations} iterations"
        )

        current_text = text
        current_critique = critique

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="NCriticsCritic",
                operation="improve",
                message_prefix="Failed to improve text with N-Critics",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "max_iterations": self.max_refinement_iterations,
                    "temperature": self.temperature,
                },
            ):
                # Iterative refinement process
                for iteration_num in range(self.max_refinement_iterations):
                    iteration_start_time = time.time()

                    logger.debug(
                        f"NCriticsCritic: Starting improvement iteration {iteration_num+1}/{self.max_refinement_iterations}"
                    )

                    try:
                        # Generate improved text based on aggregated critique
                        improved_text = self._generate_improved_text(
                            current_text,
                            current_critique["aggregated_critique"]["summary"],
                            current_critique["critic_critiques"],
                        )

                        # Re-evaluate the improved text
                        improved_critique = self._critique(improved_text)

                        # Check if the text has improved
                        current_score = current_critique["aggregated_critique"]["average_score"]
                        improved_score = improved_critique["aggregated_critique"]["average_score"]

                        # Calculate iteration time
                        iteration_time = (
                            time.time() - iteration_start_time
                        ) * 1000  # Convert to milliseconds

                        if improved_score > current_score:
                            logger.debug(
                                f"NCriticsCritic: Iteration {iteration_num+1} improved score from {current_score:.1f} to "
                                f"{improved_score:.1f} in {iteration_time:.2f}ms"
                            )
                            current_text = improved_text
                            current_critique = improved_critique
                        else:
                            # No improvement, stop iterating
                            logger.debug(
                                f"NCriticsCritic: Iteration {iteration_num+1} did not improve score "
                                f"({improved_score:.1f} <= {current_score:.1f}), stopping iterations"
                            )
                            break

                        # If the score is high enough, stop iterating
                        if improved_score >= 9.0:
                            logger.debug(
                                f"NCriticsCritic: Achieved high score ({improved_score:.1f} >= 9.0), stopping iterations"
                            )
                            break

                    except Exception as error:
                        # Log the error but continue with the current text
                        log_error(
                            error,
                            logger,
                            component="NCriticsCritic",
                            operation=f"improve_iteration_{iteration_num+1}",
                        )
                        logger.warning(
                            f"NCriticsCritic: Error in iteration {iteration_num+1}, stopping iterations: {str(error)}"
                        )
                        break

                # Calculate total processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful improvement
                logger.debug(
                    f"NCriticsCritic: Successfully improved text in {processing_time:.2f}ms"
                )

                return current_text

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="NCriticsCritic", operation="improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text with N-Critics: {str(e)}",
                component="NCriticsCritic",
                operation="improve",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "max_iterations": self.max_refinement_iterations,
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _generate_improved_text(
        self, text: str, summary: str, critiques: List[Dict[str, Any]]
    ) -> str:
        """Generate improved text based on critiques.

        Args:
            text: The text to improve.
            summary: The summary of the critiques.
            critiques: The critiques from specialized critics.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        start_time = time.time()

        logger.debug(
            f"NCriticsCritic: Generating improved text based on {len(critiques)} critiques"
        )

        try:
            # Format critiques for the prompt
            critiques_text = "\n\n".join(
                [
                    f"Critic: {critique.get('role', 'Unknown')}\n"
                    f"Issues: {', '.join(critique.get('issues', ['No issues identified']))}\n"
                    f"Suggestions: {', '.join(critique.get('suggestions', ['No suggestions provided']))}"
                    for critique in critiques
                ]
            )

            prompt = f"""
            You are a language refinement agent as described in the paper "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).

            Your task is to improve the following text based on the critiques provided by an ensemble of specialized critics:

            Original text:
            ```
            {text}
            ```

            Summary of critiques:
            {summary}

            Detailed critiques:
            {critiques_text}

            Please rewrite the text to address the issues identified by the critics. Maintain the
            original meaning and intent, but improve the quality based on the feedback.

            Improved text:
            """

            # Generate improved text using the model
            with critic_context(
                critic_name="NCriticsCritic",
                operation="generate_improved_text",
                message_prefix="Failed to generate improved text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the prompt is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "critiques_count": len(critiques),
                    "summary_length": len(summary),
                    "temperature": self.temperature,
                },
            ):
                response = self._generate(prompt)
                logger.debug(
                    f"NCriticsCritic: Generated improved text response of length {len(response)}"
                )

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()
                    logger.debug(
                        "NCriticsCritic: Removed markdown code block markers from response"
                    )

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful generation
                logger.debug(
                    f"NCriticsCritic: Successfully generated improved text in {processing_time:.2f}ms, "
                    f"length: {len(improved_text)}"
                )

                return improved_text

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(
                e,
                logger,
                component="NCriticsCritic",
                operation="generate_improved_text",
            )

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error generating improved text: {str(e)}",
                component="NCriticsCritic",
                operation="generate_improved_text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the prompt is not too long for the model",
                    "Try with a different model or temperature",
                ],
                metadata={
                    "text_length": len(text),
                    "critiques_count": len(critiques),
                    "summary_length": len(summary),
                    "temperature": self.temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )


@register_improver("n_critics")
def create_n_critics_critic(
    model: Model,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    num_critics: int = 3,
    max_refinement_iterations: int = 2,
    **options: Any,
) -> NCriticsCritic:
    """Create an N-Critics critic.

    This factory function creates an NCriticsCritic based on the paper
    "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics" (Mousavi et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        num_critics: Number of specialized critics to use.
        max_refinement_iterations: Maximum number of refinement iterations.
        **options: Additional options to pass to the NCriticsCritic.

    Returns:
        An NCriticsCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    start_time = time.time()

    logger.debug(
        f"Creating NCriticsCritic with model={model.__class__.__name__ if model else None}, "
        f"num_critics={num_critics}, max_refinement_iterations={max_refinement_iterations}, "
        f"temperature={temperature}"
    )

    try:
        # Create the critic with error handling
        with critic_context(
            critic_name="NCriticsCritic",
            operation="creation",
            message_prefix="Failed to create NCriticsCritic",
            suggestions=[
                "Check if the model is properly configured",
                "Verify that the parameters are valid",
            ],
            metadata={
                "model_type": model.__class__.__name__ if model else None,
                "num_critics": num_critics,
                "max_refinement_iterations": max_refinement_iterations,
                "temperature": temperature,
            },
        ):
            critic = NCriticsCritic(
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                num_critics=num_critics,
                max_refinement_iterations=max_refinement_iterations,
                **options,
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful creation
            logger.debug(f"Successfully created NCriticsCritic in {processing_time:.2f}ms")

            return critic

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="NCriticsCritic", operation="creation")

        # Re-raise as ImproverError with more context if not already an ImproverError
        if not isinstance(e, ImproverError):
            raise ImproverError(
                message=f"Failed to create NCriticsCritic: {str(e)}",
                component="NCriticsCritic",
                operation="creation",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the parameters are valid",
                    "Check the error message for details",
                ],
                metadata={
                    "model_type": model.__class__.__name__ if model else None,
                    "num_critics": num_critics,
                    "max_refinement_iterations": max_refinement_iterations,
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
