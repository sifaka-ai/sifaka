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
import re
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
    """

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax issues.

        Args:
            json_str: The JSON string to repair.

        Returns:
            The repaired JSON string.
        """
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        # Ensure the string starts with { and ends with }
        if not json_str.startswith("{"):
            json_str = "{" + json_str
        if not json_str.endswith("}"):
            json_str = json_str + "}"

        # Replace JavaScript-style single quotes with double quotes
        # This is a simplified approach and may not handle all cases correctly
        in_string = False
        in_escape = False
        result = []

        for char in json_str:
            if char == "\\" and not in_escape:
                in_escape = True
                result.append(char)
            elif in_escape:
                in_escape = False
                result.append(char)
            elif char == '"' and not in_escape:
                in_string = not in_string
                result.append(char)
            elif char == "'" and not in_string:
                # Replace single quotes with double quotes when not in a string
                result.append('"')
            else:
                result.append(char)

        # Fix trailing commas in arrays and objects
        repaired = "".join(result)
        repaired = repaired.replace(",]", "]").replace(",}", "}")

        # Fix missing commas between key-value pairs
        # This is a very simplified approach and may not handle all cases correctly
        repaired = re.sub(r'"\s*}\s*"', '", "', repaired)

        return repaired

    def _aggressive_json_repair(self, json_str: str) -> Dict[str, Any]:
        """Attempt more aggressive JSON repair when standard parsing fails.

        This method tries to extract key-value pairs from malformed JSON using
        regular expressions and other heuristics.

        Args:
            json_str: The malformed JSON string.

        Returns:
            A dictionary with the extracted key-value pairs.
        """
        # Initialize an empty result dictionary
        result = {}

        # Try to extract role
        role_match = re.search(r'"role"\s*:\s*"([^"]+)"', json_str)
        if role_match:
            result["role"] = role_match.group(1)

        # Try to extract needs_improvement
        needs_improvement_match = re.search(
            r'"needs_improvement"\s*:\s*(true|false)', json_str, re.IGNORECASE
        )
        if needs_improvement_match:
            result["needs_improvement"] = needs_improvement_match.group(1).lower() == "true"
        else:
            result["needs_improvement"] = True

        # Try to extract score
        score_match = re.search(r'"score"\s*:\s*(\d+)', json_str)
        if score_match:
            result["score"] = int(score_match.group(1))
        else:
            result["score"] = 5

        # Try to extract issues
        issues = []
        issues_matches = re.findall(r'"issues"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if issues_matches:
            issues_str = issues_matches[0]
            # Extract quoted strings from the issues array
            issue_items = re.findall(r'"([^"]+)"', issues_str)
            issues.extend(issue_items)

        if not issues:
            issues = ["Unable to parse issues from critique response"]
        result["issues"] = issues

        # Try to extract suggestions
        suggestions = []
        suggestions_matches = re.findall(r'"suggestions"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if suggestions_matches:
            suggestions_str = suggestions_matches[0]
            # Extract quoted strings from the suggestions array
            suggestion_items = re.findall(r'"([^"]+)"', suggestions_str)
            suggestions.extend(suggestion_items)

        if not suggestions:
            suggestions = ["General improvement needed"]
        result["suggestions"] = suggestions

        # Try to extract explanation
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', json_str)
        if explanation_match:
            result["explanation"] = explanation_match.group(1)
        else:
            result["explanation"] = "Unable to parse explanation from critique response"

        return result

    def __init__(
        self,
        model: Model,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        num_critics: int = 3,
        **options: Any,
    ):
        """Initialize the N-Critics critic.

        Args:
            model: The model to use for critiquing and improving text.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            num_critics: Number of specialized critics to use.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model is not provided or if initialization fails.
        """
        start_time = time.time()

        # Log initialization attempt
        logger.debug(
            f"Initializing NCriticsCritic with model={model.__class__.__name__ if model else None}, "
            f"num_critics={num_critics}, temperature={temperature}"
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

                # Extract and repair JSON from response
                # First, try to find JSON between triple backticks
                json_str = ""
                if "```json" in response and "```" in response.split("```json", 1)[1]:
                    # Extract JSON from code block
                    json_str = response.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in response and "```" in response.split("```", 1)[1]:
                    # Extract potential JSON from generic code block
                    code_block = response.split("```", 1)[1].split("```", 1)[0].strip()
                    if code_block.startswith("{") and code_block.endswith("}"):
                        json_str = code_block

                # If no code block found, try to extract JSON directly
                if not json_str:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]

                if not json_str:
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

                # Try to repair common JSON issues
                try:
                    # Replace single quotes with double quotes (if not within strings)
                    # This is a simplified approach and may not handle all cases correctly
                    json_str = self._repair_json(json_str)

                    # Parse JSON
                    critique = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # If parsing fails, log the error with details
                    logger.warning(
                        f"NCriticsCritic: JSON parsing error for role: {role[:30]}: {str(e)}\n"
                        f"JSON string (first 100 chars): {json_str[:100]}..."
                    )

                    # Try a more aggressive repair approach
                    try:
                        # Try to extract just the key-value pairs
                        repaired_json = self._aggressive_json_repair(json_str)
                        critique = repaired_json
                    except Exception as repair_error:
                        # If aggressive repair fails, raise the original error
                        raise json.JSONDecodeError(
                            f"Failed to parse JSON: {str(e)}. Repair attempt failed: {str(repair_error)}",
                            json_str,
                            0,
                        )

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

        This method generates improved text based on feedback from multiple critics.
        It performs a single improvement step without internal iterations.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        start_time = time.time()

        logger.debug(f"NCriticsCritic: Improving text of length {len(text)}")

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
                    "temperature": self.temperature,
                },
            ):
                # Generate improved text based on aggregated critique
                improved_text = self._generate_improved_text(
                    text,
                    critique["aggregated_critique"]["summary"],
                    critique["critic_critiques"],
                )

                # Calculate total processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful improvement
                logger.debug(
                    f"NCriticsCritic: Successfully improved text in {processing_time:.2f}ms"
                )

                return improved_text

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
                    "max_iterations": self.options.get("max_refinement_iterations", 3),
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
        **options: Additional options to pass to the NCriticsCritic.

    Returns:
        An NCriticsCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    start_time = time.time()

    logger.debug(
        f"Creating NCriticsCritic with model={model.__class__.__name__ if model else None}, "
        f"num_critics={num_critics}, temperature={temperature}"
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
                "temperature": temperature,
            },
        ):
            critic = NCriticsCritic(
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                num_critics=num_critics,
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
                    "temperature": temperature,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
