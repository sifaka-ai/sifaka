"""
Self-RAG critic for Sifaka.

This module provides a critic that uses Self-Retrieval Augmented Generation.

Based on the paper:
"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
arXiv:2310.11511 [cs.CL]
https://arxiv.org/abs/2310.11511
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from sifaka.critics.base import Critic
from sifaka.errors import ImproverError, RetrieverError
from sifaka.models.base import Model
from sifaka.registry import register_improver
from sifaka.retrievers.base import Retriever
from sifaka.utils.error_handling import critic_context, log_error

# Configure logger
logger = logging.getLogger(__name__)


class SelfRAGCritic(Critic):
    """Critic that uses Self-Retrieval Augmented Generation.

    This critic implements the Self-RAG technique, which combines retrieval
    and generation to improve text by retrieving relevant information.

    Attributes:
        model: The model to use for critiquing and improving text.
        retriever: A retriever object or function that retrieves relevant information.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        reflection_enabled: Whether to enable reflection on retrieved information.
        max_passages: Maximum number of passages to retrieve.
    """

    def __init__(
        self,
        model: Model,
        retriever: Union[Retriever, Callable[[str], List[str]]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        reflection_enabled: bool = True,
        max_passages: int = 5,
        **options: Any,
    ):
        """Initialize the Self-RAG critic.

        Args:
            model: The model to use for critiquing and improving text.
            retriever: A retriever object or function that retrieves relevant information.
            system_prompt: The system prompt to use for the model.
            temperature: The temperature to use for the model.
            reflection_enabled: Whether to enable reflection on retrieved information.
            max_passages: Maximum number of passages to retrieve.
            **options: Additional options to pass to the model.

        Raises:
            ImproverError: If the model or retriever is not provided.
        """
        # Log initialization attempt
        logger.debug(
            f"Initializing SelfRAGCritic with model={model.__class__.__name__}, "
            f"retriever={retriever.__class__.__name__ if hasattr(retriever, '__class__') else 'function'}, "
            f"temperature={temperature}, reflection_enabled={reflection_enabled}, max_passages={max_passages}"
        )

        try:
            # Use default system prompt if not provided
            if system_prompt is None:
                system_prompt = (
                    "You are an expert editor who specializes in information retrieval and generation. "
                    "Your goal is to improve text by retrieving and incorporating relevant information. "
                    "Always provide accurate information based on the retrieved passages."
                )
                logger.debug("Using default system prompt for SelfRAGCritic")

            # Initialize the base critic
            with critic_context(
                critic_name="SelfRAGCritic",
                operation="initialization",
                message_prefix="Failed to initialize SelfRAGCritic",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the retriever is properly configured",
                ],
                metadata={
                    "model_type": model.__class__.__name__,
                    "retriever_type": (
                        retriever.__class__.__name__
                        if hasattr(retriever, "__class__")
                        else "function"
                    ),
                    "temperature": temperature,
                    "reflection_enabled": reflection_enabled,
                    "max_passages": max_passages,
                },
            ):
                super().__init__(model, system_prompt, temperature, **options)
                logger.debug("Successfully initialized base Critic")

            # Validate retriever
            if not retriever:
                logger.error("Retriever not provided to SelfRAGCritic")
                raise ImproverError(
                    message="Retriever not provided",
                    component="SelfRAGCritic",
                    operation="initialization",
                    suggestions=[
                        "Provide a retriever object or function",
                        "Use a retriever that implements the Retriever protocol",
                        "Provide a function that takes a query string and returns a list of passages",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__,
                        "temperature": temperature,
                    },
                )

            # Store configuration
            self.retriever = retriever
            self.reflection_enabled = reflection_enabled
            self.max_passages = max_passages

            # Log successful initialization
            logger.debug(
                f"Successfully initialized SelfRAGCritic with model={model.__class__.__name__}, "
                f"retriever={retriever.__class__.__name__ if hasattr(retriever, '__class__') else 'function'}"
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="SelfRAGCritic", operation="initialization")

            # Re-raise as ImproverError with more context
            if not isinstance(e, ImproverError):
                raise ImproverError(
                    message=f"Failed to initialize SelfRAGCritic: {str(e)}",
                    component="SelfRAGCritic",
                    operation="initialization",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the retriever is properly configured",
                        "Check the error message for details",
                    ],
                    metadata={
                        "model_type": model.__class__.__name__ if model else None,
                        "retriever_type": (
                            retriever.__class__.__name__
                            if hasattr(retriever, "__class__")
                            else "function" if retriever else None
                        ),
                        "temperature": temperature,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            raise

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the Self-RAG technique.

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
            f"SelfRAGCritic: Critiquing text of length {len(text)}, "
            f"reflection_enabled={self.reflection_enabled}, max_passages={self.max_passages}"
        )

        # Generate queries for retrieval
        query_prompt = f"""
        Please analyze the following text and generate 3-5 search queries to retrieve relevant information that could improve it:

        ```
        {text}
        ```

        Generate search queries that would help retrieve information to:
        1. Verify factual claims
        2. Add missing context
        3. Provide supporting evidence
        4. Fill knowledge gaps

        Format your response as JSON with the following fields:
        - "needs_improvement": boolean indicating whether the text needs improvement
        - "message": a brief summary of your analysis
        - "queries": a list of search queries
        - "areas_for_improvement": a list of areas that could be improved with additional information

        JSON response:
        """

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="SelfRAGCritic",
                operation="query_generation",
                message_prefix="Failed to generate queries",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                ],
                metadata={"text_length": len(text), "temperature": self.temperature},
            ):
                # Generate queries
                response = self._generate(query_prompt)
                logger.debug(f"SelfRAGCritic: Generated query response of length {len(response)}")

                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    # No JSON found, log the issue
                    logger.warning(
                        "SelfRAGCritic: No JSON found in query response, using default response"
                    )

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Create a default response
                    return {
                        "needs_improvement": True,
                        "message": "Unable to parse critique response, but proceeding with improvement",
                        "queries": ["general information about the topic"],
                        "areas_for_improvement": ["General improvement"],
                        "retrieved_passages": [],
                        "processing_time_ms": processing_time,
                    }

                # Parse JSON
                with critic_context(
                    critic_name="SelfRAGCritic",
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
                    logger.debug("SelfRAGCritic: Successfully parsed JSON response")

                # Ensure all required fields are present
                critique.setdefault("needs_improvement", True)
                critique.setdefault("message", "Text needs improvement")
                critique.setdefault("queries", ["general information about the topic"])
                critique.setdefault("areas_for_improvement", ["General improvement"])

                # Log queries
                logger.debug(
                    f"SelfRAGCritic: Generated {len(critique['queries'])} queries: "
                    f"{', '.join(critique['queries'][:3])}{'...' if len(critique['queries']) > 3 else ''}"
                )

            # Retrieve relevant passages for each query
            retrieved_passages: List[str] = []
            with critic_context(
                critic_name="SelfRAGCritic",
                operation="retrieval",
                message_prefix="Failed to retrieve passages",
                suggestions=[
                    "Check if the retriever is properly configured",
                    "Verify that the queries are well-formed",
                ],
                metadata={
                    "query_count": len(critique["queries"]),
                    "max_passages": self.max_passages,
                    "retriever_type": (
                        self.retriever.__class__.__name__
                        if hasattr(self.retriever, "__class__")
                        else "function"
                    ),
                },
            ):
                for query in critique["queries"][: self.max_passages]:  # Limit number of queries
                    try:
                        # Handle both function-based retrievers and Retriever objects
                        if hasattr(self.retriever, "retrieve"):
                            passages = self.retriever.retrieve(query)
                        else:
                            passages = self.retriever(query)

                        # Add the query as context to each passage
                        passages = [f"Query: {query}\n\nPassage: {passage}" for passage in passages]
                        retrieved_passages.extend(passages)

                        logger.debug(
                            f"SelfRAGCritic: Retrieved {len(passages)} passages for query '{query}'"
                        )
                    except Exception as e:
                        # Log the error
                        log_error(e, logger, component="SelfRAGCritic", operation="retrieval")

                        logger.warning(
                            f"SelfRAGCritic: Error retrieving passages for query '{query}': {str(e)}"
                        )
                        continue

                # Remove duplicates while preserving order
                seen = set()
                unique_passages = []
                for passage in retrieved_passages:
                    passage_key = passage.strip()
                    if passage_key not in seen and passage_key:
                        seen.add(passage_key)
                        unique_passages.append(passage)

                # Limit the number of passages to avoid context length issues
                unique_passages = unique_passages[: self.max_passages]

                logger.debug(f"SelfRAGCritic: Filtered to {len(unique_passages)} unique passages")

                # Add retrieved passages to critique
                critique["retrieved_passages"] = unique_passages

            # Generate reflection on retrieved passages if enabled
            if self.reflection_enabled and unique_passages:
                with critic_context(
                    critic_name="SelfRAGCritic",
                    operation="reflection",
                    message_prefix="Failed to generate reflection",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the passages are not too long for the model",
                    ],
                    metadata={
                        "text_length": len(text),
                        "passage_count": len(unique_passages),
                        "temperature": 0.3,  # Using lower temperature for reflection
                    },
                ):
                    reflection_prompt = f"""
                    Please analyze the following retrieved passages in relation to the text:

                    Text to improve:
                    ```
                    {text}
                    ```

                    Retrieved passages:
                    {self._format_passages(unique_passages)}

                    Provide a brief reflection on how these passages can be used to improve the text.
                    Focus on factual accuracy, missing context, and supporting evidence.
                    """

                    reflection = self._generate(reflection_prompt, temperature=0.3)
                    critique["reflection"] = reflection

                    logger.debug(f"SelfRAGCritic: Generated reflection of length {len(reflection)}")
            else:
                critique["reflection"] = ""
                logger.debug("SelfRAGCritic: Reflection disabled or no passages retrieved")

            # Add issues field for compatibility with base Critic
            critique["issues"] = critique.get("areas_for_improvement", [])
            critique["suggestions"] = [
                f"Incorporate information from retrieved passage {i+1}"
                for i in range(len(unique_passages))
            ]

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            critique["processing_time_ms"] = processing_time

            # Log successful critique
            logger.debug(
                f"SelfRAGCritic: Successfully critiqued text in {processing_time:.2f}ms, "
                f"found {len(critique['issues'])} issues, retrieved {len(unique_passages)} passages"
            )

            # Explicitly create a Dict[str, Any] to return
            critique_result: Dict[str, Any] = critique
            return critique_result

        except json.JSONDecodeError as e:
            # Log the error
            log_error(e, logger, component="SelfRAGCritic", operation="json_parsing")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Failed to parse JSON, create a default response
            logger.warning(f"SelfRAGCritic: Failed to parse JSON in critique response: {str(e)}")
            # Create a Dict[str, Any] to return
            json_error_critique: Dict[str, Any] = {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "queries": ["general information about the topic"],
                "areas_for_improvement": ["General improvement"],
                "retrieved_passages": [],
                "issues": ["General improvement"],
                "suggestions": ["Incorporate relevant information"],
                "reflection": "",
                "processing_time_ms": processing_time,
                "error": str(e),
            }
            return json_error_critique

        except RetrieverError as e:
            # Log the error
            log_error(e, logger, component="SelfRAGCritic", operation="retrieval")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Retriever error, create a response with the error
            logger.error(f"SelfRAGCritic: Retriever error: {str(e)}")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error retrieving passages: {str(e)}",
                component="SelfRAGCritic",
                operation="retrieval",
                suggestions=[
                    "Check if the retriever is properly configured",
                    "Verify that the retriever has access to the necessary data",
                    "Check if the retriever service is available",
                ],
                metadata={
                    "text_length": len(text),
                    "retriever_type": (
                        self.retriever.__class__.__name__
                        if hasattr(self.retriever, "__class__")
                        else "function"
                    ),
                    "error_type": "RetrieverError",
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="SelfRAGCritic", operation="critique")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            logger.error(f"SelfRAGCritic: Error critiquing text: {str(e)}")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error critiquing text: {str(e)}",
                component="SelfRAGCritic",
                operation="critique",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the text is not too long for the model",
                    "Check if the retriever is properly configured",
                ],
                metadata={
                    "text_length": len(text),
                    "retriever_type": (
                        self.retriever.__class__.__name__
                        if hasattr(self.retriever, "__class__")
                        else "function"
                    ),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )

    def _format_passages(self, passages: List[str]) -> str:
        """Format a list of passages for inclusion in a prompt.

        Args:
            passages: List of passages to format

        Returns:
            Formatted passages as a string
        """
        return "\n\n".join(f"Passage {i+1}:\n{passage}" for i, passage in enumerate(passages))

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the Self-RAG technique.

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
            f"SelfRAGCritic: Improving text of length {len(text)}, "
            f"reflection_enabled={self.reflection_enabled}"
        )

        # Format retrieved passages
        retrieved_passages = critique.get("retrieved_passages", [])

        if not retrieved_passages:
            # No passages retrieved, return original text
            logger.warning(
                "SelfRAGCritic: No passages retrieved for improvement, returning original text"
            )
            return text

        # Format areas for improvement
        areas = critique.get("areas_for_improvement", [])
        areas_str = "\n".join(f"- {area}" for area in areas)

        logger.debug(
            f"SelfRAGCritic: Improving text with {len(retrieved_passages)} passages and {len(areas)} areas for improvement"
        )

        # Include reflection if available
        reflection = critique.get("reflection", "")
        reflection_section = ""
        if reflection:
            reflection_section = f"""
            Reflection on retrieved information:
            {reflection}
            """
            logger.debug(f"SelfRAGCritic: Including reflection of length {len(reflection)}")

        prompt = f"""
        Please improve the following text by incorporating relevant information from the retrieved passages:

        Original text:
        ```
        {text}
        ```

        Areas for improvement:
        {areas_str}
        {reflection_section}
        Retrieved passages:
        {self._format_passages(retrieved_passages)}

        Instructions:
        1. Incorporate relevant information from the passages to improve the text
        2. Ensure the improved text is coherent and well-structured
        3. Maintain the original style and tone
        4. Do not add information that is not supported by the passages
        5. Cite the passage number when incorporating information (e.g., [Passage 1])
        6. Focus on addressing the areas for improvement identified above

        Improved text:
        """

        try:
            # Use critic_context for consistent error handling
            with critic_context(
                critic_name="SelfRAGCritic",
                operation="improvement",
                message_prefix="Failed to improve text",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the prompt is not too long for the model",
                ],
                metadata={
                    "text_length": len(text),
                    "passage_count": len(retrieved_passages),
                    "areas_count": len(areas),
                    "temperature": self.temperature,
                },
            ):
                # Generate improved text
                response = self._generate(prompt)
                logger.debug(
                    f"SelfRAGCritic: Generated improvement response of length {len(response)}"
                )

                # Extract improved text from response
                improved_text = response.strip()

                # Remove any markdown code block markers
                if improved_text.startswith("```") and improved_text.endswith("```"):
                    improved_text = improved_text[3:-3].strip()
                    logger.debug("SelfRAGCritic: Removed markdown code block markers from response")

            # If reflection is enabled, generate a final reflection on the improvement
            if self.reflection_enabled:
                with critic_context(
                    critic_name="SelfRAGCritic",
                    operation="final_reflection",
                    message_prefix="Failed to generate final reflection",
                    suggestions=[
                        "Check if the model is properly configured",
                        "Verify that the texts are not too long for the model",
                    ],
                    metadata={
                        "original_text_length": len(text),
                        "improved_text_length": len(improved_text),
                        "temperature": 0.3,  # Using lower temperature for reflection
                    },
                ):
                    final_reflection_prompt = f"""
                    Please analyze the following original text and the improved version:

                    Original text:
                    ```
                    {text}
                    ```

                    Improved text:
                    ```
                    {improved_text}
                    ```

                    Provide a brief reflection on how the text was improved:
                    1. What information was added?
                    2. What issues were addressed?
                    3. Is the improved text more accurate and informative?

                    Keep your reflection concise (3-5 sentences).
                    """

                    try:
                        final_reflection = self._generate(final_reflection_prompt, temperature=0.3)
                        logger.info(
                            f"SelfRAGCritic: Final reflection on improvement: {final_reflection}"
                        )
                    except Exception as e:
                        # Log the error
                        log_error(
                            e,
                            logger,
                            component="SelfRAGCritic",
                            operation="final_reflection",
                        )
                        logger.warning(
                            f"SelfRAGCritic: Error generating final reflection: {str(e)}"
                        )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log successful improvement
            logger.debug(
                f"SelfRAGCritic: Successfully improved text in {processing_time:.2f}ms, "
                f"original length: {len(text)}, improved length: {len(improved_text)}"
            )

            return improved_text

        except Exception as e:
            # Log the error
            log_error(e, logger, component="SelfRAGCritic", operation="improvement")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            logger.error(f"SelfRAGCritic: Error improving text: {str(e)}")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Error improving text: {str(e)}",
                component="SelfRAGCritic",
                operation="improvement",
                suggestions=[
                    "Check if the model is properly configured",
                    "Verify that the prompt is not too long for the model",
                    "Check if the retrieved passages are valid",
                ],
                metadata={
                    "text_length": len(text),
                    "passage_count": len(retrieved_passages),
                    "areas_count": len(areas),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )


@register_improver("self_rag")
def create_self_rag_critic(
    model: Model,
    retriever: Union[Retriever, Callable[[str], List[str]]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    reflection_enabled: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> SelfRAGCritic:
    """Create a Self-RAG critic.

    This factory function creates a SelfRAGCritic based on the paper
    "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023).
    It is registered with the registry system for dependency injection.

    Args:
        model: The model to use for critiquing and improving text.
        retriever: A retriever object or function that retrieves relevant information.
        system_prompt: The system prompt to use for the model.
        temperature: The temperature to use for the model.
        reflection_enabled: Whether to enable reflection on retrieved information.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the SelfRAGCritic.

    Returns:
        A SelfRAGCritic instance.

    Raises:
        ImproverError: If the critic cannot be created.
    """
    try:
        # Log factory function call
        logger.debug(
            f"Creating SelfRAGCritic with model={model.__class__.__name__}, "
            f"retriever={retriever.__class__.__name__ if hasattr(retriever, '__class__') else 'function'}, "
            f"temperature={temperature}, reflection_enabled={reflection_enabled}, max_passages={max_passages}"
        )

        # Create the critic
        critic = SelfRAGCritic(
            model=model,
            retriever=retriever,
            system_prompt=system_prompt,
            temperature=temperature,
            reflection_enabled=reflection_enabled,
            max_passages=max_passages,
            **options,
        )

        # Log successful creation
        logger.debug("Successfully created SelfRAGCritic")

        return critic

    except Exception as e:
        # Log the error
        log_error(e, logger, component="SelfRAGCriticFactory", operation="create_critic")

        # Raise as ImproverError with more context
        raise ImproverError(
            message=f"Failed to create SelfRAGCritic: {str(e)}",
            component="SelfRAGCriticFactory",
            operation="create_critic",
            suggestions=[
                "Check if the model is properly configured",
                "Verify that the retriever is properly configured",
                "Check the error message for details",
            ],
            metadata={
                "model_type": model.__class__.__name__ if model else None,
                "retriever_type": (
                    retriever.__class__.__name__
                    if hasattr(retriever, "__class__")
                    else "function" if retriever else None
                ),
                "temperature": temperature,
                "reflection_enabled": reflection_enabled,
                "max_passages": max_passages,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
