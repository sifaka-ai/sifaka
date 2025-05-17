"""
Retrieval-enhanced critics for Sifaka.

This module provides base classes and utilities for enhancing critics with retrieval capabilities.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Type, TypeVar

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError, RetrieverError
from sifaka.retrievers.base import Retriever
from sifaka.retrievers.augmenter import RetrievalAugmenter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Critic)


class RetrievalEnhancedCritic(Critic):
    """Base class for retrieval-enhanced critics.

    This class provides a foundation for enhancing any critic with retrieval capabilities.
    It wraps an existing critic and adds retrieval functionality to its critique and improve methods.

    Attributes:
        base_critic: The base critic to enhance with retrieval.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
    """

    def __init__(
        self,
        base_critic: Critic,
        retrieval_augmenter: RetrievalAugmenter,
        include_passages_in_critique: bool = True,
        include_passages_in_improve: bool = True,
        max_passages: int = 5,
        **options: Any,
    ):
        """Initialize the retrieval-enhanced critic.

        Args:
            base_critic: The base critic to enhance with retrieval.
            retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
            include_passages_in_critique: Whether to include retrieved passages in the critique.
            include_passages_in_improve: Whether to include retrieved passages in the improve method.
            max_passages: Maximum number of passages to retrieve.
            **options: Additional options to pass to the base critic.

        Raises:
            ImproverError: If the base critic or retrieval augmenter is not provided.
        """
        from sifaka.utils.error_handling import log_error

        # Validate base critic
        if not base_critic:
            error_msg = "Base critic not provided"
            logger.error(error_msg)
            raise ImproverError(
                message=error_msg,
                improver_name="RetrievalEnhancedCritic",
                component="Critic",
                operation="initialization",
                suggestions=["Provide a valid base critic instance"],
                metadata={"retrieval_augmenter_provided": retrieval_augmenter is not None},
            )

        # Validate retrieval augmenter
        if not retrieval_augmenter:
            error_msg = "Retrieval augmenter not provided"
            logger.error(error_msg)
            raise ImproverError(
                message=error_msg,
                improver_name="RetrievalEnhancedCritic",
                component="Critic",
                operation="initialization",
                suggestions=["Provide a valid retrieval augmenter instance"],
                metadata={"base_critic_provided": base_critic is not None},
            )

        try:
            # Initialize with the base critic's model and system prompt
            name = options.get("name", f"RetrievalEnhanced{base_critic.__class__.__name__}")
            super().__init__(
                model=base_critic.model,
                system_prompt=base_critic.system_prompt,
                temperature=base_critic.temperature,
                name=name,
                **options,
            )

            self.base_critic = base_critic
            self.retrieval_augmenter = retrieval_augmenter
            self.include_passages_in_critique = include_passages_in_critique
            self.include_passages_in_improve = include_passages_in_improve
            self.max_passages = max_passages

            # Store retrieval context for reuse between critique and improve
            self._retrieval_context: Optional[Dict[str, Any]] = None

            # Log successful initialization
            logger.debug(
                f"Initialized {self.name} with base_critic={base_critic.__class__.__name__}, "
                f"retrieval_augmenter={retrieval_augmenter.__class__.__name__}, "
                f"max_passages={max_passages}"
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="initialization")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Failed to initialize retrieval-enhanced critic: {str(e)}",
                improver_name="RetrievalEnhancedCritic",
                component="Critic",
                operation="initialization",
                suggestions=[
                    "Check that the base critic is properly initialized",
                    "Verify that the retrieval augmenter is properly configured",
                ],
                metadata={
                    "base_critic_type": base_critic.__class__.__name__,
                    "retrieval_augmenter_type": retrieval_augmenter.__class__.__name__,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    def validate(self, text: str) -> bool:
        """Validate text using the base critic.

        Args:
            text: The text to validate.

        Returns:
            True if the text is valid, False otherwise.
        """
        return self.base_critic.validate(text)

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the base critic enhanced with retrieval.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
        import time
        from sifaka.utils.error_handling import improvement_context, log_error

        start_time = time.time()

        try:
            # Get retrieval context
            with improvement_context(
                improver_name=self.name,
                operation="retrieval",
                message_prefix="Failed to retrieve passages",
                suggestions=[
                    "Check the retrieval augmenter configuration",
                    "Verify that the retrieval index is properly set up",
                ],
                metadata={
                    "text_length": len(text),
                    "retrieval_augmenter_type": self.retrieval_augmenter.__class__.__name__,
                },
            ):
                logger.debug(f"{self.name}: Retrieving passages for text of length {len(text)}")
                self._retrieval_context = self.retrieval_augmenter.get_retrieval_context(text)
                logger.debug(
                    f"{self.name}: Retrieved {self._retrieval_context.get('passage_count', 0)} passages"
                )

            # Get base critique from the base critic
            with improvement_context(
                improver_name=self.name,
                operation="base_critique",
                message_prefix="Failed to get base critique",
                suggestions=["Check the base critic configuration"],
                metadata={
                    "text_length": len(text),
                    "base_critic_type": self.base_critic.__class__.__name__,
                },
            ):
                logger.debug(
                    f"{self.name}: Getting base critique from {self.base_critic.__class__.__name__}"
                )
                base_critique = self.base_critic._critique(text)
                logger.debug(f"{self.name}: Successfully got base critique")

            # Add retrieval context to the critique if enabled
            if (
                self.include_passages_in_critique
                and self._retrieval_context.get("passage_count", 0) > 0
            ):
                base_critique["retrieved_passages"] = self._retrieval_context["passages"]
                base_critique["formatted_passages"] = self._retrieval_context["formatted_passages"]
                base_critique["passage_count"] = self._retrieval_context["passage_count"]

                # Add a suggestion to incorporate retrieved information if not already present
                if "suggestions" in base_critique and isinstance(
                    base_critique["suggestions"], list
                ):
                    base_critique["suggestions"].append(
                        "Incorporate information from retrieved passages"
                    )
                else:
                    base_critique["suggestions"] = [
                        "Incorporate information from retrieved passages"
                    ]

                # Add an issue related to retrieved information if not already present
                if "issues" in base_critique and isinstance(base_critique["issues"], list):
                    base_critique["issues"].append(
                        "Could be enhanced with additional information from retrieved sources"
                    )
                else:
                    base_critique["issues"] = [
                        "Could be enhanced with additional information from retrieved sources"
                    ]

                logger.debug(
                    f"{self.name}: Added {self._retrieval_context['passage_count']} retrieved passages to critique"
                )

            # Add processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            base_critique["processing_time_ms"] = processing_time

            logger.debug(f"{self.name}: Completed critique in {processing_time:.2f}ms")

            return base_critique

        except RetrieverError as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="retrieval")

            # Try to fall back to base critic without retrieval
            try:
                logger.warning(
                    f"{self.name}: Retrieval failed, falling back to base critic without retrieval: {str(e)}"
                )
                return self.base_critic._critique(text)
            except Exception as fallback_error:
                # Log the fallback error
                log_error(
                    fallback_error,
                    logger,
                    component="RetrievalEnhancedCritic",
                    operation="fallback_critique",
                )

                # Raise as ImproverError with more context
                raise ImproverError(
                    message=f"Failed to critique text with retrieval enhancement and fallback also failed: {str(e)}, fallback error: {str(fallback_error)}",
                    improver_name=self.name,
                    component="Critic",
                    operation="critique",
                    suggestions=[
                        "Check the retrieval augmenter configuration",
                        "Verify that the base critic is working properly",
                    ],
                    metadata={
                        "text_length": len(text),
                        "retrieval_error": str(e),
                        "fallback_error": str(fallback_error),
                        "error_type": f"{type(e).__name__} -> {type(fallback_error).__name__}",
                    },
                )

        except ImproverError as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="critique")

            # Re-raise with more context
            raise ImproverError(
                message=f"Base critic failed to critique text: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="critique",
                suggestions=["Check the base critic configuration"],
                metadata={
                    "text_length": len(text),
                    "base_critic_type": self.base_critic.__class__.__name__,
                    "error_type": type(e).__name__,
                    "has_retrieval_context": self._retrieval_context is not None,
                },
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="critique")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Unexpected error critiquing text with retrieval enhancement: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="critique",
                suggestions=[
                    "Check the retrieval augmenter configuration",
                    "Verify that the base critic is working properly",
                ],
                metadata={
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                    "has_retrieval_context": self._retrieval_context is not None,
                },
            )

    def _improve(self, text: str, critique: Dict[str, Any]) -> str:
        """Improve text using the base critic enhanced with retrieval.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the text cannot be improved.
        """
        import time
        from sifaka.utils.error_handling import improvement_context, log_error

        start_time = time.time()

        try:
            # If we don't have retrieval context or it's not enabled for improve, use base improve
            if (
                not self.include_passages_in_improve
                or not self._retrieval_context
                or self._retrieval_context.get("passage_count", 0) == 0
            ):
                logger.debug(
                    f"{self.name}: No retrieval context available, using base critic for improvement"
                )
                with improvement_context(
                    improver_name=self.name,
                    operation="base_improve",
                    message_prefix="Failed to improve text with base critic",
                    suggestions=["Check the base critic configuration"],
                    metadata={
                        "text_length": len(text),
                        "base_critic_type": self.base_critic.__class__.__name__,
                    },
                ):
                    improved_text = self.base_critic._improve(text, critique)
                    logger.debug(
                        f"{self.name}: Successfully improved text with base critic, "
                        f"length before={len(text)}, length after={len(improved_text)}"
                    )
                    return improved_text

            # Create a custom improve prompt that includes retrieved information
            with improvement_context(
                improver_name=self.name,
                operation="create_prompt",
                message_prefix="Failed to create improvement prompt",
                suggestions=["Check the critique format", "Verify retrieval context structure"],
                metadata={
                    "text_length": len(text),
                    "critique_keys": (
                        list(critique.keys()) if isinstance(critique, dict) else "not_a_dict"
                    ),
                    "has_retrieval_context": self._retrieval_context is not None,
                },
            ):
                logger.debug(f"{self.name}: Creating improvement prompt with retrieval enhancement")
                prompt = self._create_improve_prompt(text, critique)
                logger.debug(f"{self.name}: Created improvement prompt of length {len(prompt)}")

            # Generate improved text
            with improvement_context(
                improver_name=self.name,
                operation="generate",
                message_prefix="Failed to generate improved text",
                suggestions=["Check the model configuration", "Verify prompt format"],
                metadata={
                    "prompt_length": len(prompt),
                    "model_type": self.model.__class__.__name__,
                },
            ):
                logger.debug(
                    f"{self.name}: Generating improved text with model {self.model.__class__.__name__}"
                )
                response = self._generate(prompt)
                logger.debug(f"{self.name}: Generated response of length {len(response)}")

            # Extract improved text from response
            improved_text = response.strip()

            # Remove any markdown code block markers
            if improved_text.startswith("```") and improved_text.endswith("```"):
                improved_text = improved_text[3:-3].strip()
                logger.debug(f"{self.name}: Removed code block markers from response")

            # Log improvement statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.debug(
                f"{self.name}: Completed improvement in {processing_time:.2f}ms, "
                f"length before={len(text)}, length after={len(improved_text)}"
            )

            return improved_text

        except ImproverError as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="improve")

            # Try to fall back to base critic without retrieval
            try:
                logger.warning(
                    f"{self.name}: Improvement with retrieval failed, falling back to base critic: {str(e)}"
                )
                return self.base_critic._improve(text, critique)
            except Exception as fallback_error:
                # Log the fallback error
                log_error(
                    fallback_error,
                    logger,
                    component="RetrievalEnhancedCritic",
                    operation="fallback_improve",
                )

                # Raise as ImproverError with more context
                raise ImproverError(
                    message=f"Failed to improve text with retrieval enhancement and fallback also failed: {str(e)}, fallback error: {str(fallback_error)}",
                    improver_name=self.name,
                    component="Critic",
                    operation="improve",
                    suggestions=[
                        "Check the model configuration",
                        "Verify that the base critic is working properly",
                    ],
                    metadata={
                        "text_length": len(text),
                        "improve_error": str(e),
                        "fallback_error": str(fallback_error),
                        "error_type": f"{type(e).__name__} -> {type(fallback_error).__name__}",
                    },
                )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="RetrievalEnhancedCritic", operation="improve")

            # Raise as ImproverError with more context
            raise ImproverError(
                message=f"Unexpected error improving text with retrieval enhancement: {str(e)}",
                improver_name=self.name,
                component="Critic",
                operation="improve",
                suggestions=[
                    "Check the model configuration",
                    "Verify the prompt format",
                    "Check if the retrieval context is properly formatted",
                ],
                metadata={
                    "text_length": len(text),
                    "error_type": type(e).__name__,
                    "has_retrieval_context": self._retrieval_context is not None,
                    "critique_keys": (
                        list(critique.keys()) if isinstance(critique, dict) else "not_a_dict"
                    ),
                },
            )

    def _create_improve_prompt(self, text: str, critique: Dict[str, Any]) -> str:
        """Create a prompt for improving text with retrieval enhancement.

        Args:
            text: The text to improve.
            critique: The critique information.

        Returns:
            A prompt for improving the text.
        """
        # Format issues and suggestions
        issues_str = ""
        if "issues" in critique and isinstance(critique["issues"], list) and critique["issues"]:
            issues_str = "Issues:\n" + "\n".join(f"- {issue}" for issue in critique["issues"])

        suggestions_str = ""
        if (
            "suggestions" in critique
            and isinstance(critique["suggestions"], list)
            and critique["suggestions"]
        ):
            suggestions_str = "Suggestions:\n" + "\n".join(
                f"- {suggestion}" for suggestion in critique["suggestions"]
            )

        # Create the prompt
        prompt = f"""
        Please improve the following text based on the critique and retrieved information:

        Original text:
        ```
        {text}
        ```

        {issues_str}

        {suggestions_str}

        Retrieved information:
        {self._retrieval_context["formatted_passages"]}

        Instructions:
        1. Address all issues mentioned in the critique
        2. Incorporate relevant information from the retrieved passages
        3. Maintain the original style and tone
        4. Ensure factual accuracy and logical coherence
        5. Cite the passage number when incorporating information (e.g., [Passage 1])

        Improved text:
        """

        return prompt


def enhance_critic_with_retrieval(
    critic: T,
    retrieval_augmenter: RetrievalAugmenter,
    include_passages_in_critique: bool = True,
    include_passages_in_improve: bool = True,
    max_passages: int = 5,
    **options: Any,
) -> RetrievalEnhancedCritic:
    """Enhance a critic with retrieval capabilities.

    Args:
        critic: The critic to enhance with retrieval.
        retrieval_augmenter: The retrieval augmenter to use for retrieving passages.
        include_passages_in_critique: Whether to include retrieved passages in the critique.
        include_passages_in_improve: Whether to include retrieved passages in the improve method.
        max_passages: Maximum number of passages to retrieve.
        **options: Additional options to pass to the enhanced critic.

    Returns:
        A retrieval-enhanced critic.

    Raises:
        ImproverError: If the critic or retrieval augmenter is not provided.
    """
    import logging
    from sifaka.utils.error_handling import log_error

    logger = logging.getLogger(__name__)

    # Log enhancement attempt
    logger.debug(
        f"Enhancing critic {critic.__class__.__name__} with retrieval using "
        f"{retrieval_augmenter.__class__.__name__}, max_passages={max_passages}"
    )

    try:
        # Create the enhanced critic
        enhanced_critic = RetrievalEnhancedCritic(
            base_critic=critic,
            retrieval_augmenter=retrieval_augmenter,
            include_passages_in_critique=include_passages_in_critique,
            include_passages_in_improve=include_passages_in_improve,
            max_passages=max_passages,
            **options,
        )

        # Log successful enhancement
        logger.debug(
            f"Successfully enhanced critic {critic.__class__.__name__} with retrieval, "
            f"created {enhanced_critic.__class__.__name__}"
        )

        return enhanced_critic

    except Exception as e:
        # Log the error
        log_error(e, logger, component="RetrievalEnhancedCriticFactory", operation="enhance_critic")

        # Raise as ImproverError with more context
        raise ImproverError(
            message=f"Failed to enhance critic with retrieval: {str(e)}",
            improver_name="RetrievalEnhancedCritic",
            component="CriticFactory",
            operation="enhance_critic",
            suggestions=[
                "Check that the critic is properly initialized",
                "Verify that the retrieval augmenter is properly configured",
            ],
            metadata={
                "critic_type": critic.__class__.__name__,
                "retrieval_augmenter_type": retrieval_augmenter.__class__.__name__,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
