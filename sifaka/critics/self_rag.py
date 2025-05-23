"""Self-RAG critic for Sifaka.

This module implements the Self-Reflective Retrieval-Augmented Generation (Self-RAG)
approach for critics. Self-RAG enables language models to decide when and what to
retrieve, and reflect on the relevance and utility of the retrieved information.

Based on Self-RAG: https://arxiv.org/abs/2310.11511

The SelfRAGCritic uses the context already available in the Thought container
(populated by the Chain's retrieval orchestration) to provide enhanced critique
and improvement suggestions.
"""

import time
from typing import Any, Dict, Optional

from sifaka.core.interfaces import Model
from sifaka.core.thought import Thought
from sifaka.models.base import create_model
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class SelfRAGCritic:
    """Critic that implements Self-Reflective Retrieval-Augmented Generation.

    This critic uses the Self-RAG approach to provide enhanced critique and
    improvement by leveraging retrieved context. It analyzes both the generated
    text and the available context to provide comprehensive feedback.

    The critic assumes that retrieval has already been performed by the Chain
    and uses the context available in the Thought container.

    Attributes:
        model: The language model to use for critique and improvement.
        retrieval_threshold: Threshold for determining when retrieval is needed.
        reflection_enabled: Whether to enable reflection on retrieved information.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        model_name: Optional[str] = None,
        retrieval_threshold: float = 0.7,
        reflection_enabled: bool = True,
        critique_prompt_template: Optional[str] = None,
        improve_prompt_template: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """Initialize the Self-RAG critic.

        Args:
            model: The language model to use for critique and improvement.
            model_name: The name of the model to use if model is not provided.
            retrieval_threshold: Threshold for determining retrieval quality.
            reflection_enabled: Whether to enable reflection on retrieved information.
            critique_prompt_template: Template for the critique prompt.
            improve_prompt_template: Template for the improvement prompt.
            **model_kwargs: Additional keyword arguments for model creation.
        """
        # Set up the model
        if model:
            self.model = model
        elif model_name:
            self.model = create_model(model_name, **model_kwargs)
        else:
            # Default to a mock model for testing
            self.model = create_model("mock:default", **model_kwargs)

        self.retrieval_threshold = retrieval_threshold
        self.reflection_enabled = reflection_enabled

        # Set up prompt templates
        self.critique_prompt_template = critique_prompt_template or (
            "Analyze the following text and retrieved context to provide a comprehensive critique.\n\n"
            "Original prompt: {prompt}\n\n"
            "Generated text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Please evaluate:\n"
            "1. How well does the text address the original prompt?\n"
            "2. How effectively does the text use the retrieved context?\n"
            "3. Are there any factual inconsistencies between the text and context?\n"
            "4. What improvements could be made?\n\n"
            "Provide your analysis in a structured format with specific suggestions."
        )

        self.improve_prompt_template = improve_prompt_template or (
            "Improve the following text based on the critique and available context.\n\n"
            "Original prompt: {prompt}\n\n"
            "Original text:\n{text}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Critique:\n{critique}\n\n"
            "Please provide an improved version that:\n"
            "1. Better addresses the original prompt\n"
            "2. More effectively incorporates relevant information from the context\n"
            "3. Addresses the issues identified in the critique\n"
            "4. Maintains factual accuracy with the retrieved context"
        )

    def critique(self, thought: Thought) -> Dict[str, Any]:
        """Critique text using Self-RAG approach.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A dictionary with critique results including reflection on retrieval.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRAGCritic",
            operation="critique",
            message_prefix="Failed to critique text with Self-RAG",
        ):
            # Check if text is available
            if not thought.text:
                return {
                    "needs_improvement": True,
                    "message": "No text available for critique",
                    "issues": ["Text is empty or None"],
                    "suggestions": ["Provide text to critique"],
                    "retrieval_reflection": "No text to analyze",
                }

            # Prepare context from retrieved documents
            context = self._prepare_context(thought)

            # Create critique prompt
            critique_prompt = self.critique_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
            )

            # Generate critique
            critique_response = self.model.generate(
                prompt=critique_prompt,
                system_prompt="You are an expert critic analyzing text quality and retrieval effectiveness.",
            )

            # Perform retrieval reflection if enabled
            retrieval_reflection = ""
            if self.reflection_enabled:
                retrieval_reflection = self._reflect_on_retrieval(thought, critique_response)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfRAGCritic: Critique completed in {processing_time:.2f}ms")

            # Parse critique response (simplified parsing)
            needs_improvement = (
                "improvement" in critique_response.lower() or "issue" in critique_response.lower()
            )

            return {
                "needs_improvement": needs_improvement,
                "message": critique_response,
                "critique": critique_response,
                "retrieval_reflection": retrieval_reflection,
                "context_used": len(context) > 0,
                "processing_time_ms": processing_time,
            }

    def improve(self, thought: Thought) -> str:
        """Improve text using Self-RAG approach.

        Args:
            thought: The Thought container with the text to improve and critique.

        Returns:
            The improved text.
        """
        start_time = time.time()

        with critic_context(
            critic_name="SelfRAGCritic",
            operation="improve",
            message_prefix="Failed to improve text with Self-RAG",
        ):
            # Check if text and critique are available
            if not thought.text:
                raise ImproverError(
                    message="No text available for improvement",
                    component="SelfRAGCritic",
                    operation="improve",
                    suggestions=["Provide text to improve"],
                )

            # Get critique from thought
            critique = ""
            if thought.critic_feedback:
                for feedback in thought.critic_feedback:
                    if feedback.critic_name == "SelfRAGCritic":
                        critique = feedback.feedback.get("critique", "")
                        break

            # Prepare context from retrieved documents
            context = self._prepare_context(thought)

            # Create improvement prompt
            improve_prompt = self.improve_prompt_template.format(
                prompt=thought.prompt,
                text=thought.text,
                context=context,
                critique=critique,
            )

            # Generate improved text
            improved_text = self.model.generate(
                prompt=improve_prompt,
                system_prompt="You are an expert editor improving text based on critique and context.",
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"SelfRAGCritic: Improvement completed in {processing_time:.2f}ms")

            return improved_text.strip()

    def _prepare_context(self, thought: Thought) -> str:
        """Prepare context string from retrieved documents.

        Args:
            thought: The Thought container with retrieved context.

        Returns:
            A formatted context string.
        """
        context_parts = []

        # Add pre-generation context
        if thought.pre_generation_context:
            for i, doc in enumerate(thought.pre_generation_context):
                context_parts.append(f"Document {i+1}: {doc.text}")

        # Add post-generation context
        if thought.post_generation_context:
            for i, doc in enumerate(thought.post_generation_context):
                context_parts.append(f"Additional Document {i+1}: {doc.text}")

        return "\n\n".join(context_parts) if context_parts else "No retrieved context available."

    def _reflect_on_retrieval(self, thought: Thought, critique: str) -> str:
        """Reflect on the quality and relevance of retrieved information.

        Args:
            thought: The Thought container with retrieved context.
            critique: The generated critique.

        Returns:
            A reflection on the retrieval quality.
        """
        if not thought.pre_generation_context and not thought.post_generation_context:
            return "No retrieval was performed for this task."

        # Simple reflection based on context availability and critique
        context_count = len(thought.pre_generation_context or []) + len(
            thought.post_generation_context or []
        )

        if context_count == 0:
            return "Retrieval was attempted but no relevant documents were found."
        elif "context" in critique.lower() and "relevant" in critique.lower():
            return f"Retrieved {context_count} documents that appear to be relevant to the task."
        elif "context" in critique.lower() and (
            "irrelevant" in critique.lower() or "not helpful" in critique.lower()
        ):
            return f"Retrieved {context_count} documents but they may not be fully relevant to the task."
        else:
            return f"Retrieved {context_count} documents for context augmentation."


def create_self_rag_critic(
    model: Optional[Model] = None,
    model_name: Optional[str] = None,
    retrieval_threshold: float = 0.7,
    reflection_enabled: bool = True,
    **model_kwargs: Any,
) -> SelfRAGCritic:
    """Create a Self-RAG critic.

    Args:
        model: The language model to use for critique and improvement.
        model_name: The name of the model to use if model is not provided.
        retrieval_threshold: Threshold for determining retrieval quality.
        reflection_enabled: Whether to enable reflection on retrieved information.
        **model_kwargs: Additional keyword arguments for model creation.

    Returns:
        A SelfRAGCritic instance.
    """
    return SelfRAGCritic(
        model=model,
        model_name=model_name,
        retrieval_threshold=retrieval_threshold,
        reflection_enabled=reflection_enabled,
        **model_kwargs,
    )
