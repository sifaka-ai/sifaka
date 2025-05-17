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
from typing import Dict, Any, Optional, List, Callable, Union

from sifaka.models.base import Model
from sifaka.critics.base import Critic
from sifaka.errors import ImproverError, RetrieverError
from sifaka.registry import register_improver
from sifaka.retrievers.base import Retriever

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
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert editor who specializes in information retrieval and generation. "
                "Your goal is to improve text by retrieving and incorporating relevant information. "
                "Always provide accurate information based on the retrieved passages."
            )

        super().__init__(model, system_prompt, temperature, **options)

        if not retriever:
            raise ImproverError("Retriever not provided")

        self.retriever = retriever
        self.reflection_enabled = reflection_enabled
        self.max_passages = max_passages

    def _critique(self, text: str) -> Dict[str, Any]:
        """Critique text using the Self-RAG technique.

        Args:
            text: The text to critique.

        Returns:
            A dictionary with critique information.

        Raises:
            ImproverError: If the text cannot be critiqued.
        """
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
            response = self._generate(query_prompt)

            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                # No JSON found, create a default response
                return {
                    "needs_improvement": True,
                    "message": "Unable to parse critique response, but proceeding with improvement",
                    "queries": ["general information about the topic"],
                    "areas_for_improvement": ["General improvement"],
                    "retrieved_passages": [],
                }

            json_str = response[json_start:json_end]
            critique = json.loads(json_str)

            # Ensure all required fields are present
            critique.setdefault("needs_improvement", True)
            critique.setdefault("message", "Text needs improvement")
            critique.setdefault("queries", ["general information about the topic"])
            critique.setdefault("areas_for_improvement", ["General improvement"])

            # Retrieve relevant passages for each query
            retrieved_passages = []
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
                except Exception as e:
                    logger.warning(f"Error retrieving passages for query '{query}': {str(e)}")
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

            # Add retrieved passages to critique
            critique["retrieved_passages"] = unique_passages

            # Generate reflection on retrieved passages if enabled
            if self.reflection_enabled and unique_passages:
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
            else:
                critique["reflection"] = ""

            # Add issues field for compatibility with base Critic
            critique["issues"] = critique.get("areas_for_improvement", [])
            critique["suggestions"] = [
                f"Incorporate information from retrieved passage {i+1}"
                for i in range(len(unique_passages))
            ]

            return critique
        except json.JSONDecodeError:
            # Failed to parse JSON, create a default response
            return {
                "needs_improvement": True,
                "message": "Unable to parse critique response, but proceeding with improvement",
                "queries": ["general information about the topic"],
                "areas_for_improvement": ["General improvement"],
                "retrieved_passages": [],
                "issues": ["General improvement"],
                "suggestions": ["Incorporate relevant information"],
                "reflection": "",
            }
        except Exception as e:
            logger.error(f"Error critiquing text: {str(e)}")
            raise ImproverError(f"Error critiquing text: {str(e)}")

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
        # Format retrieved passages
        retrieved_passages = critique.get("retrieved_passages", [])

        if not retrieved_passages:
            # No passages retrieved, return original text
            logger.warning("No passages retrieved for improvement, returning original text")
            return text

        # Format areas for improvement
        areas = critique.get("areas_for_improvement", [])
        areas_str = "\n".join(f"- {area}" for area in areas)

        # Include reflection if available
        reflection = critique.get("reflection", "")
        reflection_section = ""
        if reflection:
            reflection_section = f"""
            Reflection on retrieved information:
            {reflection}
            """

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
            response = self._generate(prompt)

            # Extract improved text from response
            improved_text = response.strip()

            # Remove any markdown code block markers
            if improved_text.startswith("```") and improved_text.endswith("```"):
                improved_text = improved_text[3:-3].strip()

            # If reflection is enabled, generate a final reflection on the improvement
            if self.reflection_enabled:
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
                    logger.info(f"Final reflection on improvement: {final_reflection}")
                except Exception as e:
                    logger.warning(f"Error generating final reflection: {str(e)}")

            return improved_text
        except Exception as e:
            logger.error(f"Error improving text: {str(e)}")
            raise ImproverError(f"Error improving text: {str(e)}")


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
    """
    return SelfRAGCritic(
        model=model,
        retriever=retriever,
        system_prompt=system_prompt,
        temperature=temperature,
        reflection_enabled=reflection_enabled,
        max_passages=max_passages,
        **options,
    )
