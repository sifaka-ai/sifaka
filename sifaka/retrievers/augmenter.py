"""
Retrieval augmenter for Sifaka.

This module provides a generic retrieval augmenter that can be used by multiple critics.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Set

from sifaka.models.base import Model
from sifaka.retrievers.base import Retriever
from sifaka.errors import RetrieverError

logger = logging.getLogger(__name__)


class RetrievalAugmenter:
    """Component that provides retrieval augmentation for critics.

    This class provides a generic retrieval augmentation capability that can be
    used by any critic that needs to retrieve relevant information. It handles
    query generation, retrieval, deduplication, and formatting of retrieved passages.

    Attributes:
        retriever: The retriever to use for retrieving passages.
        model: Optional model to use for generating queries.
        max_passages: Maximum number of passages to retrieve.
        max_queries: Maximum number of queries to generate.
        query_temperature: Temperature to use for query generation.
        include_query_context: Whether to include the query as context with each passage.
    """

    def __init__(
        self,
        retriever: Union[Retriever, Callable[[str], List[str]]],
        model: Optional[Model] = None,
        max_passages: int = 5,
        max_queries: int = 3,
        query_temperature: float = 0.3,
        include_query_context: bool = True,
    ):
        """Initialize the retrieval augmenter.

        Args:
            retriever: The retriever to use for retrieving passages.
            model: Optional model to use for generating queries.
            max_passages: Maximum number of passages to retrieve.
            max_queries: Maximum number of queries to generate.
            query_temperature: Temperature to use for query generation.
            include_query_context: Whether to include the query as context with each passage.

        Raises:
            RetrieverError: If the retriever is not provided.
        """
        if not retriever:
            raise RetrieverError("Retriever not provided")

        self.retriever = retriever
        self.model = model
        self.max_passages = max_passages
        self.max_queries = max_queries
        self.query_temperature = query_temperature
        self.include_query_context = include_query_context

    def retrieve(self, text: str, custom_queries: Optional[List[str]] = None) -> List[str]:
        """Retrieve relevant passages for the given text.

        Args:
            text: The text to retrieve passages for.
            custom_queries: Optional list of custom queries to use instead of generating them.

        Returns:
            A list of retrieved passages.

        Raises:
            RetrieverError: If retrieval fails.
        """
        try:
            # Use custom queries if provided, otherwise generate them
            queries = custom_queries if custom_queries is not None else self._generate_queries(text)

            # Limit the number of queries
            queries = queries[: self.max_queries]

            # Retrieve passages for each query
            all_passages: List[str] = []
            for query in queries:
                try:
                    # Handle both function-based retrievers and Retriever objects
                    if hasattr(self.retriever, "retrieve"):
                        passages = self.retriever.retrieve(query)
                    else:
                        passages = self.retriever(query)

                    # Add query context if enabled
                    if self.include_query_context:
                        passages = [f"Query: {query}\n\nPassage: {passage}" for passage in passages]

                    all_passages.extend(passages)
                except Exception as e:
                    logger.warning(f"Error retrieving passages for query '{query}': {str(e)}")
                    continue

            # Deduplicate and limit the number of passages
            unique_passages = self._deduplicate_passages(all_passages)
            return unique_passages[: self.max_passages]

        except Exception as e:
            logger.error(f"Error retrieving passages: {str(e)}")
            raise RetrieverError(f"Error retrieving passages: {str(e)}")

    def _generate_queries(self, text: str) -> List[str]:
        """Generate search queries based on the text.

        Args:
            text: The text to generate queries for.

        Returns:
            A list of generated queries.

        Raises:
            RetrieverError: If query generation fails.
        """
        # If no model is provided, use a simple heuristic approach
        if not self.model:
            return self._generate_heuristic_queries(text)

        try:
            # Generate queries using the model
            prompt = f"""
            Please analyze the following text and generate {self.max_queries} search queries to retrieve relevant information:

            ```
            {text}
            ```

            Generate search queries that would help retrieve information to:
            1. Verify factual claims
            2. Add missing context
            3. Provide supporting evidence
            4. Fill knowledge gaps

            Format your response as JSON with a single field "queries" containing a list of search queries.

            Example:
            {{
                "queries": [
                    "history of artificial intelligence",
                    "applications of machine learning in healthcare",
                    "ethical considerations in AI development"
                ]
            }}

            JSON response:
            """

            response = self.model.generate(prompt, temperature=self.query_temperature)

            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                # No JSON found, fall back to heuristic approach
                logger.warning("Failed to parse JSON from query generation response")
                return self._generate_heuristic_queries(text)

            json_str = response[json_start:json_end]
            result = json.loads(json_str)

            # Extract queries from result
            queries = result.get("queries", [])

            # If no queries were generated, fall back to heuristic approach
            if not queries:
                logger.warning("No queries generated by model")
                return self._generate_heuristic_queries(text)

            return list(queries)

        except Exception as e:
            logger.warning(f"Error generating queries with model: {str(e)}")
            # Fall back to heuristic approach
            return self._generate_heuristic_queries(text)

    def _generate_heuristic_queries(self, text: str) -> List[str]:
        """Generate search queries using a simple heuristic approach.

        Args:
            text: The text to generate queries for.

        Returns:
            A list of generated queries.
        """
        # Simple heuristic: use the first sentence as a query
        sentences = text.split(".")
        queries = []

        # Add the first sentence if it's not too short
        if sentences and len(sentences[0]) > 10:
            queries.append(sentences[0].strip())

        # Add the entire text as a query if it's not too long
        if len(text) < 200:
            queries.append(text.strip())
        else:
            # Otherwise, add the first 200 characters
            queries.append(text[:200].strip())

        # Add a generic query based on keywords
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "about",
            "of",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 3]

        # Use the most frequent keywords
        if keywords:
            from collections import Counter

            keyword_counts = Counter(keywords)
            top_keywords = [word for word, _ in keyword_counts.most_common(5)]
            queries.append(" ".join(top_keywords))

        return queries

    def _deduplicate_passages(self, passages: List[str]) -> List[str]:
        """Deduplicate passages while preserving order.

        Args:
            passages: The passages to deduplicate.

        Returns:
            A list of deduplicated passages.
        """
        seen: Set[str] = set()
        unique_passages: List[str] = []

        for passage in passages:
            passage_key = passage.strip()
            if passage_key and passage_key not in seen:
                seen.add(passage_key)
                unique_passages.append(passage)

        return unique_passages

    def format_passages(self, passages: List[str]) -> str:
        """Format passages for inclusion in prompts.

        Args:
            passages: The passages to format.

        Returns:
            Formatted passages as a string.
        """
        return "\n\n".join(f"Passage {i+1}:\n{passage}" for i, passage in enumerate(passages))

    def get_retrieval_context(
        self, text: str, custom_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get retrieval context for the given text.

        This method retrieves passages and returns a dictionary with the retrieved
        passages and other context information that can be used by critics.

        Args:
            text: The text to get retrieval context for.
            custom_queries: Optional list of custom queries to use instead of generating them.

        Returns:
            A dictionary with retrieval context information.
        """
        try:
            # Generate queries if not provided
            queries = custom_queries if custom_queries is not None else self._generate_queries(text)

            # Retrieve passages
            passages: List[str] = self.retrieve(text, queries)

            # Format passages
            formatted_passages = self.format_passages(passages)

            # Return context
            return {
                "queries": queries,
                "passages": passages,
                "formatted_passages": formatted_passages,
                "passage_count": len(passages),
            }

        except Exception as e:
            logger.error(f"Error getting retrieval context: {str(e)}")
            return {
                "queries": [],
                "passages": [],
                "formatted_passages": "",
                "passage_count": 0,
                "error": str(e),
            }
