"""Embedding generation utilities for semantic search and RAG.

This module provides embedding generation capabilities using various providers,
with a focus on OpenAI's text-embedding models for vector storage and retrieval.
"""

import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

from .exceptions import EmbeddingError
from .llm_client import Provider


class EmbeddingModel(str, Enum):
    """Available embedding models."""

    # OpenAI models
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"

    # Anthropic (when available)
    # ANTHROPIC_VOYAGE = "voyage-01"

    # Google (when available)
    # GOOGLE_GECKO = "textembedding-gecko"


class EmbeddingGenerator:
    """Generate embeddings for text using various providers.

    Handles embedding generation with caching, batching, and error handling.
    Currently supports OpenAI embeddings with plans for other providers.

    Features:
    - Multiple model support
    - Batch processing for efficiency
    - In-memory caching to reduce API calls
    - Dimension configuration for storage optimization

    Example:
        >>> generator = EmbeddingGenerator()
        >>> embedding = await generator.embed("Hello world")
        >>> print(f"Embedding dimension: {len(embedding)}")
    """

    def __init__(
        self,
        model: Union[str, EmbeddingModel] = EmbeddingModel.OPENAI_3_SMALL,
        dimensions: Optional[int] = None,
        provider: Optional[Provider] = None,
        api_key: Optional[str] = None,
        cache_size: int = 1000,
    ):
        """Initialize embedding generator.

        Args:
            model: Embedding model to use
            dimensions: Output dimensions (for models that support it)
            provider: Provider override (defaults to OpenAI)
            api_key: API key override
            cache_size: Maximum number of embeddings to cache
        """
        self.model = model
        self.dimensions = dimensions
        self.provider = provider or Provider.OPENAI
        self.api_key = api_key
        self._cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size

        # Model-specific configurations
        self._model_dims = {
            EmbeddingModel.OPENAI_ADA_002: 1536,
            EmbeddingModel.OPENAI_3_SMALL: 1536,  # Can be reduced
            EmbeddingModel.OPENAI_3_LARGE: 3072,  # Can be reduced
        }

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(
            f"{self.model}:{text}".encode(), usedforsecurity=False
        ).hexdigest()

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Check cache first
        results: List[Optional[List[float]]] = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self._generate_embeddings(uncached_texts)

            # Update results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding

                # Cache with size limit
                cache_key = self._get_cache_key(texts[idx])
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = embedding

        # All None values should be replaced by now
        final_results: List[List[float]] = []
        for r in results:
            if r is not None:
                final_results.append(r)
            else:
                raise EmbeddingError("Failed to generate embedding for text")
        return final_results

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the configured provider.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        if self.provider == Provider.OPENAI:
            return await self._generate_openai_embeddings(texts)
        else:
            raise EmbeddingError(
                f"Provider {self.provider} not yet supported for embeddings"
            )

    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        import os

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OpenAI API key required for embeddings")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Prepare request
        data: Dict[str, Any] = {"input": texts, "model": self.model}

        # Add dimensions if specified and supported
        if self.dimensions and self.model in [
            EmbeddingModel.OPENAI_3_SMALL,
            EmbeddingModel.OPENAI_3_LARGE,
        ]:
            data["dimensions"] = self.dimensions

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )
                response.raise_for_status()

                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]

                return embeddings

            except httpx.HTTPStatusError as e:
                raise EmbeddingError(
                    f"OpenAI API error: {e.response.status_code} - {e.response.text}"
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to generate embeddings: {e!s}")

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a, b) / (a_norm * b_norm))

    def get_model_dimensions(self) -> int:
        """Get the output dimensions for the current model.

        Returns:
            Number of dimensions in the embedding vector
        """
        if self.dimensions:
            return self.dimensions
        if isinstance(self.model, EmbeddingModel):
            return self._model_dims.get(self.model, 1536)
        else:
            # For string models, default to 1536
            return 1536


# Global instance for convenience
_default_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    model: Optional[Union[str, EmbeddingModel]] = None, **kwargs: Any
) -> EmbeddingGenerator:
    """Get or create a default embedding generator.

    Args:
        model: Model override
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        Configured embedding generator
    """
    global _default_generator

    if model or kwargs or _default_generator is None:
        _default_generator = EmbeddingGenerator(
            model=model or EmbeddingModel.OPENAI_3_SMALL, **kwargs
        )

    return _default_generator
