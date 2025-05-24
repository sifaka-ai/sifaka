"""
Cache key generation utilities for Sifaka memory system.

This module provides utilities for generating consistent, deterministic cache keys
from Thought objects and other data structures. The keys are designed to be:
- Deterministic: Same input always produces same key
- Collision-resistant: Different inputs produce different keys
- Efficient: Fast generation and comparison
- Human-readable: Include meaningful prefixes for debugging

The cache key system supports multiple strategies:
- Hash-based: Fast exact matching using SHA-256 hashes
- Content-based: Keys based on prompt and context content
- Semantic-based: Keys that account for semantic similarity
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought, Document
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class CacheKeyGenerator:
    """Utilities for generating cache keys from thoughts and other data structures.

    This class provides static methods for generating various types of cache keys
    used throughout the Sifaka memory system. All keys are deterministic and
    collision-resistant.
    """

    @staticmethod
    def thought_key(thought: Thought, include_context: bool = True) -> str:
        """Generate a cache key for a thought.

        Creates a deterministic cache key based on the thought's content.
        The key includes the prompt, system prompt, and optionally the context.

        Args:
            thought: The thought to generate a key for
            include_context: Whether to include retrieval context in the key

        Returns:
            A cache key string in format: "thought:{hash}"
        """
        key_data = {
            "prompt": thought.prompt,
            "system_prompt": thought.system_prompt,
            "version": thought.version,
        }

        if include_context:
            if thought.pre_generation_context:
                key_data["pre_context"] = [
                    {"text": doc.text, "metadata": doc.metadata}
                    for doc in thought.pre_generation_context
                ]

            if thought.post_generation_context:
                key_data["post_context"] = [
                    {"text": doc.text, "metadata": doc.metadata}
                    for doc in thought.post_generation_context
                ]

        return CacheKeyGenerator._generate_hash_key("thought", key_data)

    @staticmethod
    def prompt_key(prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a cache key for a prompt.

        Creates a simple cache key based on the prompt text and optional system prompt.
        Useful for caching model responses to specific prompts.

        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt

        Returns:
            A cache key string in format: "prompt:{hash}"
        """
        key_data = {"prompt": prompt}
        if system_prompt:
            key_data["system_prompt"] = system_prompt

        return CacheKeyGenerator._generate_hash_key("prompt", key_data)

    @staticmethod
    def context_key(documents: List[Document]) -> str:
        """Generate a cache key for a list of documents.

        Creates a cache key based on the content and metadata of documents.
        Useful for caching retrieval results.

        Args:
            documents: List of documents to generate key for

        Returns:
            A cache key string in format: "context:{hash}"
        """
        key_data = [
            {
                "text": doc.text,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in documents
        ]

        return CacheKeyGenerator._generate_hash_key("context", key_data)

    @staticmethod
    def chain_state_key(
        thought: Thought, step: str, iteration: int, chain_id: Optional[str] = None
    ) -> str:
        """Generate a cache key for chain execution state.

        Creates a cache key for storing chain checkpoints and execution state.

        Args:
            thought: The current thought
            step: Current execution step (e.g., "generation", "validation", "criticism")
            iteration: Current iteration number
            chain_id: Optional chain identifier

        Returns:
            A cache key string in format: "chain_state:{hash}"
        """
        key_data = {
            "thought_id": thought.id,
            "step": step,
            "iteration": iteration,
            "prompt_hash": CacheKeyGenerator._quick_hash(thought.prompt),
        }

        if chain_id:
            key_data["chain_id"] = chain_id

        return CacheKeyGenerator._generate_hash_key("chain_state", key_data)

    @staticmethod
    def validation_key(thought: Thought, validator_name: str) -> str:
        """Generate a cache key for validation results.

        Creates a cache key for caching validation results for specific
        thought-validator combinations.

        Args:
            thought: The thought being validated
            validator_name: Name of the validator

        Returns:
            A cache key string in format: "validation:{hash}"
        """
        key_data = {
            "thought_key": CacheKeyGenerator.thought_key(thought),
            "validator": validator_name,
        }

        return CacheKeyGenerator._generate_hash_key("validation", key_data)

    @staticmethod
    def criticism_key(thought: Thought, critic_name: str) -> str:
        """Generate a cache key for critic feedback.

        Creates a cache key for caching critic feedback for specific
        thought-critic combinations.

        Args:
            thought: The thought being critiqued
            critic_name: Name of the critic

        Returns:
            A cache key string in format: "criticism:{hash}"
        """
        key_data = {
            "thought_key": CacheKeyGenerator.thought_key(thought),
            "critic": critic_name,
        }

        return CacheKeyGenerator._generate_hash_key("criticism", key_data)

    @staticmethod
    def _generate_hash_key(prefix: str, data: Any) -> str:
        """Generate a hash-based cache key with prefix.

        Args:
            prefix: Key prefix for categorization
            data: Data to hash

        Returns:
            Cache key in format: "{prefix}:{hash}"
        """
        # Convert data to deterministic JSON string
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        # Return prefixed key (use first 16 chars of hash for readability)
        return f"{prefix}:{hash_hex[:16]}"

    @staticmethod
    def _quick_hash(text: str) -> str:
        """Generate a quick hash for text content.

        Args:
            text: Text to hash

        Returns:
            Short hash string (8 characters)
        """
        hash_obj = hashlib.md5(text.encode("utf-8"))
        return hash_obj.hexdigest()[:8]

    @staticmethod
    def parse_cache_key(cache_key: str) -> Dict[str, str]:
        """Parse a cache key to extract prefix and hash.

        Args:
            cache_key: Cache key to parse

        Returns:
            Dictionary with 'prefix' and 'hash' keys

        Raises:
            ValueError: If cache key format is invalid
        """
        try:
            prefix, hash_part = cache_key.split(":", 1)
            return {"prefix": prefix, "hash": hash_part}
        except ValueError:
            raise ValueError(f"Invalid cache key format: {cache_key}")

    @staticmethod
    def is_valid_cache_key(cache_key: str) -> bool:
        """Check if a string is a valid cache key.

        Args:
            cache_key: String to validate

        Returns:
            True if valid cache key format, False otherwise
        """
        try:
            parsed = CacheKeyGenerator.parse_cache_key(cache_key)
            return len(parsed["hash"]) >= 8  # Minimum hash length
        except ValueError:
            return False
