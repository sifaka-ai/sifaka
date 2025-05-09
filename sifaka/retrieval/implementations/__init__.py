"""
Retriever implementations for Sifaka.

This package provides concrete implementations of retrievers for the Sifaka framework:
- SimpleRetriever: A basic retriever for in-memory document collections
"""

from .simple import SimpleRetriever

__all__ = [
    "SimpleRetriever",
]
