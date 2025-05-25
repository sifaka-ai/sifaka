"""Protocols for storage components.

This module defines the protocols (interfaces) used by the storage system
to avoid circular imports and provide clear type definitions.
"""

from typing import List, Protocol

from sifaka.core.thought import Thought


class Retriever(Protocol):
    """Protocol for retriever implementations.
    
    This protocol defines the interface that all retrievers must implement
    to be compatible with the cached retriever wrapper.
    """
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents for a query.
        
        Args:
            query: Query string for document retrieval.
            
        Returns:
            List of retrieved document texts.
        """
        ...
    
    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.
        
        Args:
            thought: The thought to retrieve context for.
            is_pre_generation: Whether this is pre-generation retrieval.
            
        Returns:
            The thought with retrieved documents added.
        """
        ...
