"""
Base classes and interfaces for persistence in Sifaka.

This module defines the abstract base classes and common interfaces for
persistence operations in the Sifaka framework.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

from sifaka.core.thought import Thought


class PersistenceError(Exception):
    """Base exception for persistence operations.
    
    This exception is raised when persistence operations fail,
    such as storage errors, retrieval failures, or data corruption.
    
    Attributes:
        message: Human-readable error message
        operation: The operation that failed
        storage_type: The type of storage backend
        metadata: Additional error context
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        storage_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.operation = operation
        self.storage_type = storage_type
        self.metadata = metadata or {}
        
        # Build full error message
        full_message = message
        if storage_type:
            full_message = f"[{storage_type}] {full_message}"
        if operation:
            full_message = f"{full_message} (during {operation})"
            
        super().__init__(full_message)


class ThoughtQuery(BaseModel):
    """Query parameters for searching thoughts.
    
    This class defines the parameters that can be used to query
    stored thoughts, including filters, sorting, and pagination.
    
    Attributes:
        thought_ids: Specific thought IDs to retrieve
        chain_ids: Filter by chain IDs
        parent_ids: Filter by parent thought IDs
        prompts: Filter by prompt text (partial match)
        text_contains: Filter by generated text content
        min_iteration: Minimum iteration number
        max_iteration: Maximum iteration number
        start_date: Filter thoughts created after this date
        end_date: Filter thoughts created before this date
        has_validation_results: Filter thoughts with validation results
        has_critic_feedback: Filter thoughts with critic feedback
        has_context: Filter thoughts with retrieval context
        limit: Maximum number of results to return
        offset: Number of results to skip
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
    """
    
    # ID filters
    thought_ids: Optional[List[str]] = None
    chain_ids: Optional[List[str]] = None
    parent_ids: Optional[List[str]] = None
    
    # Content filters
    prompts: Optional[List[str]] = None
    text_contains: Optional[str] = None
    
    # Iteration filters
    min_iteration: Optional[int] = None
    max_iteration: Optional[int] = None
    
    # Date filters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Feature filters
    has_validation_results: Optional[bool] = None
    has_critic_feedback: Optional[bool] = None
    has_context: Optional[bool] = None
    
    # Pagination and sorting
    limit: Optional[int] = Field(default=100, ge=1, le=10000)
    offset: Optional[int] = Field(default=0, ge=0)
    sort_by: Optional[str] = Field(default="timestamp")
    sort_order: Optional[str] = Field(default="desc", pattern="^(asc|desc)$")


class ThoughtQueryResult(BaseModel):
    """Result of a thought query operation.
    
    This class contains the results of querying thoughts,
    including the matching thoughts and metadata about the query.
    
    Attributes:
        thoughts: List of thoughts matching the query
        total_count: Total number of thoughts matching the query (before pagination)
        query: The original query parameters
        execution_time_ms: Time taken to execute the query in milliseconds
        metadata: Additional metadata about the query execution
    """
    
    thoughts: List[Thought]
    total_count: int
    query: ThoughtQuery
    execution_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThoughtStorage(ABC):
    """Abstract base class for thought storage backends.
    
    This class defines the interface that all thought storage backends
    must implement. It provides methods for storing, retrieving, and
    querying thoughts.
    """
    
    @abstractmethod
    def save_thought(self, thought: Thought) -> None:
        """Save a thought to storage.
        
        Args:
            thought: The thought to save
            
        Raises:
            PersistenceError: If the save operation fails
        """
        pass
    
    @abstractmethod
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Retrieve a thought by ID.
        
        Args:
            thought_id: The ID of the thought to retrieve
            
        Returns:
            The thought if found, None otherwise
            
        Raises:
            PersistenceError: If the retrieval operation fails
        """
        pass
    
    @abstractmethod
    def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought from storage.
        
        Args:
            thought_id: The ID of the thought to delete
            
        Returns:
            True if the thought was deleted, False if it didn't exist
            
        Raises:
            PersistenceError: If the delete operation fails
        """
        pass
    
    @abstractmethod
    def query_thoughts(self, query: Optional[ThoughtQuery] = None) -> ThoughtQueryResult:
        """Query thoughts based on criteria.
        
        Args:
            query: Query parameters, or None for all thoughts
            
        Returns:
            Query result containing matching thoughts and metadata
            
        Raises:
            PersistenceError: If the query operation fails
        """
        pass
    
    @abstractmethod
    def get_thought_history(self, thought_id: str) -> List[Thought]:
        """Get the complete history of a thought.
        
        Args:
            thought_id: The ID of the thought
            
        Returns:
            List of thoughts in the history chain, ordered by iteration
            
        Raises:
            PersistenceError: If the operation fails
        """
        pass
    
    @abstractmethod
    def get_chain_thoughts(self, chain_id: str) -> List[Thought]:
        """Get all thoughts in a chain.
        
        Args:
            chain_id: The ID of the chain
            
        Returns:
            List of thoughts in the chain, ordered by timestamp
            
        Raises:
            PersistenceError: If the operation fails
        """
        pass
    
    @abstractmethod
    def count_thoughts(self, query: Optional[ThoughtQuery] = None) -> int:
        """Count thoughts matching query criteria.
        
        Args:
            query: Query parameters, or None for all thoughts
            
        Returns:
            Number of thoughts matching the criteria
            
        Raises:
            PersistenceError: If the count operation fails
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the storage backend.
        
        Returns:
            Dictionary containing health status and metrics
            
        Raises:
            PersistenceError: If the health check fails
        """
        pass
