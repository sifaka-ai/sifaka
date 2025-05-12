from typing import Any, List
"""
Managers for retrieval components.

This package provides managers for different aspects of retrieval:
- QueryManager: Manages query processing and expansion
- IndexManager: Manages document indexing and searching
- DocumentManager: Manages document storage and retrieval
"""
from .query import QueryManager
__all__: List[Any] = ['QueryManager']
