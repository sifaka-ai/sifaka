"""
Base retrieval interface for Sifaka.

This module provides the base Retriever abstract class that defines the interface
for all retrieval implementations in Sifaka.

## Component Lifecycle

1. **Initialization**
   - Configure retrieval sources
   - Set up indexing
   - Initialize resources

2. **Operation**
   - Process queries
   - Retrieve relevant information
   - Format results

3. **Cleanup**
   - Release resources
   - Close connections
   - Clean up temporary data

## Error Handling

1. **Query Processing Errors**
   - Invalid query format
   - Empty queries
   - Malformed queries

2. **Retrieval Errors**
   - Source unavailable
   - Index corruption
   - Timeout errors

3. **Result Processing Errors**
   - Format conversion errors
   - Result truncation issues
   - Metadata extraction failures
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..interfaces.retrieval import Retriever as RetrieverInterface
from ..interfaces.core import Component, Configurable, Identifiable


class Retriever(ABC):
    """
    Abstract base class for retrievers.

    This class defines the interface for all retrieval implementations in Sifaka.
    Retrievers are responsible for retrieving relevant information from various
    sources based on queries.

    This class implements the Retriever interface from sifaka.interfaces.retrieval.

    ## Lifecycle Management

    1. **Initialization**
       - Configure retrieval sources
       - Set up indexing
       - Initialize resources

    2. **Operation**
       - Process queries
       - Retrieve relevant information
       - Format results

    3. **Cleanup**
       - Release resources
       - Close connections
       - Clean up temporary data

    ## Error Handling

    1. **Query Processing Errors**
       - Invalid query format
       - Empty queries
       - Malformed queries

    2. **Retrieval Errors**
       - Source unavailable
       - Index corruption
       - Timeout errors

    3. **Result Processing Errors**
       - Format conversion errors
       - Result truncation issues
       - Metadata extraction failures
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the retriever name.

        Returns:
            The name of the retriever
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the retriever description.

        Returns:
            The description of the retriever
        """
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """
        Get the retriever configuration.

        Returns:
            The configuration of the retriever
        """
        pass

    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the retriever configuration.

        Args:
            config: The new configuration object

        Raises:
            ValueError: If the configuration is invalid
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the retriever.

        This method initializes any resources needed by the retriever.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up the retriever.

        This method releases any resources held by the retriever.

        Raises:
            RuntimeError: If cleanup fails
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> str:
        """
        Retrieve information based on a query.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            Retrieved information as a string

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If retrieval fails
        """
        pass
