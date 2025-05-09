"""
Retrieval interfaces for Sifaka.

This module defines the interfaces for retrieval components in the Sifaka framework.
These interfaces establish a common contract for retrieval behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Retriever**: Base interface for all retrievers
   - **DocumentStore**: Interface for document stores
   - **IndexManager**: Interface for index managers
   - **QueryProcessor**: Interface for query processors

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.interfaces.core import Component, Configurable, Identifiable, Stateful

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
QueryType = TypeVar("QueryType", contravariant=True)
DocumentType = TypeVar("DocumentType")
ResultType = TypeVar("ResultType", covariant=True)


@runtime_checkable
class DocumentStore(Protocol[DocumentType]):
    """
    Interface for document stores.

    This interface defines the contract for components that store and retrieve
    documents. It ensures that document stores can add, get, update, and delete
    documents.

    ## Lifecycle

    1. **Initialization**: Set up document store resources
    2. **Document Management**: Add, get, update, and delete documents
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide methods to add, get, update, and delete documents
    - Handle document storage and retrieval efficiently
    """

    @abstractmethod
    def add_document(self, document: DocumentType) -> str:
        """
        Add a document to the store.

        Args:
            document: The document to add

        Returns:
            The document ID

        Raises:
            ValueError: If the document is invalid
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> DocumentType:
        """
        Get a document from the store.

        Args:
            document_id: The ID of the document to get

        Returns:
            The document

        Raises:
            ValueError: If the document ID is invalid
            KeyError: If the document is not found
        """
        pass

    @abstractmethod
    def update_document(self, document_id: str, document: DocumentType) -> None:
        """
        Update a document in the store.

        Args:
            document_id: The ID of the document to update
            document: The updated document

        Raises:
            ValueError: If the document ID or document is invalid
            KeyError: If the document is not found
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the store.

        Args:
            document_id: The ID of the document to delete

        Raises:
            ValueError: If the document ID is invalid
            KeyError: If the document is not found
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the store.

        Returns:
            A list of document IDs
        """
        pass


@runtime_checkable
class IndexManager(Protocol):
    """
    Interface for index managers.

    This interface defines the contract for components that manage indexes for
    document retrieval. It ensures that index managers can create, update, and
    search indexes.

    ## Lifecycle

    1. **Initialization**: Set up index management resources
    2. **Index Management**: Create, update, and search indexes
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide methods to create, update, and search indexes
    - Handle index management efficiently
    """

    @abstractmethod
    def create_index(self, name: str, **kwargs: Any) -> None:
        """
        Create an index.

        Args:
            name: The name of the index
            **kwargs: Additional index creation parameters

        Raises:
            ValueError: If the index name or parameters are invalid
            RuntimeError: If index creation fails
        """
        pass

    @abstractmethod
    def update_index(self, name: str, **kwargs: Any) -> None:
        """
        Update an index.

        Args:
            name: The name of the index
            **kwargs: Additional index update parameters

        Raises:
            ValueError: If the index name or parameters are invalid
            KeyError: If the index is not found
            RuntimeError: If index update fails
        """
        pass

    @abstractmethod
    def delete_index(self, name: str) -> None:
        """
        Delete an index.

        Args:
            name: The name of the index

        Raises:
            ValueError: If the index name is invalid
            KeyError: If the index is not found
            RuntimeError: If index deletion fails
        """
        pass

    @abstractmethod
    def search_index(self, name: str, query: Any, **kwargs: Any) -> List[Any]:
        """
        Search an index.

        Args:
            name: The name of the index
            query: The query to search for
            **kwargs: Additional search parameters

        Returns:
            A list of search results

        Raises:
            ValueError: If the index name, query, or parameters are invalid
            KeyError: If the index is not found
            RuntimeError: If index search fails
        """
        pass


@runtime_checkable
class QueryProcessor(Protocol[QueryType, ResultType]):
    """
    Interface for query processors.

    This interface defines the contract for components that process queries for
    document retrieval. It ensures that query processors can process queries and
    return results.

    ## Lifecycle

    1. **Initialization**: Set up query processing resources
    2. **Query Processing**: Process queries and return results
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a process method to process queries
    - Return standardized results
    """

    @abstractmethod
    def process(self, query: QueryType, **kwargs: Any) -> ResultType:
        """
        Process a query.

        Args:
            query: The query to process
            **kwargs: Additional query processing parameters

        Returns:
            A query result

        Raises:
            ValueError: If the query or parameters are invalid
            RuntimeError: If query processing fails
        """
        pass


@runtime_checkable
class Retriever(Identifiable, Configurable[ConfigType], Protocol[QueryType, ResultType]):
    """
    Interface for retrievers.

    This interface defines the contract for components that retrieve information
    based on queries. It ensures that retrievers can retrieve information and
    return standardized results.

    ## Lifecycle

    1. **Initialization**: Set up retrieval resources and configuration
    2. **Retrieval**: Retrieve information based on queries
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage retrieval configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a retrieve method to retrieve information
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the retrieval configuration
    - Provide an update_config method to update the retrieval configuration
    """

    @abstractmethod
    def retrieve(self, query: QueryType, **kwargs: Any) -> ResultType:
        """
        Retrieve information based on a query.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            A retrieval result

        Raises:
            ValueError: If the query or parameters are invalid
            RuntimeError: If retrieval fails
        """
        pass


@runtime_checkable
class AsyncRetriever(Protocol[QueryType, ResultType]):
    """
    Interface for asynchronous retrievers.

    This interface defines the contract for components that retrieve information
    asynchronously based on queries. It ensures that retrievers can retrieve
    information asynchronously and return standardized results.

    ## Lifecycle

    1. **Initialization**: Set up retrieval resources and configuration
    2. **Retrieval**: Retrieve information asynchronously based on queries
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage retrieval configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async retrieve method to retrieve information
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the retrieval configuration
    - Provide an update_config method to update the retrieval configuration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the retriever name.

        Returns:
            The retriever name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the retriever description.

        Returns:
            The retriever description
        """
        pass

    @abstractmethod
    async def retrieve(self, query: QueryType, **kwargs: Any) -> ResultType:
        """
        Retrieve information asynchronously based on a query.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            A retrieval result

        Raises:
            ValueError: If the query or parameters are invalid
            RuntimeError: If retrieval fails
        """
        pass
