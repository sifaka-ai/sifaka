"""
Core retrieval implementation for Sifaka.

This module provides the core implementation of the retrieval functionality
in the Sifaka framework. It includes the RetrieverCore class, which serves
as the foundation for all retriever implementations.

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

import time
from typing import Any, Dict, List, Optional, Union, cast

from .config import RetrieverConfig
from .result import RetrievalResult, RetrievedDocument, DocumentMetadata, StringRetrievalResult
from .interfaces.retriever import Retriever, QueryProcessor
from .managers.query import QueryManager


class RetrieverCore:
    """
    Core implementation of the retriever functionality.

    This class provides the core implementation of the retriever functionality
    in the Sifaka framework. It serves as the foundation for all retriever
    implementations and handles common tasks like query processing, result
    formatting, and error handling.

    ## Lifecycle Management

    1. **Initialization**
       - Configure retrieval sources
       - Set up query processing
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

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        query_processor: Optional[QueryProcessor] = None,
    ):
        """
        Initialize the retriever core.

        Args:
            config: The retriever configuration
            query_processor: The query processor to use

        Raises:
            ValueError: If the configuration is invalid
        """
        self.config = config or RetrieverConfig()
        self.query_processor = query_processor or QueryManager(self.config.query_processing)
        self._name = "RetrieverCore"
        self._description = "Core retriever implementation for Sifaka"

    @property
    def name(self) -> str:
        """
        Get the retriever name.

        Returns:
            The name of the retriever
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Get the retriever description.

        Returns:
            The description of the retriever
        """
        return self._description

    @property
    def config(self) -> RetrieverConfig:
        """
        Get the retriever configuration.

        Returns:
            The configuration of the retriever
        """
        return self._config

    @config.setter
    def config(self, config: RetrieverConfig) -> None:
        """
        Set the retriever configuration.

        Args:
            config: The new configuration

        Raises:
            ValueError: If the configuration is invalid
        """
        if not isinstance(config, RetrieverConfig):
            raise ValueError("Config must be an instance of RetrieverConfig")
        self._config = config

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the retriever configuration.

        Args:
            config: The new configuration object

        Raises:
            ValueError: If the configuration is invalid
        """
        # Create a new config with the updated values
        new_config = self.config.model_copy(update=config)
        self.config = new_config

    def initialize(self) -> None:
        """
        Initialize the retriever.

        This method initializes any resources needed by the retriever.

        Raises:
            RuntimeError: If initialization fails
        """
        # This is a base implementation that does nothing
        pass

    def cleanup(self) -> None:
        """
        Clean up the retriever.

        This method releases any resources held by the retriever.

        Raises:
            RuntimeError: If cleanup fails
        """
        # This is a base implementation that does nothing
        pass

    def process_query(self, query: str) -> str:
        """
        Process a query before retrieval.

        Args:
            query: The query to process

        Returns:
            The processed query

        Raises:
            ValueError: If the query is empty or invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        return self.query_processor.process_query(query)

    def create_result(
        self,
        query: str,
        processed_query: str,
        documents: List[Dict[str, Any]],
        execution_time_ms: Optional[float] = None,
    ) -> StringRetrievalResult:
        """
        Create a retrieval result from raw documents.

        Args:
            query: The original query
            processed_query: The processed query
            documents: The raw documents (list of dicts with 'content', 'metadata', and 'score')
            execution_time_ms: The execution time in milliseconds

        Returns:
            A StringRetrievalResult object
        """
        retrieved_docs = []
        for doc in documents:
            content = doc.get("content", "")
            score = doc.get("score")
            
            # Extract metadata
            metadata_dict = doc.get("metadata", {})
            if not isinstance(metadata_dict, dict):
                metadata_dict = {"document_id": str(metadata_dict)}
            
            # Ensure document_id is present
            if "document_id" not in metadata_dict:
                metadata_dict["document_id"] = f"doc_{len(retrieved_docs)}"
                
            metadata = DocumentMetadata(**metadata_dict)
            
            retrieved_docs.append(
                RetrievedDocument(
                    content=content,
                    metadata=metadata,
                    score=score,
                )
            )
        
        return StringRetrievalResult(
            documents=retrieved_docs,
            query=query,
            processed_query=processed_query,
            total_results=len(retrieved_docs),
            execution_time_ms=execution_time_ms,
        )

    def retrieve(self, query: str, **kwargs: Any) -> StringRetrievalResult:
        """
        Retrieve information based on a query.

        This is a base implementation that should be overridden by subclasses.
        It processes the query and returns an empty result.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            A StringRetrievalResult object

        Raises:
            ValueError: If the query is empty or invalid
            RuntimeError: If retrieval fails
        """
        start_time = time.time()
        
        # Process the query
        processed_query = self.process_query(query)
        
        # This base implementation returns an empty result
        # Subclasses should override this method to implement actual retrieval
        
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        return self.create_result(
            query=query,
            processed_query=processed_query,
            documents=[],
            execution_time_ms=execution_time_ms,
        )
