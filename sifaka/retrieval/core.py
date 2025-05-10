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
from typing import Any, Dict, List, Optional

from sifaka.core.base import BaseComponent
from sifaka.utils.errors import RetrievalError, InputError, handle_error
from sifaka.utils.logging import get_logger

from .config import RetrieverConfig
from .result import RetrievalResult, RetrievedDocument, DocumentMetadata, StringRetrievalResult
from sifaka.interfaces.retrieval import Retriever, QueryProcessor
from .managers.query import QueryManager

logger = get_logger(__name__)


class RetrieverCore(BaseComponent):
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

    # State management is handled by BaseComponent

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        query_processor: Optional[QueryProcessor] = None,
        name: str = "RetrieverCore",
        description: str = "Core retriever implementation for Sifaka",
    ):
        """
        Initialize the retriever core.

        Args:
            config: The retriever configuration
            query_processor: The query processor to use
            name: Name of the retriever
            description: Description of the retriever

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        super().__init__(name=name, description=description)

        # Initialize state
        self._initialize_state(config, query_processor)

        # Set metadata
        self._state_manager.set_metadata("component_type", "retriever")
        self._state_manager.set_metadata("creation_time", time.time())

    def _initialize_state(
        self,
        config: Optional[RetrieverConfig] = None,
        query_processor: Optional[QueryProcessor] = None,
    ) -> None:
        """
        Initialize the retriever state.

        Args:
            config: The retriever configuration
            query_processor: The query processor to use
        """
        # Call parent's _initialize_state first
        super()._initialize_state()

        config = config or RetrieverConfig()
        query_processor = query_processor or QueryManager(config.query_processing)

        self._state_manager.update("config", config)
        self._state_manager.update("query_processor", query_processor)
        self._state_manager.update("initialized", False)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})
        self._state_manager.update("last_query", None)
        self._state_manager.update("last_result", None)

    @property
    def config(self) -> RetrieverConfig:
        """
        Get the retriever configuration.

        Returns:
            The configuration of the retriever
        """
        return self._state_manager.get("config")

    @config.setter
    def config(self, config: RetrieverConfig) -> None:
        """
        Set the retriever configuration.

        Args:
            config: The new configuration

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        if not isinstance(config, RetrieverConfig):
            raise RetrievalError(
                "Config must be an instance of RetrieverConfig",
                metadata={"config_type": type(config).__name__},
            )
        self._state_manager.update("config", config)

    @property
    def query_processor(self) -> QueryProcessor:
        """
        Get the query processor.

        Returns:
            The query processor
        """
        return self._state_manager.get("query_processor")

    @query_processor.setter
    def query_processor(self, processor: QueryProcessor) -> None:
        """
        Set the query processor.

        Args:
            processor: The new query processor
        """
        self._state_manager.update("query_processor", processor)

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the retriever configuration.

        Args:
            config: The new configuration object

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        try:
            # Create a new config with the updated values
            current_config = self.config
            new_config = current_config.model_copy(update=config)
            self.config = new_config
        except Exception as e:
            raise RetrievalError(
                f"Failed to update configuration: {str(e)}", metadata={"config_update": config}
            )

    def initialize(self) -> None:
        """
        Initialize the retriever.

        This method initializes any resources needed by the retriever.

        Raises:
            RetrievalError: If initialization fails
        """
        if self._state_manager.get("initialized", False):
            logger.debug(f"Retriever {self.name} already initialized")
            return

        try:
            # Perform initialization tasks
            logger.debug(f"Initializing retriever {self.name}")
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("initialization_time", time.time())
        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}", metadata=error_info)

    def cleanup(self) -> None:
        """
        Clean up the retriever.

        This method releases any resources held by the retriever.

        Raises:
            RetrievalError: If cleanup fails
        """
        if not self._state_manager.get("initialized", False):
            logger.debug(f"Retriever {self.name} not initialized, nothing to clean up")
            return

        try:
            # Perform cleanup tasks
            logger.debug(f"Cleaning up retriever {self.name}")
            self._state_manager.update("initialized", False)
            self._state_manager.update("result_cache", {})
            self._state_manager.set_metadata("cleanup_time", time.time())
        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to clean up retriever: {str(e)}", metadata=error_info)

    def process_query(self, query: str) -> str:
        """
        Process a query before retrieval.

        Args:
            query: The query to process

        Returns:
            The processed query

        Raises:
            InputError: If the query is empty or invalid
        """
        if not query or not isinstance(query, str):
            raise InputError(
                "Query must be a non-empty string",
                metadata={
                    "query_type": type(query).__name__,
                    "query_length": len(str(query)) if query else 0,
                },
            )

        try:
            return self.query_processor.process_query(query)
        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to process query: {str(e)}", metadata=error_info)

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

        Raises:
            RetrievalError: If result creation fails
        """
        try:
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

            result = StringRetrievalResult(
                documents=retrieved_docs,
                query=query,
                processed_query=processed_query,
                total_results=len(retrieved_docs),
                execution_time_ms=execution_time_ms,
            )

            # Store the result in state
            self._state_manager.update("last_result", result)

            return result

        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to create result: {str(e)}", metadata=error_info)

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
            InputError: If the query is empty or invalid
            RetrievalError: If retrieval fails
        """
        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Store the query in state
        self._state_manager.update("last_query", query)

        # Check if initialized
        if not self._state_manager.get("initialized", False):
            logger.debug(f"Initializing retriever {self.name} on first use")
            self.initialize()

        # Record start time
        start_time = time.time()

        try:
            # Process the query
            processed_query = self.process_query(query)

            # This base implementation returns an empty result
            # Subclasses should override this method to implement actual retrieval

            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            # Update execution time statistics
            self._update_execution_stats(execution_time_ms)

            result = self.create_result(
                query=query,
                processed_query=processed_query,
                documents=[],
                execution_time_ms=execution_time_ms,
            )

            return result

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Log error
            logger.error(f"Retrieval error in {self.name}: {str(e)}")

            # Re-raise as RetrievalError
            if not isinstance(e, (InputError, RetrievalError)):
                raise RetrievalError(
                    f"Retrieval failed: {str(e)}",
                    metadata={"query": query, "error_type": type(e).__name__},
                )
            raise

    def _update_execution_stats(self, execution_time_ms: float) -> None:
        """
        Update execution statistics.

        Args:
            execution_time_ms: The execution time in milliseconds
        """
        # Update average execution time
        avg_time = self._state_manager.get_metadata("avg_execution_time_ms", 0)
        count = self._state_manager.get("execution_count", 1)
        new_avg = ((avg_time * (count - 1)) + execution_time_ms) / count
        self._state_manager.set_metadata("avg_execution_time_ms", new_avg)

        # Update max execution time
        max_time = self._state_manager.get_metadata("max_execution_time_ms", 0)
        if execution_time_ms > max_time:
            self._state_manager.set_metadata("max_execution_time_ms", execution_time_ms)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retriever usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "name": self.name,
            "execution_count": self._state_manager.get("execution_count", 0),
            "avg_execution_time_ms": self._state_manager.get_metadata("avg_execution_time_ms", 0),
            "max_execution_time_ms": self._state_manager.get_metadata("max_execution_time_ms", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
            "initialized": self._state_manager.get("initialized", False),
            "last_query": self._state_manager.get("last_query"),
        }

    def clear_cache(self) -> None:
        """
        Clear the retriever cache.
        """
        self._state_manager.update("result_cache", {})
        logger.debug(f"Cleared cache for retriever {self.name}")

    def process(self, input_data: Any, **kwargs: Any) -> StringRetrievalResult:
        """
        Process input data.

        This method is required by the BaseComponent abstract class.

        Args:
            input_data: The input data to process (query string)
            **kwargs: Additional processing parameters

        Returns:
            A StringRetrievalResult object

        Raises:
            InputError: If the input is not a string
            RetrievalError: If retrieval fails
        """
        if not isinstance(input_data, str):
            raise InputError(
                "Input data must be a string", metadata={"input_type": type(input_data).__name__}
            )

        return self.retrieve(input_data, **kwargs)
