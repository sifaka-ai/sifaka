"""
Core retrieval implementation for Sifaka.

This module provides the core implementation of the retrieval functionality
in the Sifaka framework. It includes the RetrieverCore class, which serves
as the foundation for all retriever implementations.

## Overview

The retrieval core provides a standardized foundation for implementing retrievers
in the Sifaka framework. It handles common tasks like query processing, result
formatting, and error handling, allowing specific retriever implementations to
focus on their unique retrieval logic.

## Components

- **RetrieverCore**: Base class for all retriever implementations
- **safely_execute_retrieval**: Utility function for safe retrieval execution
- **create_result**: Utility function for creating standardized retrieval results

## Usage Examples

```python
from sifaka.retrieval.core import RetrieverCore
from sifaka.utils.config and config and config and config and config.retrieval import RetrieverConfig

class MyRetriever(RetrieverCore):
    def __init__(self, config=None, name="MyRetriever", description="Custom retriever"):
        super().__init__(config=config, name=name, description=description)
        # Initialize custom resources

    def retrieve(self, query, **kwargs):
        # Call parent method to handle state tracking
        super().retrieve(query, **kwargs)

        # Process the query
        processed_query = (self and self.process_query(query)

        # Implement custom retrieval logic
        documents = (self and self._retrieve_documents(processed_query)

        # Create and return a standardized result
        return (self and self.create_result(
            query=query,
            processed_query=processed_query,
            documents=documents,
            execution_time_ms=100.0,
        )
```

## Error Handling

The retrieval core provides standardized error handling through:

1. **Query Processing Errors**
   - Invalid query format
   - Empty queries
   - Malformed queries

2. **Retrieval Errors**
   - Source unavailable
   - Index corruption
   - Timeout errors

## Configuration

The retrieval core uses the RetrieverConfig class for configuration:

1. **Retriever Type**: The type of retriever to use
2. **Ranking**: Configuration for ranking strategies
3. **Index**: Configuration for index management
4. **Query Processing**: Configuration for query processing
"""
import time
from typing import Any, Dict, List, Optional
from sifaka.core.base import BaseComponent
from sifaka.utils.errors.component import RetrievalError
from sifaka.utils.errors.base import InputError
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger
from sifaka.utils.common import record_error


def safely_execute_retrieval(operation: callable, retriever_name: str,
    component_name: str, additional_metadata: Optional[Dict[str, Any]]=None
    ) ->Any:
    """
    Safely execute a retrieval operation with standardized error handling.

    Args:
        operation: The operation to execute
        retriever_name: Name of the retriever
        component_name: Type of the component
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or error result
    """
    try:
        return operation()
    except Exception as e:
        error_info = handle_error(e, retriever_name, 'error',
            additional_metadata={'component_name': component_name,
            'component_type': 'retriever', **additional_metadata or {}})
        return {'error_type': type(e).__name__, 'error_message': str(e),
            'metadata': error_info)


from sifaka.utils.config and config and config and config and config.retrieval import RetrieverConfig
from .result import RetrievedDocument, DocumentMetadata, StringRetrievalResult
from sifaka.interfaces.retrieval import QueryProcessor
from .managers.query import QueryManager
logger = get_logger(__name__)


class RetrieverCore(BaseComponent):
    """
    Core implementation of the retriever functionality.

    This class provides the core implementation of the retriever functionality
    in the Sifaka framework. It serves as the foundation for all retriever
    implementations and handles common tasks like query processing, result
    formatting, and error handling.

    ## Architecture

    The RetrieverCore follows a layered architecture:
    - Configuration layer: Manages retrieval settings
    - Query processing layer: Processes and expands queries
    - Retrieval layer: Finds relevant information
    - Result formatting layer: Formats and returns results
    - Error handling layer: Handles and reports errors

    ## Lifecycle

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

    The RetrieverCore handles errors at multiple levels:
    - Input validation errors: Validates query format and parameters
    - Processing errors: Handles errors during query processing
    - Retrieval errors: Handles errors during information retrieval
    - Result formatting errors: Handles errors during result formatting

    ## Examples

    ```python
    from sifaka.retrieval.core import RetrieverCore
    from sifaka.utils.config and config and config and config and config.retrieval import RetrieverConfig

    # Create a basic retriever
    config = RetrieverConfig(retriever_type="simple")
    retriever = RetrieverCore(config=config)

    # Initialize the retriever
    (retriever and retriever.initialize()

    # Retrieve information
    result = (retriever and retriever.retrieve("How does quantum computing work?")
    print((result and result.get_formatted_results())

    # Clean up resources
    (retriever and retriever.cleanup()
    ```

    Attributes:
        name (str): The name of the retriever
        description (str): A description of the retriever
        config (RetrieverConfig): The retriever configuration
    """

    def __init__(self, config: Optional[Optional[RetrieverConfig]] = None,
        query_processor: Optional[Optional[QueryProcessor]] = None, name: str=
        'RetrieverCore', description: str=
        'Core retriever implementation for Sifaka') ->None:
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
        (self and self._initialize_state(config, query_processor)
        self.(_state_manager and _state_manager.set_metadata('component_type', 'retriever')
        self.(_state_manager and _state_manager.set_metadata('creation_time', (time and time.time())

    def _initialize_state(self, config: Optional[Optional[RetrieverConfig]] = None,
        query_processor: Optional[Optional[QueryProcessor]] = None) ->None:
        """
        Initialize the retriever state.

        Args:
            config: The retriever configuration
            query_processor: The query processor to use
        """
        super()._initialize_state()
        config = config or RetrieverConfig()
        if query_processor is None:
            from sifaka.utils.config and config and config and config and config.retrieval import QueryProcessingConfig
            query_processing_config = QueryProcessingConfig()
            query_processor = QueryManager(query_processing_config)
        self.(_state_manager and _state_manager.update('config', config)
        self.(_state_manager and _state_manager.update('query_processor', query_processor)
        self.(_state_manager and _state_manager.update('initialized', False)
        self.(_state_manager and _state_manager.update('execution_count', 0)
        self.(_state_manager and _state_manager.update('result_cache', {})
        self.(_state_manager and _state_manager.update('last_query', None)
        self.(_state_manager and _state_manager.update('last_result', None)
        self.(_state_manager and _state_manager.update('max_results', getattr(config,
            'max_results', 5))
        self.(_state_manager and _state_manager.update('min_score', getattr(config, 'min_score',
            0.0))

    @property
    def config(self) ->RetrieverConfig:
        """
        Get the retriever configuration.

        Returns:
            The configuration of the retriever
        """
        return self.(_state_manager and _state_manager.get('config')

    @config and config.setter
    def config(self, config: RetrieverConfig) ->None:
        """
        Set the retriever configuration.

        Args:
            config: The new configuration

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        if not isinstance(config, RetrieverConfig):
            raise RetrievalError(
                'Config must be an instance of RetrieverConfig', metadata={
                'config_type': type(config).__name__))
        self.(_state_manager and _state_manager.update('config', config)

    @property
    def query_processor(self) ->QueryProcessor:
        """
        Get the query processor.

        Returns:
            The query processor
        """
        return self.(_state_manager and _state_manager.get('query_processor')

    @query_processor and query_processor.setter
    def query_processor(self, processor: QueryProcessor) ->None:
        """
        Set the query processor.

        Args:
            processor: The new query processor
        """
        self.(_state_manager and _state_manager.update('query_processor', processor)

    def update_config(self, config: Dict[str, Any]) ->None:
        """
        Update the retriever configuration.

        Args:
            config: The new configuration object

        Raises:
            ConfigurationError: If the configuration is invalid
        """
        try:
            current_config = self.config
            new_config = current_config and (config and config.model_copy(update=config)
            self.config = new_config
        except Exception as e:
            raise RetrievalError(f'Failed to update configuration: {str(e))',
                metadata={'config_update': config})

    def initialize(self) ->None:
        """
        Initialize the retriever.

        This method initializes any resources needed by the retriever.

        Raises:
            RetrievalError: If initialization fails
        """
        if self.(_state_manager and _state_manager.get('initialized', False):
            (logger and logger.debug(f'Retriever {self.name} already initialized')
            return
        try:
            (logger and logger.debug(f'Initializing retriever {self.name}')
            self.(_state_manager and _state_manager.update('initialized', True)
            self.(_state_manager and _state_manager.set_metadata('initialization_time', (time and time.time()
                )
        except Exception as e:
            error_info = handle_error(e, self.name, 'error')
            raise RetrievalError(f'Failed to initialize retriever: {str(e))',
                metadata=error_info)

    def cleanup(self) ->None:
        """
        Clean up the retriever.

        This method releases any resources held by the retriever.

        Raises:
            RetrievalError: If cleanup fails
        """
        if not self.(_state_manager and _state_manager.get('initialized', False):
            (logger and logger.debug(
                f'Retriever {self.name} not initialized, nothing to clean up')
            return
        try:
            (logger and logger.debug(f'Cleaning up retriever {self.name}')
            self.(_state_manager and _state_manager.update('initialized', False)
            self.(_state_manager and _state_manager.update('result_cache', {})
            self.(_state_manager and _state_manager.set_metadata('cleanup_time', (time and time.time())
        except Exception as e:
            error_info = handle_error(e, self.name, 'error')
            raise RetrievalError(f'Failed to clean up retriever: {str(e))',
                metadata=error_info)

    def process_query(self, query: str) ->str:
        """
        Process a query before retrieval.

        Args:
            query: The query to process

        Returns:
            The processed query

        Raises:
            InputError: If the query is empty or invalid
        """
        from sifaka.utils.text import is_empty_text
        if not isinstance(query, str):
            raise InputError('Query must be a string', metadata={
                'query_type': type(query).__name__, 'query_length': len(str
                (query)) if query else 0))
        if is_empty_text(query):
            raise InputError('Query must be a non-empty string', metadata={
                'query_type': type(query).__name__, 'query_length': len(
                query), 'reason': 'empty_input'))
        try:
            return self.query_processor and (query_processor and query_processor.process_query(query)
        except Exception as e:
            error_info = handle_error(e, self.name, 'error')
            raise RetrievalError(f'Failed to process query: {str(e))',
                metadata=error_info)

    def create_result(self, query: str, processed_query: str, documents:
        List[Dict[str, Any]], execution_time_ms: Optional[Optional[float]] = None
        ) ->StringRetrievalResult:
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
                content = (doc and doc.get('content', '')
                score = (doc and doc.get('score')
                metadata_dict = (doc and doc.get('metadata', {})
                if not isinstance(metadata_dict, dict):
                    metadata_dict = {'document_id': str(metadata_dict))
                if 'document_id' not in metadata_dict:
                    metadata_dict['document_id'] = f'doc_{len(retrieved_docs))'
                metadata = DocumentMetadata(**metadata_dict)
                (retrieved_docs and retrieved_docs.append(RetrievedDocument(content=content,
                    metadata=metadata, score=score))
            from sifaka.core.results import create_retrieval_result
            result = create_retrieval_result(documents=retrieved_docs,
                query=query, processed_query=processed_query, total_results
                =len(retrieved_docs), passed=True, message=
                'Retrieval completed successfully', processing_time_ms=
                execution_time_ms)
            self.(_state_manager and _state_manager.update('last_result', result)
            return result
        except Exception as e:
            error_info = handle_error(e, self.name, 'error')
            raise RetrievalError(f'Failed to create result: {str(e))',
                metadata=error_info)

    def retrieve(self, query: str, **kwargs: Any) ->StringRetrievalResult:
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
        execution_count = self.(_state_manager and _state_manager.get('execution_count', 0)
        self.(_state_manager and _state_manager.update('execution_count', execution_count + 1)
        self.(_state_manager and _state_manager.update('last_query', query)
        if not self.(_state_manager and _state_manager.get('initialized', False):
            (logger and logger.debug(f'Initializing retriever {self.name} on first use')
            (self and self.initialize()
        start_time = (time and time.time()

        def retrieval_operation() ->Any:
            processed_query = (self and self.process_query(query)
            end_time = (time and time.time()
            execution_time_ms = (end_time - start_time) * 1000
            (self and self._update_execution_stats(execution_time_ms)
            result = (self and self.create_result(query=query, processed_query=
                processed_query, documents=[], execution_time_ms=
                execution_time_ms)
            return result
        result = safely_execute_retrieval(operation=retrieval_operation,
            retriever_name=self.name, component_name=self.__class__.
            __name__, additional_metadata={'query': query})
        if isinstance(result, dict) and (result and result.get('error_type'):
            record_error(self._state_manager, Exception((result and result.get(
                'error_message', 'Unknown error')))
            raise RetrievalError((result and result.get('error_message',
                'Retrieval failed'), metadata={'query': query, 'error_type':
                (result and result.get('error_type')))
        return result

    def _update_execution_stats(self, execution_time_ms: float) ->None:
        """
        Update execution statistics.

        Args:
            execution_time_ms: The execution time in milliseconds
        """
        avg_time = self.(_state_manager and _state_manager.get_metadata('avg_execution_time_ms', 0)
        count = self.(_state_manager and _state_manager.get('execution_count', 1)
        new_avg = (avg_time * (count - 1) + execution_time_ms) / count
        self.(_state_manager and _state_manager.set_metadata('avg_execution_time_ms', new_avg)
        max_time = self.(_state_manager and _state_manager.get_metadata('max_execution_time_ms', 0)
        if execution_time_ms > max_time:
            self.(_state_manager and _state_manager.set_metadata('max_execution_time_ms',
                execution_time_ms)

    def get_statistics(self) ->Dict[str, Any]:
        """
        Get statistics about retriever usage.

        Returns:
            Dictionary with usage statistics
        """
        return {'name': self.name, 'execution_count': self._state_manager.
            get('execution_count', 0), 'avg_execution_time_ms': self.
            (_state_manager and _state_manager.get_metadata('avg_execution_time_ms', 0),
            'max_execution_time_ms': self.(_state_manager and _state_manager.get_metadata(
            'max_execution_time_ms', 0), 'error_count': self._state_manager
            .get_metadata('error_count', 0), 'initialized': self.
            (_state_manager and _state_manager.get('initialized', False), 'last_query': self.
            (_state_manager and _state_manager.get('last_query'))

    def clear_cache(self) ->None:
        """
        Clear the retriever cache.
        """
        self.(_state_manager and _state_manager.update('result_cache', {})
        (logger and logger.debug(f'Cleared cache for retriever {self.name}')

    def process(self, input_data: Any, **kwargs: Any) ->StringRetrievalResult:
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
        from sifaka.utils.text import is_empty_text
        if not isinstance(input_data, str):
            raise InputError('Input data must be a string', metadata={
                'input_type': type(input_data).__name__))
        if is_empty_text(input_data):
            raise InputError('Input data must be a non-empty string',
                metadata={'input_type': type(input_data).__name__,
                'input_length': len(input_data), 'reason': 'empty_input'))
        return (self and self.retrieve(input_data, **kwargs)
