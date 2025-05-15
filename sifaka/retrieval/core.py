from typing import Any, Dict, List, Optional, Union, cast
import time
import logging
from sifaka.core.base import BaseComponent
from sifaka.utils.errors.base import InputError
from sifaka.utils.errors.component import RetrievalError
from sifaka.core.results import RetrievalResult
from sifaka.utils.state import StateManager, create_retriever_state
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.common import record_error
from sifaka.utils.errors.safe_execution import safely_execute_retrieval
from sifaka.utils.errors.results import ErrorResult
from .result import DocumentMetadata, RetrievedDocument
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class RetrieverCore(BaseComponent):
    """
    Core implementation of retriever functionality.

    A retriever is a component that retrieves information based on a query.
    It can be used to retrieve documents from a vector store, search engine, or any other source.

    Attributes:
        name: The name of the retriever
        query_processor: Optional query processor to preprocess queries
        _state_manager: State manager for tracking retriever state
    """

    # Use PrivateAttr with factory function for state management
    _state_manager: StateManager = PrivateAttr(default_factory=create_retriever_state)

    def __init__(self, name: str = "base_retriever", query_processor: Any = None) -> None:
        """
        Initialize a new RetrieverCore.

        Args:
            name: The name of the retriever
            query_processor: Optional query processor to preprocess queries
        """
        super().__init__(name=name, description=f"Retriever: {name}")
        self.query_processor = query_processor

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize component state with standardized patterns."""
        # Call parent initialization
        super()._initialize_state()

        # Initialize retriever-specific state
        self._state_manager.update("initialized", False)
        self._state_manager.update("query_processor", self.query_processor)

        # Set component metadata
        self._state_manager.set_metadata("component_type", "retriever")
        self._state_manager.set_metadata("creation_time", time.time())

    def initialize(self) -> None:
        """
        Initialize the retriever.

        This method should be overridden by subclasses to perform any necessary initialization.
        """
        self._state_manager.update("initialized", True)
        logger.debug(f"Initialized retriever {self.name}")

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
        from sifaka.utils.text import is_empty_text

        if not isinstance(query, str):
            raise InputError(
                "Query must be a string",
                metadata={
                    "query_type": type(query).__name__,
                    "query_length": len(str(query)) if query else 0,
                },
            )
        if is_empty_text(query):
            raise InputError(
                "Query must be a non-empty string",
                metadata={
                    "query_type": type(query).__name__,
                    "query_length": len(query),
                    "reason": "empty_input",
                },
            )
        try:
            query_processor = self.query_processor
            return query_processor.process_query(query) if query_processor else query
        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to process query: {str(e)}", metadata=error_info)

    def create_result(
        self,
        query: str,
        processed_query: str,
        documents: List[Dict[str, Any]],
        execution_time_ms: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Create a retrieval result from raw documents.

        Args:
            query: The original query
            processed_query: The processed query
            documents: The raw documents (list of dicts with 'content', 'metadata', and 'score')
            execution_time_ms: The execution time in milliseconds

        Returns:
            A RetrievalResult object

        Raises:
            RetrievalError: If result creation fails
        """
        try:
            retrieved_docs: List[RetrievedDocument] = []
            for doc in documents:
                content = doc.get("content", "") if doc else ""
                score = doc.get("score") if doc else ""
                metadata_dict = doc.get("metadata", {}) if doc else ""
                if not isinstance(metadata_dict, dict):
                    metadata_dict = {"document_id": str(metadata_dict)}
                if "document_id" not in metadata_dict:
                    metadata_dict["document_id"] = f"doc_{len(retrieved_docs)}"
                metadata = DocumentMetadata(**metadata_dict)
                retrieved_docs.append(
                    RetrievedDocument(
                        content=content,
                        metadata=metadata,
                        score=(
                            float(score) if score and isinstance(score, (int, float, str)) else None
                        ),
                    )
                )
            from sifaka.core.results import create_retrieval_result

            result = create_retrieval_result(
                documents=retrieved_docs,
                query=query,
                processed_query=processed_query,
                total_results=len(retrieved_docs),
                passed=True,
                message="Retrieval completed successfully",
                processing_time_ms=execution_time_ms,
            )
            self._state_manager.update("last_result", result)
            return result
        except Exception as e:
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(f"Failed to create result: {str(e)}", metadata=error_info)

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        """
        Retrieve information based on a query.

        This is a base implementation that should be overridden by subclasses.
        It processes the query and returns an empty result.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            A RetrievalResult object

        Raises:
            InputError: If the query is empty or invalid
            RetrievalError: If retrieval fails
        """
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)
        self._state_manager.update("last_query", query)
        if not self._state_manager.get("initialized", False):
            if logger:
                logger.debug(f"Initializing retriever {self.name} on first use")
            self.initialize()
        start_time = time.time()

        def retrieval_operation() -> RetrievalResult:
            processed_query = self.process_query(query)
            end_time = time.time()
            execution_time_ms = (
                (float(end_time) - float(start_time)) * 1000
                if isinstance(end_time, (int, float)) and isinstance(start_time, (int, float))
                else 0.0
            )
            self._update_execution_stats(execution_time_ms)
            result = self.create_result(
                query=query,
                processed_query=processed_query,
                documents=[],
                execution_time_ms=execution_time_ms,
            )
            return result

        result = safely_execute_retrieval(
            retrieval_operation,
            self.name,
            None,  # default_result
            "error",  # log_level
            True,  # include_traceback
            None,  # additional_metadata
        )

        if isinstance(result, dict) and result.get("error_type"):
            error_message = result.get("error_message", "Unknown error")
            error_type = result.get("error_type", "")
            record_error(self._state_manager, Exception(error_message))
            raise RetrievalError(error_message, metadata={"query": query, "error_type": error_type})

        return cast(RetrievalResult, result)

    def _update_execution_stats(self, execution_time_ms: float) -> None:
        """
        Update execution statistics.

        Args:
            execution_time_ms: The execution time in milliseconds
        """
        avg_time = self._state_manager.get_metadata("avg_execution_time_ms", 0)
        count = self._state_manager.get("execution_count", 1)
        new_avg = (avg_time * (count - 1) + execution_time_ms) / count
        self._state_manager.set_metadata("avg_execution_time_ms", new_avg)
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
            "last_query": self._state_manager.get("last_query", ""),
        }

    def clear_cache(self) -> None:
        """
        Clear the retriever cache.
        """
        self._state_manager.update("result_cache", {})
        if logger:
            logger.debug(f"Cleared cache for retriever {self.name}")

    def process(self, input: Any) -> RetrievalResult:
        """
        Process input data.

        This method is required by the BaseComponent abstract class.

        Args:
            input_data: The input data to process (query string)
            **kwargs: Additional processing parameters

        Returns:
            A RetrievalResult object

        Raises:
            InputError: If the input is not a string
            RetrievalError: If retrieval fails
        """
        from sifaka.utils.text import is_empty_text

        if not isinstance(input, str):
            raise InputError(
                "Input data must be a string", metadata={"input_type": type(input).__name__}
            )
        if is_empty_text(input):
            raise InputError(
                "Input data must be a non-empty string",
                metadata={
                    "input_type": type(input).__name__,
                    "input_length": len(input),
                    "reason": "empty_input",
                },
            )
        return self.retrieve(input)
