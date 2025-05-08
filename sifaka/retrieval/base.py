"""
Base retrieval interface for Sifaka.

This module provides the Retriever class and RetrieverImplementation protocol
that define the interface for all retrieval implementations in Sifaka.
It follows the composition over inheritance pattern for more flexible and
maintainable code.

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

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


@dataclass(frozen=True)
class RetrieverConfig:
    """
    Immutable configuration for retrievers.

    This class provides a standardized way to configure retrievers with
    immutable properties. All retriever-specific configuration options
    are placed in the params dictionary.

    The immutable design ensures configuration consistency during retriever
    operation and prevents accidental modification of settings.

    ## Lifecycle

    1. **Creation**: Instantiate with required and optional parameters
       - Set retriever-specific options in params dictionary

    2. **Usage**: Pass to retriever factory functions
       - Factory functions extract needed parameters
       - Implementation uses params for configuration

    ## Examples

    ```python
    from sifaka.retrieval.base import RetrieverConfig

    # Create a configuration for a simple retriever
    config = RetrieverConfig(
        params={
            "max_results": 5,
            "documents": {
                "doc1": "Content of document 1",
                "doc2": "Content of document 2"
            }
        }
    )
    ```

    Attributes:
        params: Dictionary of retriever-specific parameters
    """

    params: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrieverImplementation(Protocol):
    """
    Protocol for retriever implementations.

    This protocol defines the core retrieval logic that can be composed with
    the Retriever class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.

    ## Lifecycle

    1. **Implementation**: Create a class that implements the required methods
       - Implement retrieve_impl() for the core retrieval logic
       - Implement warm_up_impl() for resource initialization
       - Optionally implement aretrieve_impl() for async retrieval

    2. **Composition**: Use with the Retriever class
       - Retriever delegates to the implementation
       - Implementation focuses on core logic

    ## Examples

    ```python
    from sifaka.retrieval.base import RetrieverImplementation, RetrieverConfig

    class SimpleImplementation(RetrieverImplementation):
        def __init__(self, config: RetrieverConfig):
            self.config = config
            self.documents = config.params.get("documents", {})
            self.max_results = config.params.get("max_results", 3)

        def retrieve_impl(self, query: str) -> str:
            # Simple implementation that searches documents
            if not query:
                return "Empty query"

            matches = []
            for doc_id, content in self.documents.items():
                if query.lower() in content.lower():
                    matches.append(content)

            if not matches:
                return "No matches found"

            return "\n\n".join(matches[:self.max_results])

        def warm_up_impl(self) -> None:
            # No special initialization needed
            pass

        async def aretrieve_impl(self, query: str) -> str:
            # Async implementation can be different or just call the sync version
            # For simple implementations, we can just return the sync result
            return self.retrieve_impl(query)
    ```
    """

    def retrieve_impl(self, query: str) -> str: ...
    def warm_up_impl(self) -> None: ...

    # Optional async method - implementations can provide this for better async support
    async def aretrieve_impl(self, query: str) -> str: ...


class Retriever(BaseModel):
    """
    Retriever that uses composition over inheritance.

    This class delegates retrieval to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.

    ## Architecture

    Retriever follows a compositional architecture:
    1. **Public API**: retrieve() method
    2. **Delegation**: Delegates to implementation for core logic
    3. **Configuration**: Manages configuration through RetrieverConfig

    ## Lifecycle

    1. **Initialization**: Set up with name, description, implementation, and config
       - Create with required parameters
       - Store implementation object
       - Set up configuration

    2. **Warm-up**: Prepare resources
       - Call warm_up() to initialize resources
       - Delegates to implementation.warm_up_impl()
       - This step is optional but recommended for performance

    3. **Retrieval**: Process queries
       - Call retrieve() to get information
       - Delegates core logic to implementation

    ## Error Handling

    The class implements these error handling patterns:
    - Input validation in retrieve()
    - Exception handling in retrieval methods

    ## Examples

    Creating a retriever with an implementation:

    ```python
    from sifaka.retrieval.base import Retriever, RetrieverImplementation, RetrieverConfig

    # Create an implementation
    class SimpleImplementation(RetrieverImplementation):
        def __init__(self, config):
            self.config = config
            self.documents = config.params.get("documents", {})
            self.max_results = config.params.get("max_results", 3)

        def retrieve_impl(self, query: str) -> str:
            # Implementation details...
            return "Retrieved information"

        def warm_up_impl(self) -> None:
            # No special initialization needed
            pass

    # Create configuration
    config = RetrieverConfig(
        params={
            "documents": {"doc1": "Content of document 1"},
            "max_results": 5
        }
    )

    # Create implementation
    implementation = SimpleImplementation(config)

    # Create retriever with implementation
    retriever = Retriever(
        name="simple_retriever",
        description="Simple in-memory document retriever",
        config=config,
        implementation=implementation
    )

    # Use the retriever
    result = retriever.retrieve("How does quantum computing work?")
    print(result)
    ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the retriever", min_length=1)
    description: str = Field(description="Description of the retriever", min_length=1)
    config: RetrieverConfig = Field(description="Configuration for the retriever")
    _implementation: RetrieverImplementation = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        config: RetrieverConfig,
        implementation: RetrieverImplementation,
        **kwargs: Any,
    ):
        """
        Initialize the retriever.

        Args:
            name: The name of the retriever
            description: The description of the retriever
            config: The configuration for the retriever
            implementation: The implementation to use
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

    def retrieve(self, query: str) -> str:
        """
        Retrieve information based on a query.

        This method delegates to the implementation's retrieve_impl method.

        Args:
            query: The query to retrieve information for

        Returns:
            Retrieved information as a string

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If retrieval fails
        """
        return self._implementation.retrieve_impl(query)

    def warm_up(self) -> None:
        """
        Warm up the retriever.

        This method delegates to the implementation's warm_up_impl method.
        """
        self._implementation.warm_up_impl()

    async def aretrieve(self, query: str) -> str:
        """
        Asynchronously retrieve information based on a query.

        This method delegates to the implementation's aretrieve_impl method if available,
        or falls back to the synchronous retrieve_impl method using asyncio.to_thread.

        Args:
            query: The query to retrieve information for

        Returns:
            Retrieved information as a string

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If retrieval fails
        """
        # Check if the implementation has an async retrieve method
        if hasattr(self._implementation, "aretrieve_impl"):
            return await self._implementation.aretrieve_impl(query)
        else:
            # Fall back to synchronous implementation using asyncio.to_thread
            import asyncio

            return await asyncio.to_thread(self._implementation.retrieve_impl, query)
