# Composition Over Inheritance Implementation Plan for Retrieval

This document outlines the plan for implementing the Composition Over Inheritance pattern in the Sifaka retrieval system.

## Current Architecture

Currently, the retrieval system uses inheritance:
- `Retriever` is an abstract base class that defines the interface
- Specific retrievers like `SimpleRetriever` inherit from `Retriever`
- The system follows a similar pattern to what was previously used in classifiers and chains

## Target Architecture

We'll refactor the retrieval system to use composition over inheritance:

1. Create a `RetrieverImplementation` protocol that defines the core retrieval logic
2. Create a `Retriever` class that delegates to a `RetrieverImplementation`
3. Create specific implementations like `SimpleRetrieverImplementation` that follow the protocol
4. Update factory functions to create retrievers with their implementations

## Implementation Status

| Retriever | Implementation Status | Factory Function Updated | Async Support | Notes |
|-----------|----------------------|-------------------------|--------------|-------|
| SimpleRetriever | ✅ Completed | ✅ Completed | ✅ Completed | Basic in-memory document retriever |

## Implementation Steps

### 1. RetrieverImplementation Protocol

The `RetrieverImplementation` protocol will be added to `base.py`:

```python
@runtime_checkable
class RetrieverImplementation(Protocol):
    """
    Protocol for retriever implementations.

    This protocol defines the core retrieval logic that can be composed with
    the Retriever class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.
    """

    def retrieve_impl(self, query: str) -> str: ...
    def warm_up_impl(self) -> None: ...
    async def aretrieve_impl(self, query: str) -> str: ...  # Optional async method
```

### 2. Retriever Class

The `Retriever` class will be updated in `base.py`:

```python
class Retriever(BaseModel):
    """
    Retriever that uses composition over inheritance.

    This class delegates retrieval to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    name: str = Field(description="Name of the retriever", min_length=1)
    description: str = Field(description="Description of the retriever", min_length=1)
    config: RetrieverConfig
    _implementation: RetrieverImplementation = PrivateAttr()

    def __init__(
        self,
        name: str,
        description: str,
        config: RetrieverConfig,
        implementation: RetrieverImplementation,
        **kwargs: Any,
    ):
        """Initialize the retriever."""
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
```

### 3. Implementation Plan for Each Retriever

#### 3.1 SimpleRetriever Implementation

```python
class SimpleRetrieverImplementation:
    """Implementation of simple retrieval logic."""

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.documents = config.params.get("documents", {})
        self.max_results = config.params.get("max_results", 3)

    def retrieve_impl(self, query: str) -> str:
        """Implement simple retrieval logic."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if not self.documents:
            return "No documents available for retrieval."

        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return "Could not extract meaningful keywords from query."

        # Find relevant documents
        relevant_docs = self._find_relevant_documents(keywords)
        if not relevant_docs:
            return "No relevant documents found for the query."

        # Format results
        return self._format_results(relevant_docs)

    def warm_up_impl(self) -> None:
        """Prepare resources for retrieval."""
        # No special initialization needed for SimpleRetriever
        pass

    async def aretrieve_impl(self, query: str) -> str:
        """
        Asynchronously implement simple retrieval logic.

        This implementation simply calls the synchronous version since
        the operations are lightweight and don't require async I/O.
        """
        return self.retrieve_impl(query)
```

### 4. Update Factory Functions

```python
def create_simple_retriever(
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    max_results: int = 3,
    name: str = "simple_retriever",
    description: str = "A simple retriever for in-memory document collections",
    **kwargs: Any,
) -> Retriever:
    """Factory function to create a simple retriever."""
    # Prepare params
    params = kwargs.pop("params", {})
    params.update({
        "max_results": max_results,
    })

    # Load documents
    if documents is not None:
        params["documents"] = documents
    elif corpus is not None:
        try:
            loaded_documents = {}
            with open(corpus, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    loaded_documents[f"doc_{i}"] = line.strip()
            params["documents"] = loaded_documents
        except FileNotFoundError:
            raise FileNotFoundError(f"Corpus file not found: {corpus}")
    else:
        # Initialize with empty dict, but warn
        import warnings
        warnings.warn("Initializing SimpleRetriever with empty document collection")
        params["documents"] = {}

    # Create config
    config = RetrieverConfig(params=params)

    # Create implementation
    implementation = SimpleRetrieverImplementation(config)

    # Create and return retriever
    return Retriever(
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
```

## Files to Modify

1. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/retrieval/base.py`
   - Add `RetrieverImplementation` protocol
   - Add `RetrieverConfig` class
   - Update `Retriever` class to use composition

2. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/retrieval/simple.py`
   - Add `SimpleRetrieverImplementation` class
   - Add `create_simple_retriever` factory function

3. `/Users/evanvolgas/Documents/not_beam/sifaka/sifaka/retrieval/__init__.py`
   - Update exports to include new classes and functions

## Implementation Strategy

1. Start with updating the base.py file to add the protocol and config classes
2. Then update the Retriever class to use composition
3. Implement the SimpleRetrieverImplementation and factory function
4. Update the __init__.py file to export the new components
5. Update any code that depends on the old Retriever class

## Benefits

- Reduced complexity by avoiding deep inheritance hierarchies
- Improved flexibility by allowing components to be combined in different ways
- Better testability by enabling testing of implementations in isolation
- Reduced coupling between components
- More consistent API across all retrievers
- Follows the same pattern used in classifiers and chains

## Next Steps After Implementation

1. Update any code that depends on the old Retriever class:
   - ✅ Update imports and usage patterns in examples
   - ✅ Update SelfRAGCritic to use the new async retrieval method
   - ✅ Ensure backward compatibility where needed

2. Update tests to work with the new pattern:
   - Update test fixtures to create retrievers using the new pattern
   - Update assertions to work with the new retriever structure
   - Add tests for the async retrieval functionality

3. Update documentation to reflect the new pattern:
   - ✅ Update docstrings to describe the new architecture
   - ✅ Update examples to use the new factory functions
   - ✅ Add implementation notes about the Composition Over Inheritance pattern

4. Implement additional retriever types:
   - Consider adding a VectorRetriever implementation
   - Consider adding a DatabaseRetriever implementation
   - Consider adding a WebRetriever implementation
