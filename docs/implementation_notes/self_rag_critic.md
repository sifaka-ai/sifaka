# Implementation Notes: Self-RAG Critic

This document provides implementation details and notes for the Self-RAG Critic in the Sifaka project.

## Overview

The Self-RAG Critic implements the Self-Reflective Retrieval-Augmented Generation approach from the paper [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). It enables language models to decide when to retrieve information, generate responses using retrieved information, and reflect on the quality and relevance of the information used.

## Implementation Details

### State Management

The Self-RAG Critic follows the standard state management pattern used in other Sifaka critics:

- Uses a `CriticState` object to store all mutable state
- Stores configuration values and components in the state's cache dictionary
- Accesses state through direct state access

```python
# Initialize state
self._state = CriticState()

# Store components in state
self._state.model = llm_provider
self._state.cache = {
    "retriever": retriever,
    "system_prompt": config.system_prompt,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
    "retrieval_threshold": config.retrieval_threshold,
    "retrieval_prompt_template": config.retrieval_prompt_template,
    "generation_prompt_template": config.generation_prompt_template,
    "reflection_prompt_template": config.reflection_prompt_template,
}
self._state.initialized = True
```

### Configuration

The Self-RAG Critic uses a dedicated configuration class that extends `PromptCriticConfig`:

```python
class SelfRAGCriticConfig(PromptCriticConfig):
    """Configuration for the Self-RAG critic."""

    retrieval_threshold: float = Field(
        default=0.5,
        description="Threshold for determining when to retrieve information",
        ge=0.0,
        le=1.0,
    )
    retrieval_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for retrieval query generation",
    )
    generation_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for response generation",
    )
    reflection_prompt_template: Optional[str] = Field(
        default=None,
        description="Template for self-reflection",
    )
```

### Core Algorithm

The core algorithm for the Self-RAG Critic is implemented in the `run` method:

1. Generate a retrieval query based on the task
2. Determine if retrieval is needed based on the query
3. If retrieval is needed, retrieve relevant information
4. Generate a response using the retrieved information (if any)
5. Generate a reflection on the response and retrieval process
6. Return the results

```python
# Step 1: Generate retrieval query
retrieval_query = self._generate_retrieval_query(task)

# Step 2: Determine if retrieval is needed
retrieval_needed = self._should_retrieve(retrieval_query)

# Step 3: Retrieve information if needed
retrieved_context = ""
if retrieval_needed:
    retrieved_context = self._retrieve_information(retrieval_query)

# Step 4: Generate response
response = self._generate_response(task, retrieved_context)

# Step 5: Generate reflection
reflection = self._generate_reflection(task, retrieved_context, response)

# Return results
return {
    "response": response,
    "retrieval_query": retrieval_query,
    "retrieved_context": retrieved_context,
    "reflection": reflection,
}
```

### Retrieval Integration

The Self-RAG Critic integrates with the retrieval system through a `Retriever` interface:

```python
class Retriever(Protocol):
    """Protocol for retrieval systems."""

    def retrieve(self, query: str) -> str:
        """
        Retrieve information based on a query.

        Args:
            query: The query to retrieve information for

        Returns:
            Retrieved information as a string
        """
        ...
```

A simple implementation, `SimpleRetriever`, is provided for testing and demonstration purposes:

```python
class SimpleRetriever:
    """A simple retriever that searches a dictionary of documents."""

    def __init__(self, documents: Dict[str, str]):
        """Initialize the retriever with a dictionary of documents."""
        self.documents = documents

    def retrieve(self, query: str) -> str:
        """Retrieve information based on a query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Find documents that match the query
        matches = []
        for key, content in self.documents.items():
            if any(term.lower() in key.lower() or term.lower() in content.lower() 
                  for term in query.lower().split()):
                matches.append(content)

        # Return formatted results
        if not matches:
            return "No relevant documents found for the query."

        return "Retrieved information:\n\n" + "\n\n".join(
            f"Document {i+1}:\n\n{content}\n" for i, content in enumerate(matches)
        )
```

### Factory Function

The Self-RAG Critic provides a factory function for easy creation:

```python
def create_self_rag_critic(
    llm_provider: Any,
    retriever: Any,
    name: str = "self_rag_critic",
    description: str = "Improves text through self-reflective retrieval-augmented generation",
    system_prompt: str = "You are an expert at deciding when to retrieve information and reflecting on its relevance.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    retrieval_threshold: float = 0.5,
    retrieval_prompt_template: Optional[str] = None,
    generation_prompt_template: Optional[str] = None,
    reflection_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRAGCriticConfig]] = None,
    **kwargs: Any,
) -> SelfRAGCritic:
    # Implementation details...
```

## Integration with Sifaka

The Self-RAG Critic is integrated with the Sifaka project in the following ways:

1. Added to the `critics` module with proper imports and exports
2. Added to the `__all__` list in `critics/__init__.py`
3. Added a default configuration `DEFAULT_SELF_RAG_CONFIG`
4. Provided comprehensive tests in `tests/critics/test_self_rag.py`
5. Provided an example in `examples/self_rag_example.py`

## Testing

The Self-RAG Critic includes comprehensive tests that verify:

1. Initialization with different configurations
2. Retrieval query generation
3. Response generation with and without retrieved information
4. Self-reflection generation
5. The full Self-RAG process
6. Factory function behavior
7. Asynchronous operations

## Future Improvements

Potential future improvements for the Self-RAG Critic include:

1. Adding support for more sophisticated retrieval systems
2. Implementing a more robust parsing of retrieval queries
3. Adding support for multi-step retrieval with different queries
4. Implementing a more sophisticated scoring mechanism for reflections
5. Adding support for tracking the history of retrievals and responses
6. Supporting streaming responses
7. Implementing a more sophisticated retrieval threshold mechanism

## References

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [Sifaka Critics Documentation](../components/critics.md)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
