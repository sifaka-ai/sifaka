# Sifaka Retrieval

This directory contains the retrieval components for the Sifaka framework. These components enable retrieving relevant information from various sources based on queries.

## Directory Structure

```
sifaka/retrieval/
├── __init__.py           # Public API exports
├── core.py               # Core retrieval implementation
├── config.py             # Retrieval configuration
├── result.py             # Retrieval result models
├── factories.py          # Factory functions
├── interfaces/           # Define protocol interfaces
│   ├── __init__.py
│   └── retriever.py      # Retriever protocols
├── managers/             # Component managers
│   ├── __init__.py
│   └── query.py          # Query management
├── strategies/           # Strategy implementations
│   ├── __init__.py
│   └── ranking.py        # Ranking strategies
└── implementations/      # Concrete retriever implementations
    ├── __init__.py
    └── simple.py         # Simple retriever
```

## Components

### Core Components

- `RetrieverCore`: Core implementation of retriever functionality
- `RetrieverConfig`: Configuration for retrievers
- `RetrievalResult`: Result models for retrieval operations

### Interfaces

- `Retriever`: Interface for retrievers
- `AsyncRetriever`: Interface for asynchronous retrievers
- `DocumentStore`: Interface for document stores
- `IndexManager`: Interface for index managers
- `QueryProcessor`: Interface for query processors

### Implementations

- `SimpleRetriever`: A basic retriever for in-memory document collections

### Managers

- `QueryManager`: Manager for query processing

### Strategies

- `RankingStrategy`: Abstract base class for ranking strategies
- `SimpleRankingStrategy`: Simple ranking strategy based on keyword matching
- `ScoreThresholdRankingStrategy`: Ranking strategy with score thresholding

### Factory Functions

- `create_simple_retriever`: Create a simple retriever
- `create_threshold_retriever`: Create a retriever with score thresholding

## Usage Examples

### Basic Usage

```python
from sifaka.retrieval import create_simple_retriever

# Create a simple retriever with a document collection
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = create_simple_retriever(documents=documents, max_results=3)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result.get_formatted_results())
```

### Using Score Thresholds

```python
from sifaka.retrieval import create_threshold_retriever

# Create a retriever with a score threshold
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = create_threshold_retriever(
    documents=documents,
    max_results=5,
    threshold=0.3,  # Only return documents with a score >= 0.3
)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result.get_formatted_results())
```

### Custom Configuration

```python
from sifaka.retrieval import SimpleRetriever, RetrieverConfig

# Create a custom configuration
config = RetrieverConfig(
    retriever_type="simple",
    ranking={"top_k": 5, "score_threshold": 0.2},
    query_processing={"preprocessing_steps": ["lowercase", "remove_stopwords", "remove_punctuation"]},
)

# Create a retriever with the custom configuration
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = SimpleRetriever(documents=documents, config=config)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result.get_formatted_results())
```

## Extending the Framework

### Creating a Custom Retriever

To create a custom retriever, extend the `RetrieverCore` class:

```python
from sifaka.retrieval import RetrieverCore, StringRetrievalResult

class CustomRetriever(RetrieverCore):
    """A custom retriever implementation."""

    def __init__(self, api_key, **kwargs):
        super().__init__(**kwargs)
        self._name = "CustomRetriever"
        self._description = "Custom retriever using an external API"
        self.api_key = api_key

    def retrieve(self, query, **kwargs):
        # Process the query
        processed_query = self.process_query(query)
        
        # Implement your custom retrieval logic here
        # For example, call an external API
        
        # Create and return a result
        return self.create_result(
            query=query,
            processed_query=processed_query,
            documents=[
                {
                    "content": "Retrieved content...",
                    "metadata": {"document_id": "doc1", "source": "external_api"},
                    "score": 0.95,
                },
                # More documents...
            ],
        )
```

### Creating a Custom Ranking Strategy

To create a custom ranking strategy, extend the `RankingStrategy` class:

```python
from sifaka.retrieval.strategies.ranking import RankingStrategy

class CustomRankingStrategy(RankingStrategy):
    """A custom ranking strategy."""

    def rank(self, query, documents, **kwargs):
        # Implement your custom ranking logic here
        # For example, use a machine learning model to rank documents
        
        # Return ranked documents with scores
        return [
            {**doc, "score": calculate_score(doc, query)}
            for doc in documents
        ]
```
