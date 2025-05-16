# Sifaka Retrieval

This package provides retrieval components for the Sifaka framework, enabling information retrieval from various sources based on queries. It follows a component-based architecture consistent with other Sifaka modules.

## Architecture

The retrieval architecture follows a component-based design for maximum flexibility and extensibility:

```
RetrieverCore
├── Document Management
│   ├── Document Store (manages document collections)
│   └── Index Management (manages search indices)
├── Query Processing
│   ├── QueryManager (handles query processing)
│   └── QueryProcessor (processes queries for retrieval)
├── Ranking Components
│   ├── RankingStrategy (abstract base for ranking)
│   ├── SimpleRankingStrategy (basic keyword matching)
│   └── ScoreThresholdRankingStrategy (filters by score)
└── Result Management
    ├── RetrievalResult (standardized result format)
    ├── StringRetrievalResult (text-focused results)
    └── RetrievedDocument (document representation)
```

## Core Components

- **RetrieverCore**: Foundation implementation for all retrievers
- **Managers**: Components for query processing and document management
- **Strategies**: Components that implement ranking and scoring logic
- **Results**: Standardized result objects for retrieval operations
- **Factories**: Factory functions for creating retrievers with sensible defaults

## Retriever Implementations

- **SimpleRetriever**: Basic in-memory retriever that works with document collections

## Usage

### Basic Usage

```python
from sifaka.retrieval import create_simple_retriever

# Create a simple retriever with in-memory documents
documents = {
    "quantum_computing": "Quantum computing uses quantum bits or qubits to perform computations using quantum mechanical phenomena.",
    "machine_learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "neural_networks": "Neural networks are computing systems inspired by the biological neural networks in animal brains."
}

# Create retriever with default settings (top 3 results)
retriever = create_simple_retriever(documents=documents)

# Retrieve information
result = retriever.retrieve("How does quantum computing work?")

# Access results
print(f"Retrieved {len(result.documents)} documents")
print(f"Top result: {result.documents[0].content}")
print(f"Scores: {[doc.score for doc in result.documents]}")

# Get formatted results as text
formatted = result.get_formatted_results()
print(formatted)
```

### Using Score Thresholds

```python
from sifaka.retrieval import create_threshold_retriever

# Create documents
documents = {
    "quantum_computing": "Quantum computing uses quantum bits or qubits to perform computations using quantum mechanical phenomena.",
    "machine_learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "neural_networks": "Neural networks are computing systems inspired by the biological neural networks in animal brains."
}

# Create retriever with score threshold (0.3)
retriever = create_threshold_retriever(
    documents=documents,
    max_results=5,
    threshold=0.3  # Only return documents with score >= 0.3
)

# Retrieve information
result = retriever.retrieve("How does deep learning work?")

# Check if any results passed the threshold
if result.documents:
    print(f"Found {len(result.documents)} relevant documents")
else:
    print("No documents met the threshold criteria")
```

### Loading Documents from a File

```python
from sifaka.retrieval import create_simple_retriever

# Create retriever from a text file (one document per line)
retriever = create_simple_retriever(
    corpus="path/to/documents.txt",
    max_results=10
)

# Retrieve information
result = retriever.retrieve("What is artificial intelligence?")

# Access statistics
stats = retriever.get_statistics()
print(f"Document count: {stats['document_count']}")
print(f"Query execution time: {stats['avg_execution_time_ms']:.2f}ms")
```

### Advanced Configuration

```python
from sifaka.retrieval import SimpleRetriever
from sifaka.utils.config.retrieval import RetrieverConfig, RankingConfig, QueryConfig

# Create advanced configuration
ranking_config = RankingConfig(
    top_k=5,
    score_threshold=0.2,
    algorithm="simple"
)

query_config = QueryConfig(
    preprocessing_steps=["lowercase", "remove_stopwords", "remove_punctuation"]
)

retriever_config = RetrieverConfig(
    max_results=5,
    ranking=ranking_config,
    query=query_config
)

# Create documents
documents = {
    "quantum_computing": "Quantum computing uses quantum bits or qubits to perform computations using quantum mechanical phenomena.",
    "machine_learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "neural_networks": "Neural networks are computing systems inspired by the biological neural networks in animal brains."
}

# Create retriever with advanced configuration
retriever = SimpleRetriever(
    documents=documents,
    config=retriever_config,
    name="CustomRetriever",
    description="Custom retriever with advanced configuration"
)

# Retrieve information
result = retriever.retrieve("Tell me about deep learning")
```

### Retrieval with Per-Query Parameters

```python
from sifaka.retrieval import create_simple_retriever

# Create a simple retriever
documents = {
    "quantum_computing": "Quantum computing uses quantum bits or qubits to perform computations using quantum mechanical phenomena.",
    "machine_learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
    "neural_networks": "Neural networks are computing systems inspired by the biological neural networks in animal brains."
}
retriever = create_simple_retriever(documents=documents)

# Retrieve with per-query parameter overrides
result1 = retriever.retrieve(
    "How does quantum computing work?",
    max_results=2  # Override default max_results just for this query
)

result2 = retriever.retrieve(
    "Tell me about neural networks",
    threshold=0.4  # Apply a threshold just for this query
)

# Compare results
print(f"Query 1 returned {len(result1.documents)} documents")
print(f"Query 2 returned {len(result2.documents)} documents")
```

## Extending

### Creating a Custom Retriever

You can create a custom retriever by extending the `RetrieverCore` class:

```python
from typing import Any, Dict
from sifaka.retrieval import RetrieverCore
from sifaka.core.results import RetrievalResult

class VectorDBRetriever(RetrieverCore):
    """Custom retriever that uses a vector database for retrieval."""

    def __init__(self, connection_string: str, **kwargs):
        """Initialize with vector database connection."""
        super().__init__(**kwargs)
        self._connection_string = connection_string

        # Connect to the vector database
        self._client = self._connect_to_db(connection_string)

        # Store in state manager
        self._state_manager.update("client", self._client)

    def _connect_to_db(self, connection_string: str) -> Any:
        """Connect to vector database."""
        # Implement connection logic here
        from vector_db_library import connect
        return connect(connection_string)

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """Retrieve from vector database."""
        # Call parent method to handle state tracking
        super().retrieve(query, **kwargs)

        # Process the query
        processed_query = self.process_query(query)

        # Get client from state
        client = self._state_manager.get("client")

        # Get max results parameter
        max_results = kwargs.get("max_results", self.config.max_results)

        # Query the vector database
        search_results = client.search(
            query=processed_query,
            limit=max_results
        )

        # Convert to standardized document format
        documents = [
            {
                "content": result.content,
                "metadata": {"document_id": result.id, "source": "vector_db"},
                "score": result.similarity
            }
            for result in search_results
        ]

        # Create and return result
        return self.create_result(
            query=query,
            processed_query=processed_query,
            documents=documents
        )
```

### Creating a Custom Ranking Strategy

```python
from typing import Any, Dict, List
from sifaka.retrieval.strategies.ranking import RankingStrategy
from sifaka.utils.config.retrieval import RankingConfig

class SemanticRankingStrategy(RankingStrategy):
    """Custom ranking strategy using semantic similarity."""

    def __init__(self, config: RankingConfig = None, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with model for semantic similarity."""
        super().__init__(config)
        self.model_name = model_name

        # Import sentence-transformers (requires installation)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def rank(self, query: str, documents: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Rank documents using semantic similarity."""
        if not documents:
            return []

        # Extract document content
        contents = [doc["content"] for doc in documents]

        # Encode query and documents
        query_embedding = self.model.encode(query)
        doc_embeddings = self.model.encode(contents)

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            [query_embedding],
            doc_embeddings
        )[0]

        # Add scores to documents
        scored_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["score"] = float(similarities[i])
            scored_docs.append(doc_copy)

        # Sort by score in descending order
        sorted_docs = sorted(scored_docs, key=lambda d: d["score"], reverse=True)

        # Apply threshold if configured
        if self.config and self.config.score_threshold is not None:
            sorted_docs = [
                doc for doc in sorted_docs
                if doc["score"] >= self.config.score_threshold
            ]

        # Apply top_k if configured
        if self.config and self.config.top_k is not None:
            sorted_docs = sorted_docs[:self.config.top_k]

        return sorted_docs
```

### Creating a Factory Function

```python
from typing import Any, Dict, Optional
from sifaka.utils.config.retrieval import RetrieverConfig
from sifaka.utils.errors.component import RetrievalError

def create_vector_retriever(
    connection_string: str,
    collection_name: str,
    max_results: int = 5,
    threshold: Optional[float] = None,
    name: str = "VectorDBRetriever",
    description: str = "Retriever using vector database",
    **kwargs: Any
) -> Any:
    """
    Create a vector database retriever.

    Args:
        connection_string: Database connection string
        collection_name: Name of the vector collection
        max_results: Maximum number of results to return
        threshold: Optional similarity threshold
        name: Retriever name
        description: Retriever description
        **kwargs: Additional parameters

    Returns:
        A configured vector database retriever
    """
    try:
        # Import the retriever class
        from my_module import VectorDBRetriever

        # Create ranking configuration
        ranking_params = {
            "top_k": max_results,
            "score_threshold": threshold
        }

        # Create configuration
        config = RetrieverConfig(
            max_results=max_results,
            ranking=ranking_params,
            **kwargs
        )

        # Create and return retriever
        return VectorDBRetriever(
            connection_string=connection_string,
            collection_name=collection_name,
            config=config,
            name=name,
            description=description
        )

    except Exception as e:
        # Handle errors
        raise RetrievalError(f"Failed to create vector retriever: {str(e)}")
```
