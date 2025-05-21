# Sifaka Refactoring Strategy

This document outlines the strategic plan for refactoring the Sifaka codebase to implement the Thought container, make retrievers available to both models and critics, and rewrite the Model interface to use Thoughts directly.

## Goals

1. Implement a central state container (`Thought`) to track the complete context throughout chain execution
2. Make retrievers available to both models and critics
3. Rewrite the Model interface to use Thoughts directly
4. Implement persistence mechanisms for Thoughts (file-based, Redis, vector databases)
5. Maintain a clean architecture without prioritizing backward compatibility

## Phase 1: Core Infrastructure

### 1.1 Implement the Thought Container

**Tasks:**
- Create `sifaka/state.py` with the `Thought` class
- Implement serialization/deserialization methods
- Add utility methods for tracking and analyzing chain execution
- Write unit tests for the `Thought` class

**Implementation Details:**
```python
# sifaka/state.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import copy
import json
import os
import time

from sifaka.results import ValidationResult, ImprovementResult

@dataclass
class Thought:
    """Container for the complete state of a generation process."""

    prompt: str
    retrieved_context: Optional[List[str]] = None
    generated_text: Optional[str] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    critic_feedback: Optional[str] = None
    improvement_results: List[ImprovementResult] = field(default_factory=list)
    iteration: int = 0
    history: List["Thought"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Add methods for history tracking, serialization, etc.
```

### 1.2 Rewrite the Model Interface

**Tasks:**
- Update `sifaka/models/base.py` to define a new Model protocol that uses Thoughts
- Create base implementations for common model functionality
- Update model factory functions to support the new interface

**Implementation Details:**
```python
# sifaka/models/base.py
from typing import Any, Protocol, runtime_checkable

from sifaka.state import Thought

@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for language model providers."""

    def generate(self, thought: Thought, **options: Any) -> Thought:
        """Generate text using a Thought container.

        Args:
            thought: The Thought container with the prompt and context.
            **options: Additional options for text generation.

        Returns:
            The updated Thought container with generated text.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        ...
```

### 1.3 Implement Basic File-Based Persistence

**Tasks:**
- Add methods to `Thought` for saving to and loading from JSON files
- Create utility functions for managing collections of Thoughts
- Write unit tests for persistence functionality

**Implementation Details:**
```python
# Additional methods for the Thought class

def to_dict(self) -> Dict[str, Any]:
    """Convert the Thought to a dictionary for serialization."""
    # Create a copy without history to avoid recursion
    thought_dict = {
        "prompt": self.prompt,
        "retrieved_context": self.retrieved_context,
        "generated_text": self.generated_text,
        "validation_results": [self._validation_result_to_dict(r) for r in self.validation_results],
        "critic_feedback": self.critic_feedback,
        "improvement_results": [self._improvement_result_to_dict(r) for r in self.improvement_results],
        "iteration": self.iteration,
        "metadata": self.metadata,
        # Convert history separately
        "history": [self._history_item_to_dict(h) for h in self.history]
    }
    return thought_dict

@staticmethod
def _validation_result_to_dict(result: ValidationResult) -> Dict[str, Any]:
    """Convert a ValidationResult to a dictionary."""
    return {
        "passed": result.passed,
        "message": result.message,
        "score": result.score,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "details": result.details
    }

@staticmethod
def _improvement_result_to_dict(result: ImprovementResult) -> Dict[str, Any]:
    """Convert an ImprovementResult to a dictionary."""
    return {
        "original_text": result.original_text,
        "improved_text": result.improved_text,
        "changes_made": result.changes_made,
        "message": result.message,
        "details": result.details,
        "processing_time_ms": result.processing_time_ms
    }

def _history_item_to_dict(self, history_item: "Thought") -> Dict[str, Any]:
    """Convert a history item to a dictionary."""
    # Don't include nested history to avoid recursion
    return {
        "prompt": history_item.prompt,
        "retrieved_context": history_item.retrieved_context,
        "generated_text": history_item.generated_text,
        "validation_results": [self._validation_result_to_dict(r) for r in history_item.validation_results],
        "critic_feedback": history_item.critic_feedback,
        "improvement_results": [self._improvement_result_to_dict(r) for r in history_item.improvement_results],
        "iteration": history_item.iteration,
        "metadata": history_item.metadata,
        "history": []  # Empty history to avoid recursion
    }

def save_to_file(self, file_path: str) -> None:
    """Save the Thought to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(self.to_dict(), f, indent=2)

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "Thought":
    """Create a Thought from a dictionary."""
    # Create the base Thought without history
    thought = cls(
        prompt=data["prompt"],
        retrieved_context=data.get("retrieved_context"),
        generated_text=data.get("generated_text"),
        validation_results=[cls._dict_to_validation_result(r) for r in data.get("validation_results", [])],
        critic_feedback=data.get("critic_feedback"),
        improvement_results=[cls._dict_to_improvement_result(r) for r in data.get("improvement_results", [])],
        iteration=data.get("iteration", 0),
        metadata=data.get("metadata", {})
    )

    # Add history items
    for history_item in data.get("history", []):
        history_thought = cls.from_dict(history_item)
        thought.history.append(history_thought)

    return thought

@classmethod
def load_from_file(cls, file_path: str) -> "Thought":
    """Load a Thought from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return cls.from_dict(data)
```

## Phase 2: Model Implementations

### 2.1 Update OpenAI Model Implementation

**Tasks:**
- Rewrite `sifaka/models/openai.py` to implement the new Model interface
- Add support for using retrieved context in prompt construction
- Update token counting to account for retrieved context

**Implementation Details:**
```python
# sifaka/models/openai.py
from typing import Any, Dict, List, Optional

import openai

from sifaka.models.base import Model
from sifaka.state import Thought

class OpenAIModel(Model):
    """OpenAI model implementation."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **options: Any):
        """Initialize the OpenAI model."""
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.options = options

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        openai.api_key = self.api_key

    def generate(self, thought: Thought, **options: Any) -> Thought:
        """Generate text using a Thought container."""
        # Combine options
        combined_options = {**self.options, **options}

        # Create prompt with retrieved context if available
        prompt = thought.prompt
        if thought.retrieved_context:
            prompt = self._create_augmented_prompt(thought.prompt, thought.retrieved_context)

        # Generate text
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            **combined_options
        )

        # Update thought with generated text
        thought.generated_text = response.choices[0].text.strip()

        return thought

    def _create_augmented_prompt(self, prompt: str, retrieved_context: List[str]) -> str:
        """Create an augmented prompt with retrieved context."""
        augmented_prompt = f"{prompt}\n\nRelevant information:\n"
        for i, document in enumerate(retrieved_context):
            augmented_prompt += f"[{i+1}] {document}\n\n"
        return augmented_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Implementation depends on the tokenizer used
        pass
```

### 2.2 Update Anthropic Model Implementation

**Tasks:**
- Rewrite `sifaka/models/anthropic.py` to implement the new Model interface
- Add support for using retrieved context in prompt construction
- Update token counting to account for retrieved context

### 2.3 Update Other Model Implementations

**Tasks:**
- Rewrite other model implementations to use the new interface
- Ensure consistent handling of retrieved context across all models
- Update factory functions to create models with the new interface

## Phase 3: Retriever Integration

### 3.1 Enhance Retriever Interface

**Tasks:**
- Update `sifaka/retrievers/base.py` to define a consistent interface for retrievers
- Add methods for working with Thoughts directly
- Implement utility functions for retriever selection and configuration

**Implementation Details:**
```python
# sifaka/retrievers/base.py
from abc import ABC, abstractmethod
from typing import List, Optional

from sifaka.state import Thought

class Retriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query."""
        pass

    def retrieve_for_thought(self, thought: Thought) -> Thought:
        """Retrieve relevant documents for a Thought.

        Args:
            thought: The Thought to retrieve documents for.

        Returns:
            The updated Thought with retrieved documents.
        """
        # Default implementation uses the prompt as the query
        thought.retrieved_context = self.retrieve(thought.prompt)
        return thought
```

### 3.2 Implement Retriever Factory

**Tasks:**
- Create factory functions for creating retrievers
- Add support for different retriever types (simple, Elasticsearch, Milvus)
- Implement configuration options for retrievers

## Phase 4: Chain Refactoring

### 4.1 Rewrite Chain Class

**Tasks:**
- Rewrite `sifaka/chain.py` to use Thoughts throughout the execution process
- Add support for retrievers at both model and critic levels
- Implement methods for accessing and manipulating Thoughts

**Implementation Details:**
```python
# sifaka/chain.py
from typing import Any, Callable, Dict, List, Optional, Union

from sifaka.models.base import Model
from sifaka.retrievers.base import Retriever
from sifaka.state import Thought
from sifaka.validators.base import Validator
from sifaka.critics.base import Critic

class Chain:
    """Main orchestrator for text generation, validation, and improvement."""

    def __init__(
        self,
        max_improvement_iterations: int = 3,
    ):
        """Initialize a new Chain instance."""
        self._model: Optional[Model] = None
        self._prompt: Optional[str] = None
        self._validators: List[Validator] = []
        self._critics: List[Critic] = []
        self._retriever: Optional[Retriever] = None
        self._max_improvement_iterations: int = max_improvement_iterations
        self._options: Dict[str, Any] = {}

    def with_model(self, model: Union[str, Model], **options: Any) -> "Chain":
        """Configure the chain with a model."""
        # Implementation details
        return self

    def with_prompt(self, prompt: str) -> "Chain":
        """Set the prompt for the chain."""
        self._prompt = prompt
        return self

    def with_retriever(self, retriever: Union[str, Retriever], **options: Any) -> "Chain":
        """Configure the chain with a retriever."""
        # Implementation details
        return self

    def validate_with(self, validator: Validator) -> "Chain":
        """Add a validator to the chain."""
        self._validators.append(validator)
        return self

    def improve_with(self, critic: Critic) -> "Chain":
        """Add a critic to the chain."""
        self._critics.append(critic)
        return self

    def with_options(self, **options: Any) -> "Chain":
        """Set options for the chain."""
        self._options.update(options)
        return self

    def run(self) -> Thought:
        """Execute the chain and return the result."""
        # Check configuration
        if not self._model:
            raise ValueError("Model not specified")

        if not self._prompt:
            raise ValueError("Prompt not specified")

        # Create initial Thought
        thought = Thought(prompt=self._prompt)

        # Retrieve relevant documents if a retriever is configured
        if self._retriever:
            thought = self._retriever.retrieve_for_thought(thought)

        # Generate text
        thought = self._model.generate(thought)

        # Main improvement loop
        for iteration in range(self._max_improvement_iterations):
            # Create a new Thought for this iteration
            if iteration > 0:
                thought = thought.next_iteration()

            # Validate current text
            self._validate_thought(thought)

            # If validation fails and we should apply improvers on validation failure
            if not thought.validation_passed() and self._options.get(
                "apply_improvers_on_validation_failure", True
            ):
                # Apply critics to improve the text
                thought = self._apply_critics(thought)

            # If validation passes, check if we need to apply improvers for enhancement
            elif thought.validation_passed() and self._critics and self._needs_improvement(thought):
                # Apply critics to improve the text
                thought = self._apply_critics(thought)
                break
            else:
                # Validation passed and no improvement needed
                break

        return thought

    def _validate_thought(self, thought: Thought) -> None:
        """Validate a Thought using all configured validators."""
        # Implementation details
        pass

    def _apply_critics(self, thought: Thought) -> Thought:
        """Apply critics to improve a Thought."""
        # Implementation details
        return thought

    def _needs_improvement(self, thought: Thought) -> bool:
        """Check if a Thought needs improvement."""
        # Implementation details
        return False
```

### 4.2 Update Validator Interface

**Tasks:**
- Update `sifaka/validators/base.py` to work with Thoughts
- Modify validator implementations to update Thoughts directly
- Ensure validators can access the complete context in Thoughts

### 4.3 Update Critic Interface

**Tasks:**
- Update `sifaka/critics/base.py` to work with Thoughts
- Modify critic implementations to update Thoughts directly
- Ensure critics can access retrieved context and validation results

## Phase 5: Advanced Memory Features

### 5.1 Implement Redis-Based Memory Store

**Tasks:**
- Create `sifaka/memory/redis_store.py` for Redis-based persistence
- Implement methods for saving, loading, and querying Thoughts
- Add support for expiration and indexing

**Implementation Details:**
```python
# sifaka/memory/redis_store.py
import json
import uuid
from typing import Any, Dict, List, Optional, Set, Union

import redis

from sifaka.state import Thought

class RedisThoughtStore:
    """Redis-based storage for Thoughts.

    This class provides methods for storing, retrieving, and querying Thoughts
    using Redis as the backend storage.

    Attributes:
        redis_client: The Redis client instance.
        namespace: The namespace to use for Redis keys.
        default_expiration: Default expiration time in seconds (None for no expiration).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        namespace: str = "thought",
        default_expiration: Optional[int] = None,
    ):
        """Initialize the Redis thought store.

        Args:
            redis_url: The URL of the Redis server.
            namespace: The namespace to use for Redis keys.
            default_expiration: Default expiration time in seconds (None for no expiration).
        """
        self.redis_client = redis.from_url(redis_url)
        self.namespace = namespace
        self.default_expiration = default_expiration

    def _get_key(self, key: str) -> str:
        """Get the full Redis key with namespace."""
        return f"{self.namespace}:{key}"

    def _get_index_key(self, index_name: str) -> str:
        """Get the full Redis key for an index."""
        return f"{self.namespace}:index:{index_name}"

    def save(
        self,
        thought: Thought,
        key: Optional[str] = None,
        expiration: Optional[int] = None,
        indexes: Optional[List[str]] = None,
    ) -> str:
        """Save a thought to Redis.

        Args:
            thought: The thought to save.
            key: Optional key to use for the thought. If not provided, a UUID is generated.
            expiration: Expiration time in seconds. If None, uses default_expiration.
            indexes: Optional list of indexes to add the thought to.

        Returns:
            The key used to store the thought.
        """
        # Generate a key if not provided
        if key is None:
            key = str(uuid.uuid4())

        # Get the full Redis key
        full_key = self._get_key(key)

        # Convert thought to JSON
        thought_json = json.dumps(thought.to_dict())

        # Store in Redis
        self.redis_client.set(full_key, thought_json)

        # Set expiration if provided
        if expiration is not None:
            self.redis_client.expire(full_key, expiration)
        elif self.default_expiration is not None:
            self.redis_client.expire(full_key, self.default_expiration)

        # Add to indexes
        if indexes:
            for index in indexes:
                index_key = self._get_index_key(index)
                self.redis_client.sadd(index_key, key)

        # Add to all thoughts index
        self.redis_client.sadd(self._get_index_key("all"), key)

        # Add metadata to search indexes
        if thought.metadata:
            # Add to metadata indexes for efficient querying
            for meta_key, meta_value in thought.metadata.items():
                if isinstance(meta_value, (str, int, float, bool)):
                    meta_index_key = self._get_index_key(f"meta:{meta_key}:{meta_value}")
                    self.redis_client.sadd(meta_index_key, key)

        return key

    def load(self, key: str) -> Optional[Thought]:
        """Load a thought from Redis.

        Args:
            key: The key of the thought to load.

        Returns:
            The loaded thought, or None if not found.
        """
        # Get the full Redis key
        full_key = self._get_key(key)

        # Get from Redis
        thought_json = self.redis_client.get(full_key)

        if thought_json is None:
            return None

        # Convert from JSON
        thought_dict = json.loads(thought_json)

        # Create thought
        return Thought.from_dict(thought_dict)

    def delete(self, key: str) -> bool:
        """Delete a thought from Redis.

        Args:
            key: The key of the thought to delete.

        Returns:
            True if the thought was deleted, False otherwise.
        """
        # Get the full Redis key
        full_key = self._get_key(key)

        # Remove from all indexes
        for index_key in self.redis_client.keys(f"{self.namespace}:index:*"):
            self.redis_client.srem(index_key, key)

        # Delete the thought
        return bool(self.redis_client.delete(full_key))

    def list_keys(self, index: Optional[str] = None) -> List[str]:
        """List all thought keys in Redis.

        Args:
            index: Optional index to list keys from. If None, lists all keys.

        Returns:
            A list of all thought keys.
        """
        if index:
            index_key = self._get_index_key(index)
            return list(self.redis_client.smembers(index_key))
        else:
            return list(self.redis_client.smembers(self._get_index_key("all")))

    def query_by_metadata(self, **metadata: Any) -> List[str]:
        """Query thoughts by metadata.

        Args:
            **metadata: Metadata key-value pairs to query by.

        Returns:
            A list of keys for thoughts matching the query.
        """
        if not metadata:
            return self.list_keys()

        # Get keys for each metadata condition
        keys_sets: List[Set[str]] = []
        for meta_key, meta_value in metadata.items():
            meta_index_key = self._get_index_key(f"meta:{meta_key}:{meta_value}")
            keys = self.redis_client.smembers(meta_index_key)
            if keys:
                keys_sets.append(set(keys))

        # Intersect all sets to get keys matching all conditions
        if keys_sets:
            result_set = set.intersection(*keys_sets)
            return list(result_set)
        else:
            return []
```

### 5.2 Implement Vector Database Integration

#### 5.2.1 Milvus Integration

**Tasks:**
- Create `sifaka/memory/milvus_store.py` for Milvus-based persistence
- Implement methods for saving, loading, and semantically searching Thoughts
- Add support for hybrid search (combining vector and metadata search)

**Implementation Details:**
```python
# sifaka/memory/milvus_store.py
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from sifaka.embeddings.base import EmbeddingModel
from sifaka.state import Thought

class MilvusThoughtStore:
    """Milvus-based storage for Thoughts with semantic search.

    This class provides methods for storing, retrieving, and semantically
    searching Thoughts using Milvus as the backend vector database.

    Attributes:
        embedding_model: The model to use for generating embeddings.
        collection_name: The name of the Milvus collection.
        dimension: The dimension of the embedding vectors.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "thoughts",
        dimension: int = 1536,  # Default for OpenAI embeddings
    ):
        """Initialize the Milvus thought store.

        Args:
            embedding_model: The model to use for generating embeddings.
            host: The Milvus server host.
            port: The Milvus server port.
            collection_name: The name of the Milvus collection.
            dimension: The dimension of the embedding vectors.
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.dimension = dimension

        # Connect to Milvus
        connections.connect(host=host, port=port)

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the thoughts collection exists."""
        if not utility.has_collection(self.collection_name):
            # Define fields for the collection
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]

            # Create collection schema
            schema = CollectionSchema(fields=fields, description="Thought collection")

            # Create collection
            collection = Collection(name=self.collection_name, schema=schema)

            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
            collection.create_index(field_name="embedding", index_params=index_params)

    def _get_collection(self) -> Collection:
        """Get the Milvus collection."""
        return Collection(name=self.collection_name)

    def _generate_embedding(self, thought: Thought) -> List[float]:
        """Generate an embedding for a thought."""
        # Combine prompt and generated text for embedding
        text = thought.prompt
        if thought.generated_text:
            text += "\n\n" + thought.generated_text

        # Generate embedding
        return self.embedding_model.embed(text)

    def save(self, thought: Thought, key: Optional[str] = None) -> str:
        """Save a thought to Milvus.

        Args:
            thought: The thought to save.
            key: Optional key to use for the thought. If not provided, a UUID is generated.

        Returns:
            The key used to store the thought.
        """
        # Generate a key if not provided
        if key is None:
            key = str(uuid.uuid4())

        # Generate embedding
        embedding = self._generate_embedding(thought)

        # Convert thought to dictionary
        thought_dict = thought.to_dict()

        # Convert to JSON for storage
        thought_json = json.dumps(thought_dict)

        # Get collection
        collection = self._get_collection()

        # Insert data
        collection.insert([
            [key],  # id
            [embedding],  # embedding
            [thought_json],  # text
            [thought.metadata],  # metadata
        ])

        return key

    def load(self, key: str) -> Optional[Thought]:
        """Load a thought from Milvus.

        Args:
            key: The key of the thought to load.

        Returns:
            The loaded thought, or None if not found.
        """
        # Get collection
        collection = self._get_collection()

        # Query by key
        collection.load()
        results = collection.query(expr=f'id == "{key}"', output_fields=["text"])
        collection.release()

        if not results:
            return None

        # Parse thought from JSON
        thought_json = results[0]["text"]
        thought_dict = json.loads(thought_json)

        # Create thought
        return Thought.from_dict(thought_dict)

    def search_similar(
        self,
        query: Union[str, Thought],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Thought, float]]:
        """Search for thoughts similar to the query.

        Args:
            query: The query string or Thought to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional metadata filter to apply.

        Returns:
            A list of (thought, similarity) tuples.
        """
        # Generate embedding for the query
        if isinstance(query, Thought):
            embedding = self._generate_embedding(query)
        else:
            embedding = self.embedding_model.embed(query)

        # Get collection
        collection = self._get_collection()

        # Prepare search parameters
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        # Prepare expression for metadata filtering
        expr = None
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')

            if conditions:
                expr = " && ".join(conditions)

        # Execute search
        collection.load()
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["text"],
        )
        collection.release()

        # Process results
        thought_results = []
        for hits in results:
            for hit in hits:
                thought_json = hit.entity.get("text")
                thought_dict = json.loads(thought_json)
                thought = Thought.from_dict(thought_dict)
                thought_results.append((thought, hit.score))

        return thought_results
```

#### 5.2.2 Elasticsearch Integration

**Tasks:**
- Create `sifaka/memory/elasticsearch_store.py` for Elasticsearch-based persistence
- Implement methods for saving, loading, and semantically searching Thoughts
- Add support for hybrid search (combining vector and keyword search)

**Implementation Details:**
```python
# sifaka/memory/elasticsearch_store.py
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from sifaka.embeddings.base import EmbeddingModel
from sifaka.state import Thought

class ElasticsearchThoughtStore:
    """Elasticsearch-based storage for Thoughts with semantic search.

    This class provides methods for storing, retrieving, and semantically
    searching Thoughts using Elasticsearch as the backend.

    Attributes:
        embedding_model: The model to use for generating embeddings.
        es_client: The Elasticsearch client.
        index_name: The name of the Elasticsearch index.
        dimension: The dimension of the embedding vectors.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        hosts: List[str] = ["http://localhost:9200"],
        index_name: str = "thoughts",
        dimension: int = 1536,  # Default for OpenAI embeddings
    ):
        """Initialize the Elasticsearch thought store.

        Args:
            embedding_model: The model to use for generating embeddings.
            hosts: The Elasticsearch hosts.
            index_name: The name of the Elasticsearch index.
            dimension: The dimension of the embedding vectors.
        """
        self.embedding_model = embedding_model
        self.es_client = Elasticsearch(hosts)
        self.index_name = index_name
        self.dimension = dimension

        # Create index if it doesn't exist
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """Ensure the thoughts index exists."""
        if not self.es_client.indices.exists(index=self.index_name):
            # Define mapping for the index
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.dimension,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "prompt": {"type": "text"},
                        "generated_text": {"type": "text"},
                        "thought_json": {"type": "text", "index": False},
                        "metadata": {"type": "object", "dynamic": True},
                    }
                }
            }

            # Create index
            self.es_client.indices.create(index=self.index_name, body=mapping)

    def _generate_embedding(self, thought: Thought) -> List[float]:
        """Generate an embedding for a thought."""
        # Combine prompt and generated text for embedding
        text = thought.prompt
        if thought.generated_text:
            text += "\n\n" + thought.generated_text

        # Generate embedding
        return self.embedding_model.embed(text)

    def save(self, thought: Thought, key: Optional[str] = None) -> str:
        """Save a thought to Elasticsearch.

        Args:
            thought: The thought to save.
            key: Optional key to use for the thought. If not provided, a UUID is generated.

        Returns:
            The key used to store the thought.
        """
        # Generate a key if not provided
        if key is None:
            key = str(uuid.uuid4())

        # Generate embedding
        embedding = self._generate_embedding(thought)

        # Convert thought to dictionary
        thought_dict = thought.to_dict()

        # Convert to JSON for storage
        thought_json = json.dumps(thought_dict)

        # Prepare document
        doc = {
            "embedding": embedding,
            "prompt": thought.prompt,
            "generated_text": thought.generated_text or "",
            "thought_json": thought_json,
            "metadata": thought.metadata,
        }

        # Index document
        self.es_client.index(index=self.index_name, id=key, document=doc)

        return key

    def load(self, key: str) -> Optional[Thought]:
        """Load a thought from Elasticsearch.

        Args:
            key: The key of the thought to load.

        Returns:
            The loaded thought, or None if not found.
        """
        try:
            # Get document
            result = self.es_client.get(index=self.index_name, id=key)

            # Parse thought from JSON
            thought_json = result["_source"]["thought_json"]
            thought_dict = json.loads(thought_json)

            # Create thought
            return Thought.from_dict(thought_dict)
        except Exception:
            return None

    def search_similar(
        self,
        query: Union[str, Thought],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_text_match: bool = True,
    ) -> List[Tuple[Thought, float]]:
        """Search for thoughts similar to the query.

        Args:
            query: The query string or Thought to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional metadata filter to apply.
            include_text_match: Whether to include text matching in the search.

        Returns:
            A list of (thought, similarity) tuples.
        """
        # Generate embedding for the query
        if isinstance(query, Thought):
            embedding = self._generate_embedding(query)
            query_text = query.prompt
            if query.generated_text:
                query_text += "\n\n" + query.generated_text
        else:
            embedding = self.embedding_model.embed(query)
            query_text = query

        # Prepare search query
        search_query: Dict[str, Any] = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": embedding},
                                },
                            }
                        }
                    ]
                }
            },
        }

        # Add text matching if enabled
        if include_text_match and query_text:
            search_query["query"]["bool"]["should"] = [
                {"match": {"prompt": {"query": query_text, "boost": 0.5}}},
                {"match": {"generated_text": {"query": query_text, "boost": 0.3}}},
            ]

        # Add metadata filter if provided
        if metadata_filter:
            filter_clauses = []
            for key, value in metadata_filter.items():
                filter_clauses.append({"term": {f"metadata.{key}": value}})

            if filter_clauses:
                search_query["query"]["bool"]["filter"] = filter_clauses

        # Execute search
        results = self.es_client.search(index=self.index_name, body=search_query)

        # Process results
        thought_results = []
        for hit in results["hits"]["hits"]:
            thought_json = hit["_source"]["thought_json"]
            thought_dict = json.loads(thought_json)
            thought = Thought.from_dict(thought_dict)
            thought_results.append((thought, hit["_score"]))

        return thought_results
```

### 5.3 Implement Hybrid Memory Architecture

**Tasks:**
- Create `sifaka/memory/hybrid_store.py` for combining multiple storage mechanisms
- Implement memory management strategies (short-term vs. long-term)
- Add support for different types of memory (factual, episodic, etc.)

**Implementation Details:**
```python
# sifaka/memory/hybrid_store.py
from typing import Any, Dict, List, Optional, Tuple, Union

from sifaka.memory.redis_store import RedisThoughtStore
from sifaka.memory.elasticsearch_store import ElasticsearchThoughtStore
from sifaka.memory.milvus_store import MilvusThoughtStore
from sifaka.state import Thought

class HybridMemoryStore:
    """Hybrid memory store combining multiple storage mechanisms.

    This class provides a unified interface for storing and retrieving Thoughts
    using multiple storage backends, with different memory management strategies.

    Attributes:
        short_term_store: The store for short-term memory (Redis).
        long_term_store: The store for long-term memory (vector database).
        short_term_expiration: Expiration time for short-term memory in seconds.
    """

    def __init__(
        self,
        short_term_store: RedisThoughtStore,
        long_term_store: Union[ElasticsearchThoughtStore, MilvusThoughtStore],
        short_term_expiration: int = 3600,  # 1 hour
    ):
        """Initialize the hybrid memory store.

        Args:
            short_term_store: The store for short-term memory (Redis).
            long_term_store: The store for long-term memory (vector database).
            short_term_expiration: Expiration time for short-term memory in seconds.
        """
        self.short_term_store = short_term_store
        self.long_term_store = long_term_store
        self.short_term_expiration = short_term_expiration

    def save(
        self,
        thought: Thought,
        memory_type: str = "both",
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a thought to memory.

        Args:
            thought: The thought to save.
            memory_type: Where to save the thought ("short_term", "long_term", or "both").
            key: Optional key to use for the thought.
            metadata: Optional metadata to add to the thought.

        Returns:
            The key used to store the thought.
        """
        # Add metadata if provided
        if metadata:
            thought.metadata.update(metadata)

        # Generate a key if not provided
        if key is None:
            import uuid
            key = str(uuid.uuid4())

        # Save to short-term memory
        if memory_type in ("short_term", "both"):
            self.short_term_store.save(
                thought,
                key=key,
                expiration=self.short_term_expiration,
                indexes=["recent"],
            )

        # Save to long-term memory
        if memory_type in ("long_term", "both"):
            self.long_term_store.save(thought, key=key)

        return key

    def load(self, key: str, fallback: bool = True) -> Optional[Thought]:
        """Load a thought from memory.

        Args:
            key: The key of the thought to load.
            fallback: Whether to fall back to long-term memory if not found in short-term.

        Returns:
            The loaded thought, or None if not found.
        """
        # Try short-term memory first
        thought = self.short_term_store.load(key)

        # Fall back to long-term memory if not found and fallback is enabled
        if thought is None and fallback:
            thought = self.long_term_store.load(key)

        return thought

    def search_similar(
        self,
        query: Union[str, Thought],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Thought, float]]:
        """Search for thoughts similar to the query.

        Args:
            query: The query string or Thought to search for.
            limit: Maximum number of results to return.
            metadata_filter: Optional metadata filter to apply.

        Returns:
            A list of (thought, similarity) tuples.
        """
        # Search in long-term memory (vector database)
        return self.long_term_store.search_similar(
            query=query,
            limit=limit,
            metadata_filter=metadata_filter,
        )

    def get_recent(self, limit: int = 10) -> List[Thought]:
        """Get recent thoughts from short-term memory.

        Args:
            limit: Maximum number of thoughts to return.

        Returns:
            A list of recent thoughts.
        """
        # Get keys from the "recent" index
        keys = self.short_term_store.list_keys(index="recent")

        # Limit the number of keys
        keys = keys[:limit]

        # Load thoughts
        thoughts = []
        for key in keys:
            thought = self.short_term_store.load(key)
            if thought:
                thoughts.append(thought)

        return thoughts

    def promote_to_long_term(self, key: str) -> bool:
        """Promote a thought from short-term to long-term memory.

        Args:
            key: The key of the thought to promote.

        Returns:
            True if the thought was promoted, False otherwise.
        """
        # Load from short-term memory
        thought = self.short_term_store.load(key)

        if thought is None:
            return False

        # Save to long-term memory
        self.long_term_store.save(thought, key=key)

        return True

    def forget(self, key: str, memory_type: str = "both") -> bool:
        """Forget a thought from memory.

        Args:
            key: The key of the thought to forget.
            memory_type: Which memory to forget from ("short_term", "long_term", or "both").

        Returns:
            True if the thought was forgotten, False otherwise.
        """
        result = True

        # Forget from short-term memory
        if memory_type in ("short_term", "both"):
            result = result and self.short_term_store.delete(key)

        # Forget from long-term memory (if supported)
        if memory_type in ("long_term", "both") and hasattr(self.long_term_store, "delete"):
            result = result and self.long_term_store.delete(key)

        return result
```

## Phase 6: Documentation and Examples

### 6.1 Update API Documentation

**Tasks:**
- Update docstrings throughout the codebase
- Create new API reference documentation
- Document the Thought container and its usage

### 6.2 Create Example Applications

**Tasks:**
- Create examples demonstrating basic usage
- Create examples showing advanced memory features
- Create examples of agents with different memory models

### 6.3 Write Migration Guide

**Tasks:**
- Document the changes from the old architecture to the new one
- Provide guidance for migrating existing code
- Highlight new capabilities and best practices

## Timeline and Priorities

1. **Week 1-2**: Implement core infrastructure (Thought container, Model interface)
2. **Week 3-4**: Update model implementations and retriever integration
3. **Week 5-6**: Rewrite Chain class and update component interfaces
4. **Week 7-8**: Implement advanced memory features
5. **Week 9-10**: Create documentation and examples

## Success Criteria

1. All components use the Thought container consistently
2. Retrievers are available to both models and critics
3. The Model interface uses Thoughts directly
4. Memory persistence works with file, Redis, and vector databases
5. The architecture is clean and maintainable
6. Documentation and examples demonstrate the new capabilities effectively
