"""Milvus vector database retriever implementation.

This module provides a MilvusRetriever that uses Milvus vector database
for semantic document retrieval. It supports both Milvus Lite (embedded)
and Milvus server connections.

Example:
    ```python
    from sifaka.retrievers.milvus import MilvusRetriever

    # Create retriever with Milvus Lite (embedded)
    retriever = MilvusRetriever(
        collection_name="documents",
        embedding_model="BAAI/bge-m3"
    )

    # Add documents
    retriever.add_document("doc1", "This is about artificial intelligence.")
    retriever.add_document("doc2", "This is about machine learning.")

    # Retrieve similar documents
    results = retriever.retrieve("Tell me about AI")
    ```
"""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union
import logging

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
        MilvusException,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    MilvusException = Exception  # Fallback for type hints

# Optional embedding functions - users can provide their own
try:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    PYMILVUS_MODEL_AVAILABLE = True
except ImportError:
    PYMILVUS_MODEL_AVAILABLE = False
    BGEM3EmbeddingFunction = None

from sifaka.core.thought import Document, Thought
from sifaka.retrievers.vector_db_base import BaseVectorDBRetriever
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class MilvusRetriever(BaseVectorDBRetriever):
    """Milvus vector database retriever for semantic search.

    This retriever uses Milvus vector database to store document embeddings
    and perform semantic similarity search. It supports both Milvus Lite
    (embedded) and Milvus server connections.

    Attributes:
        collection_name: Name of the Milvus collection.
        embedding_model: Name or path of the embedding model.
        dimension: Dimension of the embedding vectors.
        max_results: Maximum number of documents to return.
        connection_alias: Alias for the Milvus connection.
        collection: The Milvus collection object.
        embedding_function: Function to generate embeddings.
    """

    def __init__(
        self,
        collection_name: str = "sifaka_documents",
        embedding_model: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        dimension: int = 384,  # Common default for many models
        max_results: int = 3,
        connection_alias: str = "default",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        schema_config: Optional[Dict[str, Any]] = None,
        index_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        **connection_kwargs: Any,
    ):
        """Initialize the Milvus retriever.

        Args:
            collection_name: Name of the Milvus collection.
            embedding_model: Name/path of embedding model (for built-in functions).
            embedding_function: Custom embedding function (takes precedence).
            dimension: Dimension of the embedding vectors.
            max_results: Maximum number of documents to return.
            connection_alias: Alias for the Milvus connection.
            uri: URI for Milvus connection (None for Milvus Lite).
            token: Token for Milvus connection authentication.
            schema_config: Custom schema configuration.
            index_config: Custom index configuration.
            device: Device for embedding computation ("cpu", "cuda", etc.).
            **connection_kwargs: Additional connection parameters.
        """
        if not MILVUS_AVAILABLE:
            raise RetrieverError(
                "pymilvus is not installed. Please install it with: pip install 'pymilvus[model]'"
            )

        # Call parent constructor
        super().__init__(
            collection_name=collection_name,
            embedding_model=embedding_model or "default",
            dimension=dimension,
            max_results=max_results,
        )

        self.connection_alias = connection_alias
        self.device = device
        self.schema_config = schema_config or {}
        self.index_config = index_config or {}

        # Store embedding configuration
        self.custom_embedding_function = embedding_function
        self.embedding_model = embedding_model

        # Initialize connection and collection
        self._connect(uri, token, **connection_kwargs)
        self._initialize_collection()

        # Defer embedding function initialization until first use
        self.embedding_function = None
        self._embedding_initialized = False

    def _connect(
        self, uri: Optional[str] = None, token: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Connect to Milvus database."""
        with error_context(
            component="MilvusRetriever",
            operation="connection",
            error_class=RetrieverError,
            message_prefix="Failed to connect to Milvus",
        ):
            connection_params = {"alias": self.connection_alias, **kwargs}

            if uri:
                connection_params["uri"] = uri
            else:
                # Use Milvus Lite (embedded)
                connection_params["uri"] = "./milvus_lite.db"

            if token:
                connection_params["token"] = token

            connections.connect(**connection_params)
            logger.info(f"Connected to Milvus with alias: {self.connection_alias}")

    def _initialize_embedding_function(self) -> None:
        """Initialize the embedding function."""
        with error_context(
            component="MilvusRetriever",
            operation="embedding initialization",
            error_class=RetrieverError,
            message_prefix="Failed to initialize embedding function",
        ):
            # Use custom embedding function if provided
            if self.custom_embedding_function:
                self.embedding_function = self.custom_embedding_function
                logger.info("Using custom embedding function")
                return

            # Try to use built-in embedding functions if available
            if self.embedding_model and PYMILVUS_MODEL_AVAILABLE and BGEM3EmbeddingFunction:
                try:
                    # Use BGE-M3 or other pymilvus embedding functions
                    self.embedding_function = BGEM3EmbeddingFunction(
                        model_name=self.embedding_model,
                        use_fp16=False,
                        device=self.device,
                    )
                    logger.info(f"Initialized pymilvus embedding: {self.embedding_model}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize pymilvus embedding: {e}")

            # Fallback to hash-based embedding for testing
            self.embedding_function = None
            logger.info("Using fallback hash-based embeddings")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Initialize embedding function on first use
        if not self._embedding_initialized:
            self._initialize_embedding_function()
            self._embedding_initialized = True

        if self.embedding_function:
            try:
                # Use BGE-M3 embedding
                embeddings = self.embedding_function.encode_documents([text])
                return embeddings["dense"][0].tolist()
            except Exception as e:
                logger.warning(f"BGE-M3 embedding failed: {e}, using fallback")

        # Fallback: simple hash-based embedding (for testing only)
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float vector of specified dimension
        embedding = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            embedding.append(float(hash_bytes[byte_idx]) / 255.0)

        return embedding

    def _initialize_collection(self) -> None:
        """Initialize the Milvus collection."""
        with error_context(
            component="MilvusRetriever",
            operation="collection initialization",
            error_class=RetrieverError,
            message_prefix="Failed to initialize collection",
        ):
            # Use custom schema if provided, otherwise use default
            if self.schema_config.get("fields"):
                fields = self.schema_config["fields"]
            else:
                # Default schema
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        max_length=self.schema_config.get("id_max_length", 512),
                        is_primary=True,
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=self.schema_config.get("text_max_length", 65535),
                    ),
                    FieldSchema(
                        name="metadata",
                        dtype=DataType.VARCHAR,
                        max_length=self.schema_config.get("metadata_max_length", 65535),
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.dimension,
                    ),
                ]

            description = self.schema_config.get(
                "description", f"{self.collection_name} collection for semantic search"
            )
            schema = CollectionSchema(fields=fields, description=description)

            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name, using=self.connection_alias):
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using=self.connection_alias,
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                self.collection = Collection(
                    name=self.collection_name,
                    using=self.connection_alias,
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")

            # Create index for vector field with configurable parameters
            default_index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            index_params = {**default_index_params, **self.index_config}

            # Check if index exists for the embedding field specifically
            try:
                self.collection.describe_index(field_name="embedding")
                logger.info("Vector index already exists")
            except Exception:
                # Index doesn't exist, create it
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logger.info("Created vector index")

            # Load collection
            self.collection.load()
            logger.info("Collection loaded and ready")

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the vector database.

        Args:
            doc_id: Unique identifier for the document.
            text: The document text.
            metadata: Optional metadata for the document.
        """
        with error_context(
            component="MilvusRetriever",
            operation="add document",
            error_class=RetrieverError,
            message_prefix="Failed to add document to vector database",
        ):
            # Generate embedding
            embedding = self._generate_embedding(text)

            # Prepare data
            data = [
                [doc_id],  # id
                [text],  # text
                [json.dumps(metadata or {})],  # metadata
                [embedding],  # embedding
            ]

            # Insert into collection
            self.collection.insert(data)
            self.collection.flush()

            logger.debug(f"Added document {doc_id} to vector database")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="MilvusRetriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Search parameters (configurable via index_config)
            default_search_params = {
                "metric_type": self.index_config.get("metric_type", "COSINE"),
                "params": {"nprobe": 10},
            }
            search_params = self.index_config.get("search_params", default_search_params)

            # Perform vector search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=self.max_results,
                output_fields=["text", "metadata"],
            )

            # Extract document texts
            documents = []
            for hit in results[0]:
                documents.append(hit.entity.get("text"))

            logger.debug(f"Retrieved {len(documents)} documents")
            return documents

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents and add them to a thought.

        Args:
            thought: The thought to add documents to.
            is_pre_generation: Whether this is pre-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="MilvusRetriever",
            operation="retrieval for thought",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents for thought",
        ):
            # Determine the query based on whether this is pre or post-generation
            if is_pre_generation:
                query = thought.prompt
            else:
                # For post-generation, use both the prompt and the generated text
                query = f"{thought.prompt}\n\n{thought.text}"

            # Retrieve documents
            document_texts = self.retrieve(query)

            # Convert to Document objects
            documents = [
                Document(
                    text=text,
                    metadata={
                        "source": "vector_db",
                        "query": query,
                        "collection": self.collection_name,
                    },
                    score=1.0 - (i * 0.1),  # Simple scoring based on rank
                )
                for i, text in enumerate(document_texts)
            ]

            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        with error_context(
            component="MilvusRetriever",
            operation="clear collection",
            error_class=RetrieverError,
            message_prefix="Failed to clear collection",
        ):
            # Delete all entities
            self.collection.delete(expr="id != ''")
            self.collection.flush()
            logger.info(f"Cleared collection: {self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        with error_context(
            component="MilvusRetriever",
            operation="get collection stats",
            error_class=RetrieverError,
            message_prefix="Failed to get collection statistics",
        ):
            # Get collection statistics using the correct API
            try:
                stats = self.collection.num_entities
                row_count = stats
            except Exception:
                # Fallback if num_entities doesn't work
                row_count = 0

            return {
                "collection_name": self.collection_name,
                "row_count": row_count,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
            }

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        try:
            connections.disconnect(alias=self.connection_alias)
            logger.info(f"Disconnected from Milvus: {self.connection_alias}")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.disconnect()
        except Exception:
            pass  # Ignore errors during cleanup
