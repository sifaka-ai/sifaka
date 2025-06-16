"""Milvus persistence implementation for Sifaka.

This module provides Milvus-based storage for vector search and semantic
similarity operations on thoughts.

Note: This is a placeholder implementation. To use Milvus storage:
1. Install: uv add pymilvus
2. Set up Milvus server (or use Zilliz Cloud)
3. Configure embedding model for text vectorization
"""

import json
from typing import Any, Dict, List, Optional

from .base import SifakaBasePersistence
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MilvusPersistence(SifakaBasePersistence):
    """Milvus-based persistence for vector search and semantic similarity.

    This implementation provides:
    - Vector storage for semantic search
    - Similarity-based thought retrieval
    - Scalable vector indexing
    - Hybrid search (vector + metadata filtering)
    - Real-time vector updates

    Collection Schema:
    ```python
    {
        "id": "VARCHAR",           # Thought ID
        "key_prefix": "VARCHAR",   # Storage key prefix
        "embedding": "FLOAT_VECTOR",  # Text embedding (768 or 1536 dimensions)
        "prompt": "VARCHAR",       # Original prompt text
        "final_text": "VARCHAR",   # Final generated text
        "conversation_id": "VARCHAR",  # Conversation grouping
        "techniques": "JSON",      # Applied techniques
        "created_at": "INT64",     # Unix timestamp
        "metadata": "JSON"         # Additional metadata
    }
    ```

    Prerequisites:
    - Milvus 2.3+ server or Zilliz Cloud
    - pymilvus Python library
    - Embedding model (OpenAI, Sentence Transformers, etc.)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "sifaka_thoughts",
        embedding_model: str = "text-embedding-ada-002",
        embedding_dim: int = 1536,
        key_prefix: str = "sifaka",
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
    ):
        """Initialize Milvus persistence.

        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the Milvus collection
            embedding_model: Model for text embeddings
            embedding_dim: Dimension of embeddings
            key_prefix: Prefix for storage keys
            index_type: Vector index type (IVF_FLAT, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
        """
        super().__init__(key_prefix)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric_type = metric_type

        self.connection = None
        self.collection = None

        logger.debug(f"Initialized MilvusPersistence with collection '{collection_name}'")

    async def _ensure_connection(self):
        """Ensure Milvus connection and collection are initialized."""
        if self.connection is None:
            try:
                from pymilvus import (
                    connections,
                    Collection,
                    FieldSchema,
                    CollectionSchema,
                    DataType,
                )

                # Connect to Milvus
                connections.connect(alias="default", host=self.host, port=self.port)
                self.connection = connections.get_connection("default")

                # Define collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema(name="key_prefix", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(
                        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim
                    ),
                    FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name="final_text", dtype=DataType.VARCHAR, max_length=50000),
                    FieldSchema(name="conversation_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="techniques", dtype=DataType.JSON),
                    FieldSchema(name="created_at", dtype=DataType.INT64),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]

                schema = CollectionSchema(
                    fields=fields, description="Sifaka thoughts with vector embeddings"
                )

                # Create or get collection
                if not Collection.exists(self.collection_name):
                    self.collection = Collection(name=self.collection_name, schema=schema)

                    # Create vector index
                    index_params = {
                        "index_type": self.index_type,
                        "metric_type": self.metric_type,
                        "params": {"nlist": 1024},  # Adjust based on data size
                    }
                    self.collection.create_index(field_name="embedding", index_params=index_params)

                    logger.info(f"Created Milvus collection '{self.collection_name}' with index")
                else:
                    self.collection = Collection(self.collection_name)

                # Load collection into memory
                self.collection.load()

                logger.debug("Milvus connection and collection ready")

            except ImportError:
                raise ImportError(
                    "pymilvus is required for Milvus persistence. " "Install with: uv add pymilvus"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model."""
        try:
            if self.embedding_model.startswith("text-embedding"):
                # OpenAI embedding
                import openai

                response = await openai.embeddings.create(model=self.embedding_model, input=text)
                return response.data[0].embedding
            else:
                # Sentence Transformers or other models
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.embedding_model)
                embedding = model.encode(text)
                return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dim

    async def _store_raw(self, key: str, data: str) -> None:
        """Store raw data in Milvus with vector embedding."""
        await self._ensure_connection()

        try:
            # Parse the key to extract thought ID
            key_parts = key.split(":")

            if len(key_parts) >= 3 and key_parts[1] == "thought":
                # Only store regular thoughts in Milvus (not snapshots)
                thought_id = key_parts[2]
                await self._store_thought_vector(thought_id, data)
            else:
                # Milvus is primarily for vector storage, skip other data types
                logger.debug(f"Skipping non-thought data for Milvus: {key}")

        except Exception as e:
            logger.error(f"Failed to store Milvus data for key {key}: {e}")
            raise

    async def _store_thought_vector(self, thought_id: str, data: str) -> None:
        """Store thought data with vector embedding in Milvus."""
        thought_data = json.loads(data)

        # Extract text for embedding
        text_for_embedding = ""
        if thought_data.get("final_text"):
            text_for_embedding = thought_data["final_text"]
        elif thought_data.get("current_text"):
            text_for_embedding = thought_data["current_text"]
        else:
            text_for_embedding = thought_data.get("prompt", "")

        # Generate embedding
        embedding = await self._get_embedding(text_for_embedding)

        # Prepare data for insertion
        entities = [
            {
                "id": thought_id,
                "key_prefix": self.key_prefix,
                "embedding": embedding,
                "prompt": thought_data.get("prompt", "")[:10000],  # Truncate if too long
                "final_text": (thought_data.get("final_text") or "")[:50000],
                "conversation_id": thought_data.get("conversation_id", ""),
                "techniques": thought_data.get("techniques_applied", []),
                "created_at": int(thought_data.get("created_at", 0)),
                "metadata": {
                    "iteration": thought_data.get("iteration", 0),
                    "max_iterations": thought_data.get("max_iterations", 3),
                    "validations_count": len(thought_data.get("validations", [])),
                    "critiques_count": len(thought_data.get("critiques", [])),
                },
            }
        ]

        # Insert or upsert data
        self.collection.insert(entities)
        self.collection.flush()  # Ensure data is persisted

    async def _retrieve_raw(self, key: str) -> Optional[str]:
        """Retrieve raw data from Milvus (limited functionality)."""
        await self._ensure_connection()

        try:
            key_parts = key.split(":")

            if len(key_parts) >= 3 and key_parts[1] == "thought":
                thought_id = key_parts[2]
                return await self._retrieve_thought_data(thought_id)
            else:
                # Milvus doesn't store non-thought data
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve Milvus data for key {key}: {e}")
            return None

    async def _retrieve_thought_data(self, thought_id: str) -> Optional[str]:
        """Retrieve thought data from Milvus by ID."""
        expr = f'id == "{thought_id}" && key_prefix == "{self.key_prefix}"'

        results = self.collection.query(
            expr=expr,
            output_fields=[
                "id",
                "prompt",
                "final_text",
                "conversation_id",
                "techniques",
                "created_at",
                "metadata",
            ],
        )

        if results:
            result = results[0]
            # Reconstruct basic thought data (without full audit trail)
            thought_data = {
                "id": result["id"],
                "prompt": result["prompt"],
                "final_text": result["final_text"],
                "conversation_id": result["conversation_id"],
                "techniques_applied": result["techniques"],
                "created_at": result["created_at"],
                "metadata": result["metadata"],
            }
            return json.dumps(thought_data)

        return None

    async def _delete_raw(self, key: str) -> bool:
        """Delete data from Milvus."""
        await self._ensure_connection()

        try:
            key_parts = key.split(":")

            if len(key_parts) >= 3 and key_parts[1] == "thought":
                thought_id = key_parts[2]
                expr = f'id == "{thought_id}" && key_prefix == "{self.key_prefix}"'
                self.collection.delete(expr)
                self.collection.flush()
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete Milvus data for key {key}: {e}")
            return False

    async def _list_keys(self, pattern: str) -> List[str]:
        """List all keys from Milvus (limited functionality)."""
        await self._ensure_connection()

        try:
            if f"{self.key_prefix}:thought:" in pattern:
                # Query all thought IDs
                expr = f'key_prefix == "{self.key_prefix}"'
                results = self.collection.query(expr=expr, output_fields=["id"])

                return [f"{self.key_prefix}:thought:{result['id']}" for result in results]

            return []

        except Exception as e:
            logger.error(f"Failed to list Milvus keys: {e}")
            return []

    async def semantic_search(
        self, query_text: str, limit: int = 10, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        await self._ensure_connection()

        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query_text)

            # Search parameters
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}

            # Perform vector search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=f'key_prefix == "{self.key_prefix}"',
                output_fields=["id", "prompt", "final_text", "conversation_id", "techniques"],
            )

            # Filter by score threshold and format results
            filtered_results = []
            for hit in results[0]:
                if hit.score >= score_threshold:
                    filtered_results.append(
                        {
                            "id": hit.entity.get("id"),
                            "prompt": hit.entity.get("prompt"),
                            "final_text": hit.entity.get("final_text"),
                            "conversation_id": hit.entity.get("conversation_id"),
                            "techniques": hit.entity.get("techniques"),
                            "similarity_score": hit.score,
                        }
                    )

            return filtered_results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []

    async def close(self) -> None:
        """Close Milvus connection."""
        if self.connection:
            from pymilvus import connections

            connections.disconnect("default")
            self.connection = None
            self.collection = None
            logger.debug("Milvus connection closed")
