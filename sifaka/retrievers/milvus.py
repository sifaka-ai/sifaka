"""Milvus vector database retriever using MCP.

This module provides a MilvusRetriever that uses MCP (Model Context Protocol)
to communicate with a Milvus MCP server for semantic document retrieval.

Example:
    ```python
    from sifaka.retrievers.milvus import MilvusRetriever
    from sifaka.retrievers.mcp_base import MCPServerConfig, MCPTransportType

    # Create MCP server configuration
    config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/milvus"
    )

    # Create retriever
    retriever = MilvusRetriever(
        mcp_config=config,
        collection_name="documents",
        embedding_model="BAAI/bge-m3"
    )

    # Add documents
    retriever.add_document("doc1", "This is about artificial intelligence.")

    # Retrieve similar documents
    results = retriever.retrieve("Tell me about AI")
    ```
"""

import asyncio
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought
from sifaka.retrievers.base import BaseMCPRetriever, MCPServerConfig, MCPRequest
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class MilvusRetriever:
    """Milvus vector database retriever using MCP.

    This retriever uses MCP (Model Context Protocol) for all communication with
    the Milvus database, providing standardized communication, better error
    handling, and improved scalability.

    Attributes:
        collection_name: Name of the Milvus collection.
        embedding_model: Name or path of the embedding model.
        dimension: Dimension of the embedding vectors.
        max_results: Maximum number of documents to return.
        mcp_retriever: Internal MCP-based retriever.
    """

    def __init__(
        self,
        mcp_config: MCPServerConfig,
        collection_name: str = "sifaka_documents",
        embedding_model: Optional[str] = None,
        dimension: int = 384,
        max_results: int = 3,
        schema_config: Optional[Dict[str, Any]] = None,
        index_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Milvus retriever with MCP backend.

        Args:
            mcp_config: MCP server configuration.
            collection_name: Name of the Milvus collection.
            embedding_model: Name/path of embedding model.
            dimension: Dimension of the embedding vectors.
            max_results: Maximum number of documents to return.
            schema_config: Custom schema configuration.
            index_config: Custom index configuration.
        """
        # Store configuration
        self.collection_name = collection_name
        self.embedding_model = embedding_model or "BAAI/bge-m3"
        self.dimension = dimension
        self.max_results = max_results
        self.schema_config = schema_config or {}
        self.index_config = index_config or {}

        # Create internal MCP-based retriever
        self.mcp_retriever = BaseMCPRetriever(mcp_config, max_results)
        self._loop = None

        logger.info(f"Initialized MilvusRetriever with MCP backend: {mcp_config.name}")

    def _get_loop(self):
        """Get or create event loop for async operations."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

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
            message_prefix="Failed to add document to Milvus via MCP",
        ):
            # Add collection context to metadata
            enhanced_metadata = {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                **(metadata or {}),
            }

            # Use async operation via event loop
            loop = self._get_loop()
            loop.run_until_complete(
                self.mcp_retriever.add_document(doc_id, text, enhanced_metadata)
            )

            logger.debug(f"Added document {doc_id} to Milvus collection: {self.collection_name}")

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
            message_prefix="Failed to retrieve documents from Milvus via MCP",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            # Use async operation via event loop
            loop = self._get_loop()
            document_texts = loop.run_until_complete(self.mcp_retriever.retrieve(query))

            logger.debug(f"Retrieved {len(document_texts)} documents from Milvus")
            return document_texts

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
            message_prefix="Failed to retrieve documents for thought from Milvus via MCP",
        ):
            # Use async operation via event loop
            loop = self._get_loop()
            enhanced_thought = loop.run_until_complete(
                self.mcp_retriever.retrieve_for_thought(thought, is_pre_generation)
            )

            # Enhance document metadata with Milvus-specific info
            context_docs = (
                enhanced_thought.pre_generation_context
                if is_pre_generation
                else enhanced_thought.post_generation_context
            )

            for doc in context_docs:
                doc.metadata.update(
                    {
                        "source": "milvus_mcp",
                        "collection": self.collection_name,
                        "embedding_model": self.embedding_model,
                        "dimension": self.dimension,
                    }
                )

            return enhanced_thought

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        with error_context(
            component="MilvusRetriever",
            operation="clear collection",
            error_class=RetrieverError,
            message_prefix="Failed to clear Milvus collection via MCP",
        ):

            async def _clear_collection():
                if not self.mcp_retriever.mcp_client._connected:
                    await self.mcp_retriever.mcp_client.connect()

                request = MCPRequest(
                    method="clear_collection", params={"collection_name": self.collection_name}
                )

                response = await self.mcp_retriever.mcp_client.transport.send_request(request)
                success = response.result is not None and response.result.get("success", False)

                if success:
                    logger.info(f"Cleared Milvus collection: {self.collection_name}")
                else:
                    logger.error(f"Failed to clear collection: {response.error}")

                return success

            loop = self._get_loop()
            loop.run_until_complete(_clear_collection())

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        with error_context(
            component="MilvusRetriever",
            operation="get collection stats",
            error_class=RetrieverError,
            message_prefix="Failed to get Milvus collection stats via MCP",
        ):

            async def _get_stats():
                if not self.mcp_retriever.mcp_client._connected:
                    await self.mcp_retriever.mcp_client.connect()

                request = MCPRequest(
                    method="get_collection_stats", params={"collection_name": self.collection_name}
                )

                response = await self.mcp_retriever.mcp_client.transport.send_request(request)

                if response.result:
                    return response.result
                else:
                    return {
                        "collection_name": self.collection_name,
                        "error": response.error or "Unknown error",
                        "embedding_model": self.embedding_model,
                        "dimension": self.dimension,
                    }

            loop = self._get_loop()
            return loop.run_until_complete(_get_stats())

    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            loop = self._get_loop()
            loop.run_until_complete(self.mcp_retriever.disconnect())
            logger.info(f"Disconnected from Milvus MCP server")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus MCP server: {e}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.disconnect()
        except Exception:
            pass  # Ignore errors during cleanup
