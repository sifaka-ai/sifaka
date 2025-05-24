"""Base MCP (Model Context Protocol) infrastructure for Sifaka retrievers.

This module provides the base classes and abstractions for implementing
MCP-based retrievers in Sifaka. MCP provides a standardized protocol for
communication between AI applications and data sources.

The MCP implementation in Sifaka focuses on retrieval operations and provides:
- Standardized communication protocol
- Error handling and fallback mechanisms
- Multi-server composition capabilities
- Transport abstraction (WebSocket, HTTP, etc.)

Example:
    ```python
    from sifaka.retrievers.mcp_base import MCPClient, MCPServerConfig

    # Create MCP client
    config = MCPServerConfig(
        name="redis-server",
        transport_type="websocket",
        url="ws://localhost:8080/mcp"
    )
    client = MCPClient(config)

    # Query documents
    documents = await client.query("artificial intelligence")
    ```
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

from sifaka.core.thought import Document, Thought
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class MCPTransportType(Enum):
    """Supported MCP transport types."""

    WEBSOCKET = "websocket"
    HTTP = "http"
    STDIO = "stdio"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        name: Human-readable name for the server.
        transport_type: Type of transport to use.
        url: Server URL or connection string.
        timeout: Connection timeout in seconds.
        retry_attempts: Number of retry attempts for failed requests.
        retry_delay: Delay between retry attempts in seconds.
        auth_token: Optional authentication token.
        headers: Additional headers for HTTP transport.
        capabilities: Server capabilities to request.
    """

    name: str
    transport_type: MCPTransportType
    url: str
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    auth_token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=lambda: ["query", "search"])


@dataclass
class MCPRequest:
    """MCP request message.

    Attributes:
        method: The method to call on the server.
        params: Parameters for the method call.
        id: Unique request identifier.
    """

    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class MCPResponse:
    """MCP response message.

    Attributes:
        result: The result data from the server.
        error: Error information if the request failed.
        id: Request identifier this response corresponds to.
    """

    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    def __init__(self, config: MCPServerConfig):
        """Initialize the transport.

        Args:
            config: Server configuration.
        """
        self.config = config
        self.connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the server."""
        ...

    @abstractmethod
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send a request and wait for response.

        Args:
            request: The request to send.

        Returns:
            The response from the server.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if the server is healthy, False otherwise.
        """
        ...


class WebSocketTransport(MCPTransport):
    """WebSocket transport implementation for MCP."""

    def __init__(self, config: MCPServerConfig):
        """Initialize WebSocket transport.

        Args:
            config: Server configuration.
        """
        super().__init__(config)
        self.websocket = None
        self._request_counter = 0

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if not WEBSOCKETS_AVAILABLE:
            raise RetrieverError(
                "WebSocket transport requires 'websockets' package. "
                "Install with: pip install websockets>=11.0.0"
            )

        with error_context(
            component="WebSocketTransport",
            operation="connection",
            error_class=RetrieverError,
            message_prefix="Failed to connect to MCP server via WebSocket",
        ):
            # Prepare connection headers
            headers = self.config.headers.copy()
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.config.url, extra_headers=headers, timeout=self.config.timeout
            )
            self.connected = True
            logger.info(f"Connected to MCP server: {self.config.name} at {self.config.url}")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request via WebSocket.

        Args:
            request: The request to send.

        Returns:
            The response from the server.
        """
        if not self.connected or not self.websocket:
            raise RetrieverError("WebSocket not connected")

        with error_context(
            component="WebSocketTransport",
            operation="send request",
            error_class=RetrieverError,
            message_prefix="Failed to send MCP request",
        ):
            # Generate request ID if not provided
            if not request.id:
                self._request_counter += 1
                request.id = f"req_{self._request_counter}"

            # Send request
            request_data = {
                "jsonrpc": "2.0",
                "method": request.method,
                "params": request.params,
                "id": request.id,
            }

            await self.websocket.send(json.dumps(request_data))
            logger.debug(f"Sent MCP request: {request.method}")

            # Wait for response
            response_data = await self.websocket.recv()
            response_json = json.loads(response_data)

            # Parse response
            response = MCPResponse(
                result=response_json.get("result"),
                error=response_json.get("error"),
                id=response_json.get("id"),
            )

            if response.error:
                raise RetrieverError(f"MCP server error: {response.error}")

            logger.debug(f"Received MCP response for: {request.method}")
            return response

    async def health_check(self) -> bool:
        """Check WebSocket server health.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            if not self.connected:
                return False

            # Send ping request
            request = MCPRequest(method="ping", params={})
            response = await self.send_request(request)
            return response.result is not None
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            return False


def create_transport(config: MCPServerConfig) -> MCPTransport:
    """Factory function to create MCP transport.

    Args:
        config: Server configuration.

    Returns:
        Transport instance for the specified type.

    Raises:
        ValueError: If transport type is not supported.
    """
    if config.transport_type == MCPTransportType.WEBSOCKET:
        return WebSocketTransport(config)
    elif config.transport_type == MCPTransportType.HTTP:
        raise NotImplementedError("HTTP transport not yet implemented")
    elif config.transport_type == MCPTransportType.STDIO:
        raise NotImplementedError("STDIO transport not yet implemented")
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")


class MCPClient:
    """MCP client for communicating with MCP servers.

    This client provides a high-level interface for communicating with MCP servers
    that implement retrieval capabilities. It handles connection management,
    request/response processing, and error handling.

    Attributes:
        config: Server configuration.
        transport: Transport layer for communication.
        max_results: Maximum number of results to return.
    """

    def __init__(self, config: MCPServerConfig, max_results: int = 10):
        """Initialize MCP client.

        Args:
            config: Server configuration.
            max_results: Maximum number of results to return.
        """
        self.config = config
        self.transport = create_transport(config)
        self.max_results = max_results
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server."""
        with error_context(
            component="MCPClient",
            operation="connection",
            error_class=RetrieverError,
            message_prefix="Failed to connect to MCP server",
        ):
            await self.transport.connect()
            self._connected = True
            logger.info(f"MCP client connected to: {self.config.name}")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._connected:
            await self.transport.disconnect()
            self._connected = False
            logger.info(f"MCP client disconnected from: {self.config.name}")

    async def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Query the MCP server for documents.

        Args:
            query: The search query.
            context: Optional context for the query.

        Returns:
            List of retrieved documents.
        """
        if not self._connected:
            await self.connect()

        with error_context(
            component="MCPClient",
            operation="query",
            error_class=RetrieverError,
            message_prefix="Failed to query MCP server",
        ):
            request = MCPRequest(
                method="query",
                params={"query": query, "context": context or {}, "max_results": self.max_results},
            )

            response = await self.transport.send_request(request)

            # Convert response to Document objects
            documents = []
            if response.result and isinstance(response.result, list):
                for doc_data in response.result:
                    if isinstance(doc_data, dict):
                        documents.append(
                            Document(
                                text=doc_data.get("text", ""),
                                metadata=doc_data.get("metadata", {}),
                                score=doc_data.get("score", 1.0),
                            )
                        )
                    elif isinstance(doc_data, str):
                        # Simple text response
                        documents.append(
                            Document(
                                text=doc_data, metadata={"source": self.config.name}, score=1.0
                            )
                        )

            logger.debug(f"Retrieved {len(documents)} documents from MCP server")
            return documents

    async def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a document to the MCP server.

        Args:
            doc_id: Document identifier.
            text: Document text.
            metadata: Optional document metadata.

        Returns:
            True if successful, False otherwise.
        """
        if not self._connected:
            await self.connect()

        with error_context(
            component="MCPClient",
            operation="add document",
            error_class=RetrieverError,
            message_prefix="Failed to add document to MCP server",
        ):
            request = MCPRequest(
                method="add_document",
                params={"doc_id": doc_id, "text": text, "metadata": metadata or {}},
            )

            response = await self.transport.send_request(request)
            return response.result is not None

    async def health_check(self) -> bool:
        """Check if the MCP server is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            return await self.transport.health_check()
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Async context manager exit."""
        await self.disconnect()


class BaseMCPRetriever(ABC):
    """Base class for MCP-based retrievers.

    This class provides common functionality for retrievers that use MCP
    for communication with data sources. It handles connection management,
    error handling, and provides a consistent interface.

    Attributes:
        mcp_client: MCP client for server communication.
        max_results: Maximum number of results to return.
    """

    def __init__(self, config: MCPServerConfig, max_results: int = 10):
        """Initialize MCP retriever.

        Args:
            config: MCP server configuration.
            max_results: Maximum number of results to return.
        """
        self.mcp_client = MCPClient(config, max_results)
        self.max_results = max_results
        self.config = config

    async def retrieve(self, query: str) -> List[str]:
        """Retrieve documents for a query.

        Args:
            query: The search query.

        Returns:
            List of document texts.
        """
        documents = await self.mcp_client.query(query)
        return [doc.text for doc in documents]

    async def retrieve_for_thought(
        self, thought: Thought, is_pre_generation: bool = True
    ) -> Thought:
        """Retrieve documents and add them to a thought.

        Args:
            thought: The thought to add documents to.
            is_pre_generation: Whether this is pre-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        # Determine the query based on whether this is pre or post-generation
        if is_pre_generation:
            query = thought.prompt
        else:
            # For post-generation, use both the prompt and the generated text
            query = f"{thought.prompt}\n\n{thought.text}"

        # Retrieve documents
        documents = await self.mcp_client.query(query)

        # Add documents to the thought
        if is_pre_generation:
            return thought.add_pre_generation_context(documents)
        else:
            return thought.add_post_generation_context(documents)

    async def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the retriever.

        Args:
            doc_id: Document identifier.
            text: Document text.
            metadata: Optional document metadata.
        """
        await self.mcp_client.add_document(doc_id, text, metadata)

    async def health_check(self) -> bool:
        """Check if the retriever is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return await self.mcp_client.health_check()

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        await self.mcp_client.disconnect()


# Basic retrievers that don't use MCP
class MockRetriever:
    """Mock retriever for testing.

    This retriever returns predefined documents for any query, making it useful
    for testing and development.

    Attributes:
        documents: List of documents to return for any query.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        max_results: int = 3,
    ):
        """Initialize the retriever.

        Args:
            documents: List of documents to return for any query.
            max_results: Maximum number of documents to return.
        """
        self.documents = documents or [
            "This is a mock document about artificial intelligence.",
            "This is a mock document about machine learning.",
            "This is a mock document about natural language processing.",
            "This is a mock document about deep learning.",
            "This is a mock document about neural networks.",
        ]
        self.max_results = max_results

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="MockRetriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            # In a real implementation, this would search for relevant documents
            # For the mock, we just return the predefined documents
            results = self.documents[: self.max_results]

            logger.debug(f"Retrieved {len(results)} documents")
            return results

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.

        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="MockRetriever",
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
                    metadata={"source": "mock", "query": query},
                    score=1.0 - (i * 0.1),  # Mock scores
                )
                for i, text in enumerate(document_texts)
            ]

            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)


class InMemoryRetriever:
    """In-memory retriever for simple document collections.

    This retriever stores documents in memory and performs simple keyword matching
    to retrieve relevant documents for a query.

    Attributes:
        documents: Dictionary mapping document IDs to document texts.
        metadata: Dictionary mapping document IDs to metadata.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        documents: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        max_results: int = 3,
    ):
        """Initialize the retriever.

        Args:
            documents: Dictionary mapping document IDs to document texts.
            metadata: Dictionary mapping document IDs to metadata.
            max_results: Maximum number of documents to return.
        """
        self.documents = documents or {}
        self.metadata = metadata or {}
        self.max_results = max_results

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the retriever.

        Args:
            doc_id: The document ID.
            text: The document text.
            metadata: Optional metadata for the document.
        """
        self.documents[doc_id] = text
        if metadata:
            self.metadata[doc_id] = metadata

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="InMemoryRetriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")

            # Simple keyword matching
            query_terms = set(query.lower().split())
            results = []

            for doc_id, text in self.documents.items():
                # Count how many query terms appear in the document
                doc_terms = set(text.lower().split())
                matches = len(query_terms.intersection(doc_terms))

                if matches > 0:
                    results.append((doc_id, text, matches))

            # Sort by number of matches (descending)
            results.sort(key=lambda x: x[2], reverse=True)

            # Return the top results
            top_results = [text for _, text, _ in results[: self.max_results]]

            logger.debug(f"Retrieved {len(top_results)} documents")
            return top_results

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.

        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="InMemoryRetriever",
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
                    metadata={"source": "in-memory", "query": query},
                    score=1.0 - (i * 0.1),  # Simple scoring
                )
                for i, text in enumerate(document_texts)
            ]

            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)
