"""Core MCP (Model Context Protocol) infrastructure for Sifaka.

This module provides the foundational classes and abstractions for implementing
MCP-based communication in Sifaka. MCP provides a standardized protocol for
communication between AI applications and external services.

The MCP implementation provides:
- Standardized communication protocol
- Multiple transport types (WebSocket, STDIO, HTTP)
- Error handling and retry mechanisms
- Health monitoring and connection management
"""

import asyncio
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore


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
        url: Server URL or connection string (optional for STDIO).
        command: Command to run for STDIO transport (optional).
        args: Command arguments for STDIO transport (optional).
        timeout: Connection timeout in seconds.
        retry_attempts: Number of retry attempts for failed requests.
        retry_delay: Delay between retry attempts in seconds.
        auth_token: Optional authentication token.
        headers: Additional headers for HTTP transport.
        capabilities: Server capabilities to request.
    """

    name: str
    transport_type: MCPTransportType
    url: str = ""
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
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
        self.websocket: Optional[Any] = None
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


class STDIOTransport(MCPTransport):
    """STDIO transport implementation for MCP.

    This transport communicates with MCP servers via stdin/stdout,
    which is the most common way to run community MCP servers.
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize STDIO transport.

        Args:
            config: Server configuration. The URL should be the command to run.
        """
        super().__init__(config)
        self.process: Optional[Any] = None
        self._request_counter = 0

    async def connect(self) -> None:
        """Start the MCP server process and perform initialization."""
        with error_context(
            component="STDIOTransport",
            operation="connection",
            error_class=RetrieverError,
            message_prefix="Failed to start MCP server process",
        ):
            # Determine command parts
            if self.config.command:
                # Use command and args from config
                command_parts = [self.config.command] + self.config.args
            elif self.config.url:
                # Parse the command from the URL (backward compatibility)
                # URL format: "npx @modelcontextprotocol/server-memory" or "python server.py"
                command_parts = self.config.url.split()
            else:
                raise RetrieverError("Either command or url must be specified for STDIO transport")

            # Start the process
            self.process = await asyncio.create_subprocess_exec(
                *command_parts,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Perform MCP initialization sequence
            await self._initialize_mcp()

            self.connected = True
            logger.info(
                f"Started MCP server process: {self.config.name} with command: {self.config.url}"
            )

    async def _initialize_mcp(self) -> None:
        """Perform MCP initialization handshake."""
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": "init_1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "sifaka-mcp-client", "version": "1.0.0"},
            },
        }

        init_json = json.dumps(init_request) + "\n"
        if self.process and self.process.stdin:
            self.process.stdin.write(init_json.encode())
            await self.process.stdin.drain()

        # Read initialization response
        if self.process and self.process.stdout:
            response_line = await self.process.stdout.readline()
        else:
            raise RetrieverError("MCP process not properly initialized")
        if not response_line:
            raise RetrieverError("No initialization response from MCP server")

        response_data = response_line.decode().strip()
        response_json = json.loads(response_data)

        if response_json.get("error"):
            raise RetrieverError(f"MCP initialization failed: {response_json['error']}")

        # Send initialized notification
        initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}

        notif_json = json.dumps(initialized_notification) + "\n"
        if self.process and self.process.stdin:
            self.process.stdin.write(notif_json.encode())
            await self.process.stdin.drain()

        logger.debug("MCP initialization completed successfully")

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self.process and self.connected:
            try:
                # Send termination signal
                self.process.terminate()

                # Wait for process to exit (with timeout)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if it doesn't exit gracefully
                    self.process.kill()
                    await self.process.wait()

            except Exception as e:
                logger.warning(f"Error stopping MCP server process: {e}")
            finally:
                self.connected = False
                self.process = None
                logger.info(f"Stopped MCP server process: {self.config.name}")

    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request via STDIO.

        Args:
            request: The request to send.

        Returns:
            The response from the server.
        """
        if not self.connected or not self.process:
            raise RetrieverError("STDIO process not connected")

        with error_context(
            component="STDIOTransport",
            operation="send request",
            error_class=RetrieverError,
            message_prefix="Failed to send MCP request via STDIO",
        ):
            # Generate request ID if not provided
            if not request.id:
                self._request_counter += 1
                request.id = f"req_{self._request_counter}"

            # Prepare request data
            request_data = {
                "jsonrpc": "2.0",
                "method": request.method,
                "params": request.params,
                "id": request.id,
            }

            # Send request to stdin
            request_json = json.dumps(request_data) + "\n"
            if self.process and self.process.stdin:
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
            else:
                raise RetrieverError("MCP process not available for request")

            logger.debug(f"Sent MCP request via STDIO: {request.method}")

            # Read response from stdout
            response_line = await self.process.stdout.readline()
            if not response_line:
                raise RetrieverError("No response received from MCP server")

            response_data = response_line.decode().strip()
            response_json = json.loads(response_data)

            # Parse response
            response = MCPResponse(
                result=response_json.get("result"),
                error=response_json.get("error"),
                id=response_json.get("id"),
            )

            if response.error:
                raise RetrieverError(f"MCP server error: {response.error}")

            logger.debug(f"Received MCP response via STDIO for: {request.method}")
            return response

    async def health_check(self) -> bool:
        """Check STDIO server health.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            if not self.connected or not self.process:
                return False

            # Check if process is still running
            if self.process.returncode is not None:
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
        return STDIOTransport(config)
    else:
        raise ValueError(f"Unsupported transport type: {config.transport_type}")


class MCPClient:
    """MCP client for communicating with MCP servers.

    This client provides a high-level interface for communicating with MCP servers
    that implement various capabilities. It handles connection management,
    request/response processing, and error handling.

    Attributes:
        config: Server configuration.
        transport: Transport layer for communication.
        max_results: Maximum number of results to return.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        max_results: int = 10,
    ):
        """Initialize the MCP client.

        Args:
            config: Server configuration.
            max_results: Maximum number of results to return.
        """
        self.config = config
        self.max_results = max_results
        self.transport = create_transport(config)
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if not self._connected:
            await self.transport.connect()
            self._connected = True
            logger.info(f"Connected to MCP server: {self.config.name}")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._connected:
            await self.transport.disconnect()
            self._connected = False
            logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The result from the tool call.
        """
        if not self._connected:
            await self.connect()

        request = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

        response = await self.transport.send_request(request)
        return response.result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the MCP server.

        Returns:
            List of available tools with their descriptions.
        """
        if not self._connected:
            await self.connect()

        request = MCPRequest(method="tools/list", params={})
        response = await self.transport.send_request(request)
        tools = response.result.get("tools", []) if response.result else []
        return tools

    async def health_check(self) -> bool:
        """Check if the MCP server is healthy.

        Returns:
            True if the server is healthy, False otherwise.
        """
        if not self._connected:
            try:
                await self.connect()
            except Exception:
                return False

        return await self.transport.health_check()

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to use a different approach
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(coro)
