"""MCP (Model Context Protocol) infrastructure for Sifaka.

This module provides the core MCP infrastructure used across all Sifaka components
for communicating with external services like Redis and Milvus via standardized protocols.

The MCP implementation provides:
- Standardized communication protocol
- Multiple transport types (WebSocket, STDIO, HTTP)
- Error handling and retry mechanisms
- Health monitoring and connection management

Example:
    ```python
    from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType
    
    # Create MCP client for Redis
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379"
    )
    
    client = MCPClient(redis_config)
    await client.connect()
    
    # Use client for operations
    response = await client.call_tool("get", {"key": "my-key"})
    ```
"""

from sifaka.mcp.base import (
    MCPClient,
    MCPRequest,
    MCPResponse,
    MCPServerConfig,
    MCPTransport,
    MCPTransportType,
    STDIOTransport,
    WebSocketTransport,
    create_transport,
)

__all__ = [
    "MCPClient",
    "MCPRequest", 
    "MCPResponse",
    "MCPServerConfig",
    "MCPTransport",
    "MCPTransportType",
    "STDIOTransport",
    "WebSocketTransport",
    "create_transport",
]
