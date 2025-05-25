#!/usr/bin/env python3
"""
Unit tests for MCP client functionality.

Tests the core MCP client, transport layers, error handling, and protocol compliance.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from sifaka.retrievers.base import (
    MCPClient,
    MCPServerConfig,
    MCPTransportType,
    MCPRequest,
    MCPResponse,
    WebSocketTransport,
    STDIOTransport,
    create_transport,
)
from sifaka.utils.error_handling import RetrieverError


class TestMCPServerConfig:
    """Test MCP server configuration."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = MCPServerConfig(
            name="test-server",
            transport_type=MCPTransportType.WEBSOCKET,
            url="ws://localhost:8080/mcp",
        )

        assert config.name == "test-server"
        assert config.transport_type == MCPTransportType.WEBSOCKET
        assert config.url == "ws://localhost:8080/mcp"
        assert config.timeout == 30.0
        assert config.retry_attempts == 3

    def test_config_with_auth(self):
        """Test config with authentication."""
        config = MCPServerConfig(
            name="secure-server",
            transport_type=MCPTransportType.WEBSOCKET,
            url="wss://api.example.com/mcp",
            auth_token="secret-token",
            headers={"X-API-Key": "api-key"},
        )

        assert config.auth_token == "secret-token"
        assert config.headers["X-API-Key"] == "api-key"


class TestMCPRequest:
    """Test MCP request message structure."""

    def test_request_creation(self):
        """Test basic request creation."""
        request = MCPRequest(method="query", params={"text": "test query"})

        assert request.method == "query"
        assert request.params["text"] == "test query"
        assert request.id is None

    def test_request_with_id(self):
        """Test request with custom ID."""
        request = MCPRequest(
            method="add_document", params={"doc_id": "doc1", "text": "content"}, id="req-123"
        )

        assert request.id == "req-123"


class TestMCPResponse:
    """Test MCP response message structure."""

    def test_response_success(self):
        """Test successful response."""
        response = MCPResponse(result={"documents": ["doc1", "doc2"]}, error=None, id="req-123")

        assert response.result["documents"] == ["doc1", "doc2"]
        assert response.error is None
        assert response.id == "req-123"

    def test_response_error(self):
        """Test error response."""
        response = MCPResponse(
            result=None, error={"code": -1, "message": "Server error"}, id="req-123"
        )

        assert response.result is None
        assert response.error["message"] == "Server error"


class TestTransportFactory:
    """Test transport factory function."""

    def test_websocket_transport_creation(self):
        """Test WebSocket transport creation."""
        config = MCPServerConfig(
            name="ws-server", transport_type=MCPTransportType.WEBSOCKET, url="ws://localhost:8080"
        )

        transport = create_transport(config)
        assert isinstance(transport, WebSocketTransport)

    def test_stdio_transport_creation(self):
        """Test STDIO transport creation."""
        config = MCPServerConfig(
            name="stdio-server", transport_type=MCPTransportType.STDIO, url="python -m mcp_server"
        )

        transport = create_transport(config)
        assert isinstance(transport, STDIOTransport)

    def test_unsupported_transport(self):
        """Test unsupported transport type."""
        config = MCPServerConfig(
            name="http-server", transport_type=MCPTransportType.HTTP, url="http://localhost:8080"
        )

        with pytest.raises(NotImplementedError):
            create_transport(config)


class TestWebSocketTransport:
    """Test WebSocket transport implementation."""

    @pytest.fixture
    def ws_config(self):
        """WebSocket configuration fixture."""
        return MCPServerConfig(
            name="ws-test", transport_type=MCPTransportType.WEBSOCKET, url="ws://localhost:8080/mcp"
        )

    @pytest.fixture
    def ws_transport(self, ws_config):
        """WebSocket transport fixture."""
        return WebSocketTransport(ws_config)

    @pytest.mark.asyncio
    async def test_websocket_connect_success(self, ws_transport):
        """Test successful WebSocket connection."""
        with patch("websockets.connect") as mock_connect:
            mock_ws = AsyncMock()

            # Create a proper async mock that can be awaited
            async def mock_connect_func(*args, **kwargs):
                return mock_ws

            mock_connect.side_effect = mock_connect_func

            await ws_transport.connect()

            assert ws_transport.websocket == mock_ws
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_connect_failure(self, ws_transport):
        """Test WebSocket connection failure."""
        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(RetrieverError, match="Failed to connect to MCP server"):
                await ws_transport.connect()

    @pytest.mark.asyncio
    async def test_websocket_send_request(self, ws_transport):
        """Test sending request via WebSocket."""
        mock_ws = AsyncMock()
        ws_transport.websocket = mock_ws
        ws_transport.connected = True  # Set connected state

        # Mock response
        mock_response = {"result": {"documents": ["doc1"]}, "error": None, "id": "test-id"}
        mock_ws.recv.return_value = json.dumps(mock_response)

        request = MCPRequest(method="query", params={"text": "test"}, id="test-id")
        response = await ws_transport.send_request(request)

        assert response.result["documents"] == ["doc1"]
        assert response.error is None
        mock_ws.send.assert_called_once()


class TestSTDIOTransport:
    """Test STDIO transport implementation."""

    @pytest.fixture
    def stdio_config(self):
        """STDIO configuration fixture."""
        return MCPServerConfig(
            name="stdio-test", transport_type=MCPTransportType.STDIO, url="python -m test_server"
        )

    @pytest.fixture
    def stdio_transport(self, stdio_config):
        """STDIO transport fixture."""
        return STDIOTransport(stdio_config)

    @pytest.mark.asyncio
    async def test_stdio_connect_success(self, stdio_transport):
        """Test successful STDIO connection."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()

        # Mock the initialization method to avoid the complex handshake
        with patch.object(stdio_transport, "_initialize_mcp") as mock_init:
            with patch(
                "asyncio.create_subprocess_exec", return_value=mock_process
            ) as mock_subprocess:
                mock_init.return_value = None  # Successful initialization

                await stdio_transport.connect()

                assert stdio_transport.process == mock_process
                assert stdio_transport.connected is True
                mock_subprocess.assert_called_once()
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_stdio_send_request(self, stdio_transport):
        """Test sending request via STDIO."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()

        # Mock response
        mock_response = {"result": {"documents": ["doc1"]}, "error": None, "id": "test-id"}
        mock_process.stdout.readline.return_value = (json.dumps(mock_response) + "\n").encode()

        stdio_transport.process = mock_process
        stdio_transport.connected = True  # Set connected state

        request = MCPRequest(method="query", params={"text": "test"}, id="test-id")
        response = await stdio_transport.send_request(request)

        assert response.result["documents"] == ["doc1"]
        assert response.error is None


class TestMCPClient:
    """Test MCP client functionality."""

    @pytest.fixture
    def client_config(self):
        """Client configuration fixture."""
        return MCPServerConfig(
            name="test-client",
            transport_type=MCPTransportType.WEBSOCKET,
            url="ws://localhost:8080/mcp",
        )

    @pytest.fixture
    def mcp_client(self, client_config):
        """MCP client fixture."""
        return MCPClient(client_config, max_results=5)

    def test_client_initialization(self, mcp_client, client_config):
        """Test client initialization."""
        assert mcp_client.config == client_config
        assert mcp_client.max_results == 5
        assert not mcp_client._connected

    @pytest.mark.asyncio
    async def test_client_connect(self, mcp_client):
        """Test client connection."""
        with patch.object(mcp_client.transport, "connect") as mock_connect:
            await mcp_client.connect()

            mock_connect.assert_called_once()
            assert mcp_client._connected

    @pytest.mark.asyncio
    async def test_client_query(self, mcp_client):
        """Test client query method."""
        with patch.object(mcp_client.transport, "send_request") as mock_send:
            mock_response = MCPResponse(
                result=["doc1", "doc2"],
                error=None,
                id="query-1",  # MCPClient.query expects a list directly
            )
            mock_send.return_value = mock_response
            mcp_client._connected = True

            results = await mcp_client.query("test query")

            # Results should be Document objects
            assert len(results) == 2
            assert results[0].text == "doc1"
            assert results[1].text == "doc2"
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_add_document(self, mcp_client):
        """Test client add document method."""
        with patch.object(mcp_client.transport, "send_request") as mock_send:
            mock_response = MCPResponse(result={"success": True}, error=None, id="add-1")
            mock_send.return_value = mock_response
            mcp_client._connected = True

            success = await mcp_client.add_document("doc1", "content", {"type": "text"})

            assert success is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_error_handling(self, mcp_client):
        """Test client error handling."""
        with patch.object(mcp_client.transport, "send_request") as mock_send:
            mock_response = MCPResponse(
                result=None, error={"code": -1, "message": "Server error"}, id="error-1"
            )
            mock_send.return_value = mock_response
            mcp_client._connected = True

            # The query method should return empty list on error, not raise
            results = await mcp_client.query("test query")
            assert results == []


if __name__ == "__main__":
    pytest.main([__file__])
