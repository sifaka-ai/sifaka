#!/usr/bin/env python3
"""Simple test to verify the fixed MCP format works."""

import asyncio
from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType


async def test_simple_query():
    """Test simple query with fixed format."""
    print("Testing simple query with fixed MCP format...")

    # Create Milvus MCP configuration
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )

    # Create MCP client
    mcp_client = MCPClient(milvus_config)

    try:
        print("Connecting to Milvus MCP server...")
        await mcp_client.connect()
        print("✓ Connected successfully")

        # Test the new list-of-dicts format
        print("\nTesting new list-of-dicts format...")
        vector = [float(i % 10) for i in range(384)]  # 384-dimensional vector
        data = [{"vector": vector, "key": "test_new_format", "content": "test content"}]

        result = await mcp_client.call_tool(
            "milvus_insert_data", {"collection_name": "test_fixed_mcp", "data": data}
        )
        print(f"Insert result: {result}")

        # Query for the data
        print("\nQuerying for the data...")
        query_result = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fixed_mcp",
                "filter_expr": "key == 'test_new_format'",
                "output_fields": ["key", "content"],
                "limit": 1,
            },
        )
        print(f"Query result: {query_result}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_simple_query())
    if result:
        print("\n🎉 Test completed!")
    else:
        print("\n💥 Test failed!")
