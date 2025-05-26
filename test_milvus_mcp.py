#!/usr/bin/env python3
"""Test Milvus MCP server directly."""

import asyncio
from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType


async def test_milvus_mcp():
    """Test Milvus MCP server directly."""
    print("Testing Milvus MCP server...")

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

        print("\nListing available tools...")
        tools = await mcp_client.list_tools()
        print(f"Available tools: {[tool.get('name', 'unknown') for tool in tools]}")

        print("\nListing collections...")
        result = await mcp_client.call_tool("milvus_list_collections", {})
        print(f"List collections result: {result}")

        print("\nTesting collection creation...")
        schema = {
            "dimension": 384,
            "primary_field": "id",
            "id_type": "int",
            "vector_field": "vector",
            "metric_type": "COSINE",
            "auto_id": True,
            "enable_dynamic_field": True,
        }

        create_result = await mcp_client.call_tool(
            "milvus_create_collection",
            {"collection_name": "test_debug_collection", "collection_schema": schema},
        )
        print(f"Create collection result: {create_result}")

        print("\nListing collections again...")
        result = await mcp_client.call_tool("milvus_list_collections", {})
        print(f"List collections result: {result}")

        print("\nLoading collection...")
        load_result = await mcp_client.call_tool(
            "milvus_load_collection", {"collection_name": "test_debug_collection"}
        )
        print(f"Load collection result: {load_result}")

        print("\nGetting collection info...")
        info_result = await mcp_client.call_tool(
            "milvus_get_collection_info", {"collection_name": "test_debug_collection"}
        )
        print(f"Collection info: {info_result}")

        print("\nTesting data insertion with dynamic fields...")

        # Try format with dynamic fields
        dummy_vector = [float(i % 10) for i in range(384)]
        print(f"Vector type: {type(dummy_vector)}, first few elements: {dummy_vector[:5]}")

        # Test with dynamic fields - format 1
        data3 = {
            "vector": dummy_vector,  # Direct vector
            "key": "test_key_dynamic",
            "content": "test content dynamic",
            "text": "test text dynamic",
        }

        insert_result3 = await mcp_client.call_tool(
            "milvus_insert_data", {"collection_name": "test_debug_collection", "data": data3}
        )
        print(f"Insert result 3 (dynamic fields, direct vector): {insert_result3}")

        # Test with dynamic fields - format 2 (all as lists)
        data4 = {
            "vector": [dummy_vector],  # Vector as list
            "key": ["test_key_dynamic_2"],
            "content": ["test content dynamic 2"],
            "text": ["test text dynamic 2"],
        }

        insert_result4 = await mcp_client.call_tool(
            "milvus_insert_data", {"collection_name": "test_debug_collection", "data": data4}
        )
        print(f"Insert result 4 (dynamic fields, all lists): {insert_result4}")

        return True

    except Exception as e:
        print(f"❌ Milvus MCP test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_milvus_mcp())
    if result:
        print("\n🎉 Milvus MCP server is working!")
    else:
        print("\n💥 Milvus MCP server test failed!")
