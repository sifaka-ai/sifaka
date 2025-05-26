#!/usr/bin/env python3
"""Drop all test collections to start fresh."""

import asyncio
from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType


async def drop_test_collections():
    """Drop all test collections."""
    print("Dropping test collections...")

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

        # List all collections first
        print("\nListing all collections...")
        collections_result = await mcp_client.call_tool("milvus_list_collections", {})
        print(f"Collections: {collections_result}")

        # Clear all data from test collections (since drop isn't available)
        test_collections = [
            "test_collection",
            "test_debug_collection",
            "test_explicit_schema",
            "test_dynamic_enabled",
            "test_single_record",
            "test_fixed_mcp",
        ]

        for collection_name in test_collections:
            try:
                print(f"\nClearing collection: {collection_name}")
                result = await mcp_client.call_tool(
                    "milvus_delete_entities",
                    {
                        "collection_name": collection_name,
                        "filter_expr": "id >= 0",  # Delete all entities
                    },
                )
                print(f"✓ Cleared {collection_name}: {result}")
            except Exception as e:
                print(f"⚠️  Could not clear {collection_name}: {e}")

        # List collections again to confirm
        print("\nListing collections after cleanup...")
        collections_result = await mcp_client.call_tool("milvus_list_collections", {})
        print(f"Remaining collections: {collections_result}")

        return True

    except Exception as e:
        print(f"❌ Failed to drop collections: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(drop_test_collections())
    if result:
        print("\n🎉 Collections dropped successfully!")
    else:
        print("\n💥 Failed to drop collections!")
