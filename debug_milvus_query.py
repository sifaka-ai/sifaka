#!/usr/bin/env python3
"""Debug Milvus query to see what the response looks like."""

import asyncio
from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType


async def debug_milvus_query():
    """Debug Milvus query response format."""
    print("Testing Milvus query response...")

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

        print("\nQuerying for all records to see what's stored...")
        result_all = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fixed_mcp",
                "filter_expr": "",  # Get all records
                "output_fields": ["key", "content", "text"],
                "limit": 10,
            },
        )

        print(f"All records result: {result_all}")

        print("\nTrying different filter expressions...")

        # Try 1: Original
        result1 = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_single_record",
                "filter_expr": "key == 'milvus_test_001'",
                "output_fields": ["key", "content"],
                "limit": 1,
            },
        )
        newline = "\n"
        print(
            f"Filter 1 (key == 'milvus_test_001'): {len(result1['content'][0]['text'].split(newline)) - 3} results"
        )

        # Try 2: Array syntax
        result2 = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fixed_mcp",
                "filter_expr": "key == 'milvus_test_001'",
                "output_fields": ["key", "content"],
                "limit": 1,
            },
        )
        print(
            f"Filter 2 (key[0] == 'milvus_test_001'): {len(result2['content'][0]['text'].split(newline)) - 3} results"
        )

        # Try 3: Contains
        result3 = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fixed_mcp",
                "filter_expr": "'milvus_test_001' in key",
                "output_fields": ["key", "content"],
                "limit": 1,
            },
        )
        print(
            f"Filter 3 ('milvus_test_001' in key): {len(result3['content'][0]['text'].split(newline)) - 3} results"
        )

        result = result1  # Use first result for rest of processing

        print(f"Specific query result: {result}")

        if result and "content" in result:
            content = result["content"]
            print(f"Content: {content}")
            if content and isinstance(content, list) and len(content) > 0:
                response_text = content[0].get("text", "")
                print(f"Response text: {response_text}")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(debug_milvus_query())
    if result:
        print("\n🎉 Debug completed!")
    else:
        print("\n💥 Debug failed!")
