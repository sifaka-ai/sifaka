#!/usr/bin/env python3
"""Debug the fresh collection to see what's stored."""

import asyncio
from sifaka.mcp import MCPClient, MCPServerConfig, MCPTransportType


async def debug_fresh_collection():
    """Debug what's in the fresh collection."""
    print("Debugging fresh collection...")
    
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
        
        print("\nQuerying all records in fresh collection...")
        result_all = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fresh_start",
                "filter_expr": "",  # Get all records
                "output_fields": ["key", "content", "text"],
                "limit": 10,
            },
        )
        
        print(f"All records result: {result_all}")
        
        if result_all and "content" in result_all:
            content = result_all["content"]
            if content and isinstance(content, list) and len(content) > 0:
                response_text = content[0].get("text", "")
                print(f"\nResponse text:\n{response_text}")
        
        print("\nTesting specific query...")
        result_specific = await mcp_client.call_tool(
            "milvus_query",
            {
                "collection_name": "test_fresh_start",
                "filter_expr": "key == 'milvus_test_001'",
                "output_fields": ["key", "content"],
                "limit": 1,
            },
        )
        
        print(f"Specific query result: {result_specific}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(debug_fresh_collection())
    if result:
        print("\n🎉 Debug completed!")
    else:
        print("\n💥 Debug failed!")
