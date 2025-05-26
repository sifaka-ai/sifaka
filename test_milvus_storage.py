#!/usr/bin/env python3
"""Simple test script to verify Milvus storage is working."""

import asyncio
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage.milvus import MilvusStorage
from sifaka.core.thought import Thought


async def test_milvus_storage():
    """Test Milvus storage functionality."""
    print("Testing Milvus storage...")

    # Create Milvus MCP configuration
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )

    # Create Milvus storage with completely new collection name
    milvus_storage = MilvusStorage(mcp_config=milvus_config, collection_name="test_fresh_start")

    # Create a test thought
    test_thought = Thought(
        prompt="Test prompt for Milvus",
        text="This is a test thought for Milvus storage verification with vector search capabilities.",
        id="milvus_test_001",
    )

    try:
        print("Storing thought in Milvus...")
        await milvus_storage._set_async(test_thought.id, test_thought)
        print("✓ Successfully stored thought")

        print("Retrieving thought from Milvus...")
        retrieved_thought = await milvus_storage._get_async(test_thought.id)
        print("✓ Successfully retrieved thought")

        if retrieved_thought and hasattr(retrieved_thought, "text"):
            print(f"Retrieved text: {retrieved_thought.text}")
            if retrieved_thought.text == test_thought.text:
                print("✅ Milvus storage working correctly!")

                # Test search functionality
                print("\nTesting search functionality...")
                search_results = await milvus_storage._search_async("vector search", limit=5)
                print(f"Search results: {len(search_results)} items found")
                if search_results:
                    print("✅ Milvus search working!")
                else:
                    print("⚠️ Search returned no results (might be expected)")

                return True
            else:
                print("❌ Text content doesn't match")
                print(f"Expected: {test_thought.text}")
                print(f"Got: {retrieved_thought.text}")
                return False
        else:
            print("❌ Retrieved thought is invalid")
            print(f"Retrieved: {retrieved_thought}")
            return False

    except Exception as e:
        print(f"❌ Milvus storage test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_milvus_storage())
    if result:
        print("\n🎉 Milvus storage is working correctly!")
    else:
        print("\n💥 Milvus storage test failed!")
