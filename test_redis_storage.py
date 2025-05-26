#!/usr/bin/env python3
"""Simple test script to verify Redis storage is working."""

import asyncio
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage.redis import RedisStorage
from sifaka.core.thought import Thought


async def test_redis_storage():
    """Test Redis storage functionality."""
    print("Testing Redis storage...")
    
    # Create Redis MCP configuration
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-redis src/main.py",
    )
    
    # Create Redis storage
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:test")
    
    # Create a test thought
    test_thought = Thought(
        prompt="Test prompt",
        text="This is a test thought for Redis storage verification.",
        id="test_001"
    )
    
    try:
        print("Storing thought in Redis...")
        await redis_storage._set_async(test_thought.id, test_thought)
        print("✓ Successfully stored thought")
        
        print("Retrieving thought from Redis...")
        retrieved_thought = await redis_storage._get_async(test_thought.id)
        print("✓ Successfully retrieved thought")
        
        if retrieved_thought and hasattr(retrieved_thought, 'text'):
            print(f"Retrieved text: {retrieved_thought.text}")
            if retrieved_thought.text == test_thought.text:
                print("✅ Redis storage working correctly!")
                return True
            else:
                print("❌ Text content doesn't match")
                return False
        else:
            print("❌ Retrieved thought is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Redis storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_redis_storage())
    if result:
        print("\n🎉 Redis storage is working correctly!")
    else:
        print("\n💥 Redis storage test failed!")
