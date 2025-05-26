#!/usr/bin/env python3
"""Simple test script to verify three-tier storage is working."""

import asyncio
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage.redis import RedisStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.cached import CachedStorage
from sifaka.core.thought import Thought


async def test_three_tier_storage():
    """Test three-tier storage functionality."""
    print("Testing three-tier storage...")
    
    # Layer 1: Memory storage (fastest)
    memory_storage = MemoryStorage()
    
    # Layer 2: Redis storage (persistent cache)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-redis src/main.py",
    )
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:test")
    
    # Create two-tier cached storage: Memory → Redis
    cached_storage = CachedStorage(
        cache=memory_storage,  # L1: Memory (fastest)
        persistence=redis_storage,  # L2: Redis (persistent)
    )
    
    # Create a test thought
    test_thought = Thought(
        prompt="Test prompt for three-tier storage",
        text="This is a test thought for three-tier storage verification.",
        id="three_tier_test_001"
    )
    
    try:
        print("Storing thought in three-tier storage...")
        await cached_storage._set_async(test_thought.id, test_thought)
        print("✓ Successfully stored thought")
        
        print("Retrieving thought from three-tier storage...")
        retrieved_thought = await cached_storage._get_async(test_thought.id)
        print("✓ Successfully retrieved thought")
        
        if retrieved_thought and hasattr(retrieved_thought, 'text'):
            print(f"Retrieved text: {retrieved_thought.text}")
            if retrieved_thought.text == test_thought.text:
                print("✅ Three-tier storage working correctly!")
                
                # Test cache layers
                print("\nTesting cache layers...")
                
                # Clear memory cache and test Redis persistence
                memory_storage.data.clear()
                print("Cleared memory cache")
                
                retrieved_from_redis = await cached_storage._get_async(test_thought.id)
                if retrieved_from_redis and retrieved_from_redis.text == test_thought.text:
                    print("✅ Redis persistence layer working!")
                else:
                    print("❌ Redis persistence layer failed")
                    return False
                
                return True
            else:
                print("❌ Text content doesn't match")
                return False
        else:
            print("❌ Retrieved thought is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Three-tier storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_three_tier_storage())
    if result:
        print("\n🎉 Three-tier storage is working correctly!")
    else:
        print("\n💥 Three-tier storage test failed!")
