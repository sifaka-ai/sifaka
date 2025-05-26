#!/usr/bin/env python3
"""Debug MCP Connections for Redis and Milvus.

This script tests the MCP server connections and manually creates thoughts
to verify that Redis and Milvus storage are working correctly.
"""

import asyncio
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv

from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.redis import RedisStorage
from sifaka.storage.milvus import MilvusStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_test_thought(content: str, thought_id: str = None) -> Thought:
    """Create a test thought for debugging."""
    if thought_id is None:
        thought_id = str(uuid.uuid4())

    return Thought(
        id=thought_id,
        prompt="Debug MCP connection test",  # Required field
        text=content,  # Generated text content
        metadata={
            "test": True,
            "created_at": datetime.now().isoformat(),
            "debug_session": "mcp_connection_test",
        },
    )


async def test_redis_connection():
    """Test Redis MCP connection."""
    print("Testing Redis MCP connection...")

    try:
        # Create Redis MCP configuration
        redis_config = MCPServerConfig(
            name="redis-server",
            transport_type=MCPTransportType.STDIO,
            url="uv run --directory mcp/mcp-redis src/main.py",
        )

        # Create Redis storage
        redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:debug")

        # Test MCP connection first
        print("Testing MCP connection...")
        try:
            await redis_storage.mcp_client.connect()
            print("✓ MCP client connected successfully")

            # Test direct MCP tool calls
            print("Testing direct MCP tool calls...")
            test_key = "sifaka:debug:direct_test"
            test_value = "direct_test_value"

            # Try SET
            set_result = await redis_storage.mcp_client.call_tool(
                "set", {"key": test_key, "value": test_value}
            )
            print(f"Direct SET result: {set_result}")

            # Try GET
            get_result = await redis_storage.mcp_client.call_tool("get", {"key": test_key})
            print(f"Direct GET result: {get_result}")

        except Exception as e:
            print(f"✗ MCP client connection failed: {e}")
            return False

        # Test connection by storing and retrieving a thought
        test_thought = create_test_thought(
            "This is a test thought for Redis MCP connection debugging.", "redis_test_001"
        )

        print("Testing Redis storage implementation...")
        print(f"Storing thought with ID: {test_thought.id}")
        print(f"Thought data: {test_thought.model_dump()}")

        # Test the actual Redis storage implementation
        try:
            print("Attempting to store thought using Redis storage...")
            await redis_storage._set_async(test_thought.id, test_thought)
            print("✓ Successfully stored thought in Redis")

            print("Attempting to retrieve thought using Redis storage...")
            retrieved_thought = await redis_storage._get_async(test_thought.id)
            print("✓ Successfully retrieved thought from Redis")

            if retrieved_thought:
                print(f"Retrieved thought type: {type(retrieved_thought)}")
                if hasattr(retrieved_thought, "text"):
                    print(f"Retrieved text: {retrieved_thought.text}")
                    if retrieved_thought.text == test_thought.text:
                        print("✓ Redis storage working correctly!")
                        return True
                    else:
                        print("✗ Text content doesn't match")
                        print(f"Expected: {test_thought.text}")
                        print(f"Got: {retrieved_thought.text}")
                        return False
                else:
                    print(f"✗ Retrieved object has no text attribute: {retrieved_thought}")
                    return False
            else:
                print("✗ No thought retrieved from Redis storage")
                return False

        except Exception as e:
            print(f"✗ Redis storage test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print(f"Retrieved object type: {type(retrieved_thought)}")
        print(f"Retrieved object: {retrieved_thought}")
        if hasattr(retrieved_thought, "__dict__"):
            print(f"Retrieved object dict: {retrieved_thought.__dict__}")

        # Try to manually reconstruct the Thought if it's a dict
        if isinstance(retrieved_thought, dict):
            print("Retrieved object is a dict, trying to reconstruct Thought...")
            try:
                from sifaka.core.thought import Thought

                reconstructed_thought = Thought.from_dict(retrieved_thought)
                print(f"✓ Successfully reconstructed Thought: {reconstructed_thought.text}")
                retrieved_thought = reconstructed_thought
            except Exception as e:
                print(f"✗ Failed to reconstruct Thought: {e}")
                import traceback

                traceback.print_exc()

        if (
            retrieved_thought
            and hasattr(retrieved_thought, "text")
            and retrieved_thought.text == test_thought.text
        ):
            print("✓ Redis MCP connection working correctly!")
            return True
        else:
            print("✗ Retrieved thought doesn't match stored thought")
            print(f"Expected: {test_thought.text}")
            print(
                f"Got: {retrieved_thought.text if hasattr(retrieved_thought, 'text') else 'No text attribute'}"
            )
            return False

    except Exception as e:
        print(f"✗ Redis MCP connection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_milvus_connection():
    """Test Milvus MCP connection."""
    logger.info("Testing Milvus MCP connection...")

    try:
        # Create Milvus MCP configuration
        milvus_config = MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.STDIO,
            url="uv run --directory mcp/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
        )

        # Create Milvus storage
        milvus_storage = MilvusStorage(mcp_config=milvus_config, collection_name="debug_thoughts")

        # Test connection by storing and retrieving a thought
        test_thought = create_test_thought(
            "This is a test thought for Milvus MCP connection debugging with vector search capabilities.",
            "milvus_test_001",
        )

        logger.info("Attempting to store thought in Milvus...")
        await milvus_storage._set_async(test_thought.id, test_thought)
        logger.info("✓ Successfully stored thought in Milvus")

        logger.info("Attempting to retrieve thought from Milvus...")
        retrieved_thought = await milvus_storage._get_async(test_thought.id)
        logger.info("✓ Successfully retrieved thought from Milvus")

        if retrieved_thought and retrieved_thought.text == test_thought.text:
            logger.info("✓ Milvus MCP connection working correctly!")
            return True
        else:
            logger.error("✗ Retrieved thought doesn't match stored thought")
            return False

    except Exception as e:
        logger.error(f"✗ Milvus MCP connection failed: {e}")
        return False


async def test_memory_storage():
    """Test memory storage as a baseline."""
    print("Testing memory storage (baseline)...")

    try:
        memory_storage = MemoryStorage()

        test_thought = create_test_thought(
            "This is a test thought for memory storage baseline.", "memory_test_001"
        )

        # Use sync methods for memory storage
        memory_storage.set(test_thought.id, test_thought)
        retrieved_thought = memory_storage.get(test_thought.id)

        print(f"Stored thought ID: {test_thought.id}")
        print(f"Stored thought text: {test_thought.text}")

        if retrieved_thought:
            print(f"Retrieved thought type: {type(retrieved_thought)}")
            if hasattr(retrieved_thought, "text"):
                print(f"Retrieved text: {retrieved_thought.text}")
                if retrieved_thought.text == test_thought.text:
                    print("✓ Memory storage working correctly!")
                    return True
                else:
                    print("✗ Text content doesn't match")
                    return False
            else:
                print(f"✗ Retrieved object has no text attribute: {retrieved_thought}")
                return False
        else:
            print("✗ No thought retrieved from memory storage")
            return False

    except Exception as e:
        print(f"✗ Memory storage failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_all_connections():
    """Test all storage connections."""
    logger.info("=" * 60)
    logger.info("MCP CONNECTION DEBUG SESSION")
    logger.info("=" * 60)

    results = {}

    # Test memory storage first (baseline)
    results["memory"] = await test_memory_storage()

    # Test Redis MCP connection
    results["redis"] = await test_redis_connection()

    # Test Milvus MCP connection
    results["milvus"] = await test_milvus_connection()

    # Summary
    logger.info("=" * 60)
    logger.info("CONNECTION TEST RESULTS")
    logger.info("=" * 60)

    for storage_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{storage_type.upper():10} : {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if results["redis"] and results["milvus"]:
        logger.info("🎉 All MCP connections working! Three-tier caching should work.")
    elif results["redis"]:
        logger.info("⚠️  Redis working, but Milvus failed. Two-tier caching possible.")
    elif results["milvus"]:
        logger.info("⚠️  Milvus working, but Redis failed. Direct Milvus storage possible.")
    else:
        logger.info("❌ Both MCP connections failed. Check service setup.")

    return results


def main():
    """Run the MCP connection debug session."""
    try:
        results = asyncio.run(test_all_connections())

        # Exit with appropriate code
        if results["redis"] and results["milvus"]:
            exit(0)  # All good
        else:
            exit(1)  # Some failures

    except KeyboardInterrupt:
        logger.info("Debug session interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error during debug session: {e}")
        exit(1)


if __name__ == "__main__":
    main()
