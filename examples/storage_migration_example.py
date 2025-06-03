"""Storage Migration Example for Sifaka.

This example demonstrates the new PydanticAI-native storage system with:
- Multiple storage backends (Memory, File, Redis, Hybrid)
- Thought persistence and retrieval
- Conversation management
- MCP tool integration for search

Prerequisites:
1. For Redis: Start Redis server and MCP Redis server
2. Set environment variables: ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka import SifakaEngine
from sifaka.storage import (
    MemoryPersistence,
    SifakaFilePersistence,
    RedisPersistence,
    HybridPersistence,
    BackendConfig,
    BackendRole,
)
from sifaka.storage.tools import create_retrieval_tools
from sifaka.utils.logging import get_logger

# Import PydanticAI MCP for Redis (optional)
try:
    from pydantic_ai.mcp import MCPServerStdio

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


async def demonstrate_memory_persistence():
    """Demonstrate in-memory persistence."""
    print("\n" + "=" * 60)
    print("1. MEMORY PERSISTENCE DEMO")
    print("=" * 60)

    # Create engine with memory persistence
    memory_persistence = MemoryPersistence(key_prefix="demo")
    engine = SifakaEngine(persistence=memory_persistence)

    print("✅ Created SifakaEngine with MemoryPersistence")

    # Process a thought
    thought = await engine.think("Explain the benefits of renewable energy", max_iterations=2)
    print(f"✅ Processed thought: {thought.id}")
    print(f"   Final text length: {len(thought.final_text) if thought.final_text else 0}")

    # Retrieve the thought
    retrieved = await engine.get_thought(thought.id)
    print(f"✅ Retrieved thought: {retrieved.id if retrieved else 'Not found'}")

    # Get storage stats
    if hasattr(memory_persistence, "get_stats"):
        stats = await memory_persistence.get_stats()
        print(f"✅ Storage stats: {stats}")

    return thought


async def demonstrate_file_persistence():
    """Demonstrate file-based persistence."""
    print("\n" + "=" * 60)
    print("2. FILE PERSISTENCE DEMO")
    print("=" * 60)

    # Create storage directory
    storage_dir = Path("storage_demo")
    storage_dir.mkdir(exist_ok=True)

    # Create engine with file persistence
    file_persistence = SifakaFilePersistence(
        storage_dir=str(storage_dir), key_prefix="demo", auto_backup=True
    )
    engine = SifakaEngine(persistence=file_persistence)

    print(f"✅ Created SifakaEngine with SifakaFilePersistence at {storage_dir}")

    # Process a thought
    thought = await engine.think("Describe the process of photosynthesis", max_iterations=2)
    print(f"✅ Processed thought: {thought.id}")

    # Check if files were created
    thoughts_dir = storage_dir / "thoughts"
    thought_files = list(thoughts_dir.glob("*.json"))
    print(f"✅ Created {len(thought_files)} thought files")

    # Create a backup
    backup_path = await file_persistence.create_backup("demo_backup")
    print(f"✅ Created backup at: {backup_path}")

    return thought


async def demonstrate_redis_persistence():
    """Demonstrate Redis-based persistence (if available)."""
    print("\n" + "=" * 60)
    print("3. REDIS PERSISTENCE DEMO")
    print("=" * 60)

    if not REDIS_AVAILABLE:
        print("❌ Redis MCP not available - skipping Redis demo")
        return None

    try:
        # Create Redis MCP server
        redis_mcp_server = MCPServerStdio(
            "uv",
            args=[
                "run",
                "--directory",
                "/Users/evanvolgas/Documents/not_beam/sifaka/mcp/mcp-redis",
                "src/main.py",
            ],
            tool_prefix="redis",
        )

        # Create engine with Redis persistence
        redis_persistence = RedisPersistence(
            mcp_server=redis_mcp_server, key_prefix="demo", ttl_seconds=3600  # 1 hour TTL
        )
        engine = SifakaEngine(persistence=redis_persistence)

        print("✅ Created SifakaEngine with RedisPersistence")

        # Process a thought
        thought = await engine.think("Explain machine learning algorithms", max_iterations=2)
        print(f"✅ Processed thought: {thought.id}")

        # Get Redis info
        redis_info = await redis_persistence.get_redis_info()
        print(f"✅ Redis info: {redis_info.get('version', 'Unknown')}")

        return thought

    except Exception as e:
        print(f"❌ Redis demo failed: {e}")
        return None


async def demonstrate_hybrid_persistence():
    """Demonstrate hybrid multi-backend persistence."""
    print("\n" + "=" * 60)
    print("4. HYBRID PERSISTENCE DEMO")
    print("=" * 60)

    # Create storage backends
    storage_dir = Path("hybrid_demo")
    storage_dir.mkdir(exist_ok=True)

    # Create hybrid persistence using new flexible approach
    backends = [
        BackendConfig(
            backend=MemoryPersistence(key_prefix="hybrid"),
            role=BackendRole.CACHE,
            priority=0,
            name="MemoryCache",
        ),
        BackendConfig(
            backend=SifakaFilePersistence(storage_dir=str(storage_dir), key_prefix="hybrid"),
            role=BackendRole.PRIMARY,
            priority=1,
            read_repair_target=True,
            name="FilePrimary",
        ),
    ]

    hybrid_persistence = HybridPersistence(
        backends=backends, key_prefix="hybrid", write_through=True, read_repair=True
    )

    engine = SifakaEngine(persistence=hybrid_persistence)

    print("✅ Created SifakaEngine with HybridPersistence (Memory + File)")

    # Process multiple thoughts
    thoughts = []
    prompts = [
        "Explain quantum computing",
        "Describe climate change effects",
        "What is artificial intelligence?",
    ]

    for i, prompt in enumerate(prompts):
        thought = await engine.think(prompt, max_iterations=1)
        thoughts.append(thought)
        print(f"✅ Processed thought {i+1}: {thought.id}")

    # Get hybrid storage stats
    stats = await hybrid_persistence.get_backend_stats()
    print(f"✅ Hybrid storage stats:")
    print(f"   Global stats:")
    for key, value in stats["global_stats"].items():
        print(f"      {key}: {value}")
    print(f"   Backend stats:")
    for backend_name, backend_stats in stats["backend_stats"].items():
        print(
            f"      {backend_name}: {backend_stats['read_count']} reads, {backend_stats['write_count']} writes"
        )

    return thoughts


async def demonstrate_retrieval_tools():
    """Demonstrate MCP retrieval tools."""
    print("\n" + "=" * 60)
    print("5. RETRIEVAL TOOLS DEMO")
    print("=" * 60)

    # Create persistence with some test data
    memory_persistence = MemoryPersistence(key_prefix="tools_demo")
    engine = SifakaEngine(persistence=memory_persistence)

    # Create some test thoughts
    test_prompts = [
        "Explain renewable energy sources",
        "Describe solar panel technology",
        "What are wind turbines?",
    ]

    thoughts = []
    for prompt in test_prompts:
        thought = await engine.think(prompt, max_iterations=1)
        thoughts.append(thought)

    print(f"✅ Created {len(thoughts)} test thoughts")

    # Create retrieval tools
    tools = create_retrieval_tools(memory_persistence)
    print(f"✅ Created {len(tools)} retrieval tools")

    # Demonstrate text search
    search_tool = tools[0]  # search_thoughts_by_text
    results = await search_tool("renewable", limit=5)
    print(f"✅ Text search for 'renewable' found {len(results)} results")

    # Demonstrate statistics
    stats_tool = tools[-1]  # get_thought_statistics
    stats = await stats_tool()
    print(f"✅ Storage statistics: {stats.get('total_thoughts', 0)} total thoughts")

    return thoughts


async def main():
    """Run all storage migration demonstrations."""
    print("🚀 SIFAKA STORAGE MIGRATION DEMO")
    print("This demo showcases the new PydanticAI-native storage system")

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("❌ Missing API keys. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY")
        return

    try:
        # Run all demonstrations
        await demonstrate_memory_persistence()
        await demonstrate_file_persistence()
        await demonstrate_redis_persistence()
        await demonstrate_hybrid_persistence()
        await demonstrate_retrieval_tools()

        print("\n" + "=" * 60)
        print("✅ ALL STORAGE DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("• Memory persistence for development")
        print("• File persistence with backups")
        print("• Redis persistence via MCP (if available)")
        print("• Hybrid multi-backend storage")
        print("• MCP retrieval tools for search")
        print("• Complete thought audit trails")
        print("• Conversation management")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Storage demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
