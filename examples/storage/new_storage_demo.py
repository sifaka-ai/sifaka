#!/usr/bin/env python3
"""
Comprehensive demo of the new simple storage system.

This example shows how to:
1. Use memory-only storage (default)
2. Use file persistence
3. Use Redis caching
4. Use Milvus vector storage
5. Combine multiple storage layers
6. Store and retrieve thoughts
7. Search thoughts semantically
"""

import tempfile
from pathlib import Path

from sifaka.storage import MemoryStorage, FileStorage, RedisStorage, MilvusStorage, CachedStorage
from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType


def demo_memory_storage():
    """Demo 1: Simple memory storage (default)."""
    print("=== Demo 1: Memory Storage (Default) ===")
    
    # Memory storage - no persistence, fastest access
    storage = MemoryStorage()
    
    # Create and store a thought
    thought = Thought(prompt="What is AI?", text="AI is artificial intelligence...")
    thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
    
    storage.set(thought_key, thought.model_dump())
    print(f"Stored thought: {thought.prompt}")
    
    # Retrieve the thought
    stored_data = storage.get(thought_key)
    if stored_data:
        retrieved_thought = Thought.model_validate(stored_data)
        print(f"Retrieved thought: {retrieved_thought.prompt}")
    
    # Search (returns all values for memory storage)
    results = storage.search("AI", limit=5)
    print(f"Search results: {len(results)} thoughts found")
    
    print("✓ Memory storage demo complete\n")


def demo_file_storage():
    """Demo 2: File persistence."""
    print("=== Demo 2: File Storage ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "thoughts.json"
        storage = FileStorage(str(file_path))
        
        # Store multiple thoughts
        thoughts = [
            Thought(prompt="What is machine learning?", text="ML is a subset of AI..."),
            Thought(prompt="What is deep learning?", text="Deep learning uses neural networks..."),
            Thought(prompt="What is NLP?", text="NLP processes human language...")
        ]
        
        for thought in thoughts:
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            storage.set(thought_key, thought.model_dump())
            print(f"Stored: {thought.prompt}")
        
        # File should exist and contain data
        print(f"File exists: {file_path.exists()}")
        print(f"File size: {file_path.stat().st_size} bytes")
        
        # Create new storage instance to test persistence
        storage2 = FileStorage(str(file_path))
        results = storage2.search("learning", limit=10)
        print(f"After restart: {len(results)} thoughts persisted")
    
    print("✓ File storage demo complete\n")


def demo_redis_storage():
    """Demo 3: Redis storage (requires Redis MCP server)."""
    print("=== Demo 3: Redis Storage ===")
    
    # Configure Redis MCP server
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379"
    )
    
    try:
        storage = RedisStorage(redis_config, key_prefix="sifaka_demo")
        
        # Store a thought
        thought = Thought(prompt="What is distributed computing?", text="Distributed computing...")
        thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
        
        storage.set(thought_key, thought.model_dump())
        print(f"Stored in Redis: {thought.prompt}")
        
        # Retrieve from Redis
        stored_data = storage.get(thought_key)
        if stored_data:
            print("✓ Successfully retrieved from Redis")
        else:
            print("⚠ Redis storage not available (MCP server not running)")
            
    except Exception as e:
        print(f"⚠ Redis demo skipped: {e}")
    
    print("✓ Redis storage demo complete\n")


def demo_milvus_storage():
    """Demo 4: Milvus vector storage (requires Milvus MCP server)."""
    print("=== Demo 4: Milvus Vector Storage ===")
    
    # Configure Milvus MCP server
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus"
    )
    
    try:
        storage = MilvusStorage(milvus_config, collection_name="sifaka_demo_thoughts")
        
        # Store thoughts with different topics
        ai_thoughts = [
            Thought(prompt="Explain neural networks", text="Neural networks are inspired by the brain..."),
            Thought(prompt="What is computer vision?", text="Computer vision enables machines to see..."),
            Thought(prompt="Describe natural language processing", text="NLP helps computers understand text...")
        ]
        
        for thought in ai_thoughts:
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            storage.set(thought_key, thought.model_dump())
            print(f"Stored in Milvus: {thought.prompt}")
        
        # Semantic search (this is where Milvus shines!)
        print("\nPerforming semantic search...")
        results = storage.search("artificial intelligence and machine learning", limit=3)
        print(f"Found {len(results)} semantically similar thoughts")
        
        # Search for computer vision
        vision_results = storage.search("image recognition and visual processing", limit=2)
        print(f"Found {len(vision_results)} vision-related thoughts")
        
    except Exception as e:
        print(f"⚠ Milvus demo skipped: {e}")
    
    print("✓ Milvus storage demo complete\n")


def demo_layered_storage():
    """Demo 5: Layered storage - Memory + Redis + Milvus."""
    print("=== Demo 5: Layered Storage (Memory + File + Milvus) ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create storage layers
        memory = MemoryStorage()  # L1: Fastest cache
        file_storage = FileStorage(str(Path(temp_dir) / "cache.json"))  # L2: Persistent cache
        
        # For demo, we'll use file storage instead of Milvus to avoid MCP dependency
        # In production, you'd use MilvusStorage here
        milvus_config = MCPServerConfig(
            name="milvus-server",
            transport_type=MCPTransportType.STDIO,
            url="npx -y @milvus-io/mcp-server-milvus"
        )
        
        try:
            # Try to use Milvus, fall back to file if not available
            milvus = MilvusStorage(milvus_config, collection_name="sifaka_layered_demo")
            print("Using Milvus for L3 storage")
        except:
            # Fallback to file storage for demo
            milvus = FileStorage(str(Path(temp_dir) / "milvus_fallback.json"))
            print("Using file storage as Milvus fallback")
        
        # Create layered storage: Memory -> File -> Milvus
        # First layer: Memory + File caching
        l2_cache = CachedStorage(cache=memory, persistence=file_storage)
        
        # Second layer: L2 cache + Milvus persistence
        full_storage = CachedStorage(cache=l2_cache, persistence=milvus)
        
        print("Created 3-tier storage: Memory → File → Milvus")
        
        # Store some thoughts
        thoughts = [
            Thought(prompt="What is quantum computing?", text="Quantum computing uses quantum mechanics..."),
            Thought(prompt="Explain blockchain technology", text="Blockchain is a distributed ledger..."),
            Thought(prompt="What is edge computing?", text="Edge computing brings computation closer...")
        ]
        
        for thought in thoughts:
            thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
            full_storage.set(thought_key, thought.model_dump())
            print(f"Stored in layered storage: {thought.prompt}")
        
        print(f"\nMemory cache has: {len(memory)} items")
        print(f"File cache has: {len(file_storage)} items")
        print(f"Milvus storage has: {len(milvus)} items")
        
        # Test cache hierarchy
        print("\nTesting cache hierarchy...")
        
        # Clear memory cache
        memory.clear()
        print("Cleared memory cache")
        
        # Retrieve - should come from file cache and re-populate memory
        thought_key = f"thought_{thoughts[0].chain_id}_{thoughts[0].iteration}"
        retrieved_data = full_storage.get(thought_key)
        
        if retrieved_data:
            print("✓ Retrieved from file cache, re-cached in memory")
            print(f"Memory cache now has: {len(memory)} items")
        
        # Search across all layers
        search_results = full_storage.search("computing", limit=5)
        print(f"Search found: {len(search_results)} results across all layers")
    
    print("✓ Layered storage demo complete\n")


def demo_chain_integration():
    """Demo 6: Using storage with Chain."""
    print("=== Demo 6: Chain Integration ===")
    
    from sifaka.core.chain import Chain
    from sifaka.models.mock import MockModel
    
    # Create storage
    storage = MemoryStorage()
    
    # Create chain with custom storage
    model = MockModel()
    chain = Chain(
        model=model,
        prompt="Write about the future of AI",
        storage=storage
    )
    
    print("Created chain with custom storage")
    
    # Run the chain - thoughts will be automatically stored
    try:
        result = chain.run()
        print(f"Chain completed: {result.text[:100]}...")
        
        # Check what was stored
        print(f"Storage now contains: {len(storage)} items")
        
        # Search stored thoughts
        search_results = storage.search("AI", limit=10)
        print(f"Found {len(search_results)} AI-related thoughts")
        
    except Exception as e:
        print(f"Chain demo skipped: {e}")
    
    print("✓ Chain integration demo complete\n")


if __name__ == "__main__":
    print("🚀 Sifaka New Storage System Demo\n")
    print("This demo shows the new simple, flexible storage system.")
    print("By default, everything is in memory. You can optionally add persistence.\n")
    
    # Run all demos
    demo_memory_storage()
    demo_file_storage()
    demo_redis_storage()
    demo_milvus_storage()
    demo_layered_storage()
    demo_chain_integration()
    
    print("🎉 All demos complete!")
    print("\nKey benefits of the new storage system:")
    print("✓ Simple: Just get/set/search/clear methods")
    print("✓ Default: Memory storage with no external dependencies")
    print("✓ Flexible: Choose your persistence layer")
    print("✓ Composable: Layer multiple storage backends")
    print("✓ Semantic: Vector search with Milvus")
    print("✓ Fast: Memory caching for hot data")
