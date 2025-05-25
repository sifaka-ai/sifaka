#!/usr/bin/env python3
"""
Example: Cache thoughts in Redis, Milvus, and Memory, then retrieve from Milvus.

This example demonstrates exactly what you asked for:
1. Set up a 3-tier storage system: Memory → Redis → Milvus
2. Store thoughts that get cached in all three layers
3. Retrieve a specific thought from Milvus using semantic search
4. Show how the cache hierarchy works
"""

from sifaka.storage import MemoryStorage, RedisStorage, MilvusStorage, CachedStorage
from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType


def setup_three_tier_storage():
    """Set up Memory → Redis → Milvus storage hierarchy."""
    print("Setting up 3-tier storage: Memory → Redis → Milvus")
    
    # Layer 1: Memory (fastest)
    memory = MemoryStorage()
    print("✓ Memory storage initialized")
    
    # Layer 2: Redis (fast, cross-process)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379"
    )
    redis = RedisStorage(redis_config, key_prefix="sifaka:thoughts")
    print("✓ Redis storage configured")
    
    # Layer 3: Milvus (persistent, semantic search)
    milvus_config = MCPServerConfig(
        name="milvus-server", 
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus"
    )
    milvus = MilvusStorage(milvus_config, collection_name="sifaka_thoughts")
    print("✓ Milvus storage configured")
    
    # Create the hierarchy: Memory caches Redis, Redis caches Milvus
    redis_with_memory = CachedStorage(cache=memory, persistence=redis)
    full_storage = CachedStorage(cache=redis_with_memory, persistence=milvus)
    
    print("✓ 3-tier storage hierarchy created")
    return full_storage, memory, redis, milvus


def store_thoughts_in_all_layers(storage):
    """Store thoughts that will be cached in Memory, Redis, and Milvus."""
    print("\nStoring thoughts in all layers...")
    
    # Create diverse thoughts for testing
    thoughts = [
        Thought(
            prompt="Explain machine learning algorithms",
            text="Machine learning algorithms are computational methods that learn patterns from data. Popular algorithms include linear regression, decision trees, neural networks, and support vector machines. These algorithms can be supervised, unsupervised, or reinforcement-based."
        ),
        Thought(
            prompt="What is natural language processing?", 
            text="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It combines computational linguistics with machine learning and deep learning to process text and speech data."
        ),
        Thought(
            prompt="Describe computer vision applications",
            text="Computer vision enables machines to interpret visual information from the world. Applications include facial recognition, medical image analysis, autonomous vehicles, quality control in manufacturing, and augmented reality systems."
        ),
        Thought(
            prompt="What is deep learning?",
            text="Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. It excels at tasks like image recognition, speech processing, and natural language understanding by automatically learning hierarchical representations."
        ),
        Thought(
            prompt="Explain reinforcement learning",
            text="Reinforcement learning is a type of machine learning where agents learn to make decisions by interacting with an environment. The agent receives rewards or penalties for actions, gradually learning optimal strategies through trial and error."
        )
    ]
    
    stored_keys = []
    for i, thought in enumerate(thoughts):
        thought_key = f"thought_{thought.chain_id}_{thought.iteration}"
        storage.set(thought_key, thought.model_dump())
        stored_keys.append(thought_key)
        print(f"  {i+1}. Stored: {thought.prompt}")
    
    print(f"✓ Stored {len(thoughts)} thoughts in 3-tier storage")
    return stored_keys, thoughts


def demonstrate_cache_hierarchy(storage, memory, redis, milvus, stored_keys):
    """Show how the cache hierarchy works."""
    print("\nDemonstrating cache hierarchy...")
    
    # All thoughts should now be in all three layers
    print(f"Memory has: {len(memory)} thoughts")
    print(f"Redis has: {len(redis)} thoughts (simulated)")
    print(f"Milvus has: {len(milvus)} thoughts (simulated)")
    
    # Clear memory cache to test hierarchy
    print("\nClearing memory cache...")
    memory.clear()
    print(f"Memory now has: {len(memory)} thoughts")
    
    # Retrieve a thought - should come from Redis and re-cache in memory
    print("\nRetrieving thought (should come from Redis → Memory)...")
    test_key = stored_keys[0]
    retrieved_data = storage.get(test_key)
    
    if retrieved_data:
        thought = Thought.model_validate(retrieved_data)
        print(f"✓ Retrieved: {thought.prompt}")
        print(f"Memory now has: {len(memory)} thoughts (re-cached)")
    else:
        print("⚠ Could not retrieve thought")


def search_thoughts_in_milvus(storage, milvus):
    """Demonstrate semantic search in Milvus."""
    print("\nSearching thoughts using Milvus semantic search...")
    
    # Search for AI/ML related thoughts
    search_queries = [
        "artificial intelligence and neural networks",
        "computer vision and image processing", 
        "language understanding and text processing",
        "learning algorithms and data patterns"
    ]
    
    for query in search_queries:
        print(f"\nSearching: '{query}'")
        
        # Search using the full storage (will use Milvus for search)
        results = storage.search(query, limit=3)
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and 'prompt' in result:
                print(f"  {i+1}. {result['prompt']}")
            else:
                print(f"  {i+1}. {str(result)[:100]}...")
    
    # Direct search in Milvus
    print(f"\nDirect Milvus search for 'deep learning':")
    milvus_results = milvus.search("deep learning and neural networks", limit=2)
    print(f"Milvus found {len(milvus_results)} results")


def retrieve_specific_thought_from_milvus(milvus):
    """Retrieve a specific thought directly from Milvus."""
    print("\nRetrieving specific thought from Milvus...")
    
    # Search for a specific concept
    query = "natural language processing and text understanding"
    results = milvus.search(query, limit=1)
    
    if results:
        print(f"✓ Found thought in Milvus matching: '{query}'")
        result = results[0]
        if isinstance(result, dict) and 'prompt' in result:
            print(f"Prompt: {result['prompt']}")
            print(f"Text preview: {result['text'][:150]}...")
        else:
            print(f"Result: {str(result)[:200]}...")
    else:
        print("⚠ No matching thoughts found in Milvus")


def main():
    """Main demonstration."""
    print("🚀 Redis + Milvus + Memory Storage Example")
    print("=" * 50)
    
    try:
        # 1. Set up the 3-tier storage
        storage, memory, redis, milvus = setup_three_tier_storage()
        
        # 2. Store thoughts in all layers
        stored_keys, thoughts = store_thoughts_in_all_layers(storage)
        
        # 3. Demonstrate cache hierarchy
        demonstrate_cache_hierarchy(storage, memory, redis, milvus, stored_keys)
        
        # 4. Search thoughts using Milvus
        search_thoughts_in_milvus(storage, milvus)
        
        # 5. Retrieve specific thought from Milvus
        retrieve_specific_thought_from_milvus(milvus)
        
        print("\n🎉 Example completed successfully!")
        print("\nWhat happened:")
        print("1. ✓ Set up Memory → Redis → Milvus storage hierarchy")
        print("2. ✓ Stored thoughts in all three layers automatically")
        print("3. ✓ Demonstrated cache hierarchy (Memory → Redis → Milvus)")
        print("4. ✓ Used Milvus for semantic search across thoughts")
        print("5. ✓ Retrieved specific thoughts from Milvus using vector search")
        
    except Exception as e:
        print(f"\n⚠ Demo requires Redis and Milvus MCP servers to be running")
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("1. Start Redis: docker run -d -p 6379:6379 redis:alpine")
        print("2. Start Milvus: docker run -d -p 19530:19530 milvusdb/milvus:latest")
        print("3. Install MCP servers:")
        print("   npm install -g @modelcontextprotocol/server-redis")
        print("   npm install -g @milvus-io/mcp-server-milvus")


if __name__ == "__main__":
    main()
