#!/usr/bin/env python3
"""Example demonstrating the unified 3-tier storage architecture.

This example shows how to use the new unified storage system that replaces
the inconsistent storage patterns with a clean memory → cache → persistence
architecture across all Sifaka components.

Setup Instructions:
1. Start Docker Redis:
   docker run -d -p 6379:6379 redis:latest

2. Install Redis MCP Server:
   npm install -g @modelcontextprotocol/server-redis

3. Install Milvus MCP Server:
   npm install -g @milvus-io/mcp-server-milvus

4. Run this example:
   python examples/storage/unified_storage_example.py

The example demonstrates:
1. Unified storage manager setup
2. Thought storage with vector search
3. Checkpoint storage for recovery
4. Cached retriever wrapper
5. Performance metrics storage
6. Cross-component consistency
"""

import time
from datetime import datetime

from sifaka.core.thought import Thought, Document
from sifaka.retrievers import InMemoryRetriever
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage import SifakaStorage, ChainCheckpoint
from sifaka.utils.logging import configure_logging


def create_storage_configs():
    """Create MCP configurations for Redis and Milvus."""
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    return redis_config, milvus_config


def demonstrate_unified_storage():
    """Demonstrate the unified storage architecture."""
    print("🏗️  Unified Storage Architecture Demo")
    print("=" * 50)

    # Create storage manager
    redis_config, milvus_config = create_storage_configs()
    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        memory_size=100,  # Small for demo
        cache_ttl=300,  # 5 minutes
    )

    print(f"✅ Created unified storage manager")
    print(f"   Memory size: 100 items")
    print(f"   Cache TTL: 300 seconds")
    print()

    return storage


def demonstrate_thought_storage(storage: SifakaStorage):
    """Demonstrate thought storage with vector search."""
    print("💭 Thought Storage Demo")
    print("-" * 30)

    # Get thought storage
    thought_storage = storage.get_thought_storage()

    # Create and save some thoughts
    thoughts = [
        Thought(
            prompt="Write about artificial intelligence",
            text="AI is transforming how we work and live...",
            chain_id="demo_chain_1",
        ),
        Thought(
            prompt="Explain machine learning concepts",
            text="Machine learning enables computers to learn patterns...",
            chain_id="demo_chain_1",
        ),
        Thought(
            prompt="Discuss renewable energy solutions",
            text="Solar and wind power are becoming more efficient...",
            chain_id="demo_chain_2",
        ),
    ]

    print(f"Saving {len(thoughts)} thoughts...")
    for thought in thoughts:
        thought_storage.save_thought(thought)
        print(f"  ✅ Saved thought: {thought.prompt[:40]}...")

    # Test retrieval
    print(f"\nTesting thought retrieval...")
    retrieved = thought_storage.get_thought(thoughts[0].id)
    if retrieved:
        print(f"  ✅ Retrieved: {retrieved.prompt[:40]}...")

    # Test vector similarity search
    print(f"\nTesting vector similarity search...")
    similar = thought_storage.find_similar_thoughts_by_text("artificial intelligence", limit=2)
    print(f"  ✅ Found {len(similar)} similar thoughts")
    for sim_thought in similar:
        print(f"     - {sim_thought.prompt[:40]}...")

    print()
    return thought_storage


def demonstrate_checkpoint_storage(storage: SifakaStorage):
    """Demonstrate checkpoint storage for recovery."""
    print("🔄 Checkpoint Storage Demo")
    print("-" * 30)

    # Get checkpoint storage
    checkpoint_storage = storage.get_checkpoint_storage()

    # Create a sample thought for checkpoints
    thought = Thought(
        prompt="Complex multi-step generation task",
        text="This is a complex task that might need recovery...",
        chain_id="recovery_demo",
    )

    # Create checkpoints at different stages
    checkpoints = [
        ChainCheckpoint(
            chain_id="recovery_demo",
            current_step="pre_retrieval",
            iteration=1,
            thought=thought,
            recovery_point="start_retrieval",
            completed_validators=[],
            completed_critics=[],
        ),
        ChainCheckpoint(
            chain_id="recovery_demo",
            current_step="validation",
            iteration=1,
            thought=thought,
            recovery_point="start_validation",
            completed_validators=["LengthValidator"],
            completed_critics=[],
        ),
        ChainCheckpoint(
            chain_id="recovery_demo",
            current_step="criticism",
            iteration=2,
            thought=thought,
            recovery_point="start_criticism",
            completed_validators=["LengthValidator", "BiasValidator"],
            completed_critics=["ReflexionCritic"],
        ),
    ]

    print(f"Saving {len(checkpoints)} checkpoints...")
    for checkpoint in checkpoints:
        checkpoint_storage.save_checkpoint(checkpoint)
        print(
            f"  ✅ Saved checkpoint: {checkpoint.current_step} (iteration {checkpoint.iteration})"
        )

    # Test recovery scenarios
    print(f"\nTesting checkpoint recovery...")
    latest = checkpoint_storage.get_latest_checkpoint("recovery_demo")
    if latest:
        print(f"  ✅ Latest checkpoint: {latest.current_step} at iteration {latest.iteration}")

    # Test similarity search for recovery patterns
    print(f"\nTesting recovery pattern search...")
    similar_checkpoints = checkpoint_storage.find_similar_checkpoints(checkpoints[1], limit=2)
    print(f"  ✅ Found {len(similar_checkpoints)} similar execution patterns")

    print()
    return checkpoint_storage


def demonstrate_cached_retriever(storage: SifakaStorage):
    """Demonstrate cached retriever wrapper."""
    print("📚 Cached Retriever Demo")
    print("-" * 30)

    # Create a base retriever
    base_retriever = InMemoryRetriever()

    # Add some documents
    docs = {
        "ai_doc1": "Artificial intelligence is revolutionizing technology",
        "ai_doc2": "Machine learning algorithms can identify patterns in data",
        "energy_doc1": "Solar panels convert sunlight into electricity",
        "energy_doc2": "Wind turbines generate clean renewable energy",
    }

    for doc_id, text in docs.items():
        base_retriever.add_document(doc_id, text)

    # Wrap with caching
    cached_retriever = storage.get_retriever_cache(base_retriever)

    print(f"Created cached retriever wrapping InMemoryRetriever")
    print(f"Added {len(docs)} documents to base retriever")

    # Test caching behavior
    query = "artificial intelligence machine learning"

    print(f"\nTesting cache performance...")
    print(f"Query: '{query}'")

    # First query (cache miss)
    start_time = time.time()
    results1 = cached_retriever.retrieve(query)
    time1 = time.time() - start_time
    print(f"  1st query: {len(results1)} results in {time1*1000:.2f}ms (cache miss)")

    # Second query (cache hit)
    start_time = time.time()
    results2 = cached_retriever.retrieve(query)
    time2 = time.time() - start_time
    print(f"  2nd query: {len(results2)} results in {time2*1000:.2f}ms (cache hit)")

    # Show cache stats
    stats = cached_retriever.get_cache_stats()
    print(f"  Cache hit rate: {stats['cache_performance']['hit_rate']:.2f}")

    print()
    return cached_retriever


def demonstrate_metrics_storage(storage: SifakaStorage):
    """Demonstrate performance metrics storage."""
    print("📊 Metrics Storage Demo")
    print("-" * 30)

    # Get metrics storage
    metrics_storage = storage.get_metrics_storage()

    # Record some sample metrics
    operations = [
        ("text_generation", "AnthropicModel", 1250.5),
        ("validation", "LengthValidator", 15.2),
        ("validation", "BiasValidator", 89.7),
        ("criticism", "ReflexionCritic", 2100.8),
        ("retrieval", "MilvusRetriever", 156.3),
        ("text_generation", "AnthropicModel", 1180.2),  # Another generation
    ]

    print(f"Recording {len(operations)} performance metrics...")
    for operation, component, duration in operations:
        metric_id = metrics_storage.record_metric(
            operation=operation,
            component=component,
            duration_ms=duration,
            tags=["demo", "unified_storage"],
        )
        print(f"  ✅ {component}.{operation}: {duration:.1f}ms")

    # Analyze performance
    print(f"\nGenerating performance summary...")
    summary = metrics_storage.get_performance_summary(hours=1)

    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Components analyzed: {len(summary['components'])}")

    if summary["bottlenecks"]:
        print(
            f"  Top bottleneck: {summary['bottlenecks'][0]['operation']} "
            f"({summary['bottlenecks'][0]['avg_duration_ms']:.1f}ms avg)"
        )

    print()
    return metrics_storage


def demonstrate_cross_component_consistency(storage: SifakaStorage):
    """Demonstrate consistency across all storage components."""
    print("🔄 Cross-Component Consistency Demo")
    print("-" * 40)

    # Get comprehensive storage stats
    stats = storage.get_storage_stats()

    print("Storage configuration:")
    print(f"  Memory size: {stats['config']['memory_size']}")
    print(f"  Cache TTL: {stats['config']['cache_ttl']}s")
    print(f"  Redis server: {stats['config']['redis_server']}")
    print(f"  Milvus server: {stats['config']['milvus_server']}")

    print(f"\nComponent statistics:")
    for component, component_stats in stats["components"].items():
        if "memory" in component_stats:
            memory_stats = component_stats["memory"]
            print(f"  {component}:")
            print(f"    Memory: {memory_stats['size']}/{memory_stats['max_size']} items")
            print(f"    Hit rate: {memory_stats['hit_rate']:.2f}")

    # Test health check
    print(f"\nPerforming health check...")
    health = storage.health_check()
    print(f"  Overall status: {health['overall']}")
    for backend, backend_health in health["backends"].items():
        print(f"  {backend}: {backend_health['status']}")

    print()


def main():
    """Run the unified storage architecture demo."""
    print("🚀 Sifaka Unified Storage Architecture Demo")
    print("=" * 60)

    # Configure logging
    configure_logging(level="INFO")

    try:
        # Create unified storage
        storage = demonstrate_unified_storage()

        # Demonstrate each component
        demonstrate_thought_storage(storage)
        demonstrate_checkpoint_storage(storage)
        demonstrate_cached_retriever(storage)
        demonstrate_metrics_storage(storage)
        demonstrate_cross_component_consistency(storage)

        print("🎉 Unified Storage Demo Completed Successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   • Consistent 3-tier architecture across all components")
        print("   • Fast memory access with automatic caching")
        print("   • Vector similarity search for semantic queries")
        print("   • Unified configuration and management")
        print("   • Predictable performance characteristics")
        print("   • Easy testing and debugging")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure Redis and Milvus MCP servers are available.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
