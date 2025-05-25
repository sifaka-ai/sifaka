#!/usr/bin/env python3
"""
Flexible Storage Configuration Demo

This example demonstrates the new flexible storage backend configuration
that allows users to choose which storage tiers to enable based on their needs.

PREREQUISITES FOR FULL DEMO:
1. Redis: Run `docker run -d -p 6379:6379 redis:latest`
2. Milvus: Follow https://milvus.io/docs/install_standalone-docker.md and pray it works
3. Milvus MCP Server: Clone https://github.com/zilliztech/mcp-server-milvus and figure it out
   - Clone to: ./mcp-server-milvus (in your project directory)
   - Install with: cd mcp-server-milvus && uv sync
   - Test with: uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530

NOTE: Demo will work with memory-only storage even if Redis/Milvus are unavailable.

Run with: python examples/storage/flexible_storage_demo.py
"""

import time
from sifaka.storage import SifakaStorage
from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType


def demonstrate_storage_configurations():
    """Demonstrate different storage configuration options."""
    print("🏗️  Flexible Storage Configuration Demo")
    print("=" * 50)

    # Configuration 1: Memory-only (simplest, always works)
    print("\n1. Memory-Only Storage (Default)")
    print("-" * 30)
    storage1 = SifakaStorage()
    print(
        f"   Enabled backends: memory={storage1.enable_memory}, redis={storage1.enable_redis}, milvus={storage1.enable_milvus}"
    )
    print("   ✅ Perfect for: Development, testing, simple applications")
    print("   ✅ Benefits: No external dependencies, fast setup")

    # Configuration 2: Memory + Redis (caching)
    print("\n2. Memory + Redis Caching")
    print("-" * 30)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    try:
        storage2 = SifakaStorage(redis_config=redis_config)
        print(
            f"   Enabled backends: memory={storage2.enable_memory}, redis={storage2.enable_redis}, milvus={storage2.enable_milvus}"
        )
        print("   ✅ Perfect for: Production apps, cross-process caching")
        print("   ✅ Benefits: Shared cache, persistence across restarts")
    except Exception as e:
        print(f"   ⚠️  Redis not available: {e}")
        print("   → Falling back to memory-only storage")
        storage2 = SifakaStorage()

    # Configuration 3: Memory + Milvus (vector search)
    print("\n3. Memory + Milvus Vector Storage")
    print("-" * 30)
    # Using local Milvus MCP server
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv --directory ./mcp-server-milvus run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )

    try:
        storage3 = SifakaStorage(milvus_config=milvus_config)
        print(
            f"   Enabled backends: memory={storage3.enable_memory}, redis={storage3.enable_redis}, milvus={storage3.enable_milvus}"
        )
        print("   ✅ Perfect for: AI applications, semantic search")
        print("   ✅ Benefits: Vector similarity search, long-term storage")
    except Exception as e:
        print(f"   ⚠️  Milvus not available: {e}")
        print("   → Falling back to memory-only storage")
        storage3 = SifakaStorage()

    # Configuration 4: Full 3-tier (all backends)
    print("\n4. Full 3-Tier Storage")
    print("-" * 30)
    try:
        storage4 = SifakaStorage(redis_config, milvus_config)
        print(
            f"   Enabled backends: memory={storage4.enable_memory}, redis={storage4.enable_redis}, milvus={storage4.enable_milvus}"
        )
        print("   ✅ Perfect for: Production AI systems, maximum performance")
        print("   ✅ Benefits: All features, optimal performance")
    except Exception as e:
        print(f"   ⚠️  Full storage not available: {e}")
        print("   → Using best available configuration")
        storage4 = storage2 if storage2.enable_redis else storage1

    # Configuration 5: Custom explicit control
    print("\n5. Custom Backend Control")
    print("-" * 30)
    storage5 = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        enable_memory=True,
        enable_redis=False,  # Explicitly disable Redis
        enable_milvus=True,  # But enable Milvus
    )
    print(
        f"   Enabled backends: memory={storage5.enable_memory}, redis={storage5.enable_redis}, milvus={storage5.enable_milvus}"
    )
    print("   ✅ Perfect for: Custom requirements, specific use cases")
    print("   ✅ Benefits: Fine-grained control, skip unwanted backends")

    return storage4  # Return three tier storage for testing


def test_storage_operations(storage: SifakaStorage):
    """Test basic storage operations."""
    print("\n🔧 Testing Storage Operations")
    print("-" * 30)

    # Get thought storage
    thought_storage = storage.get_thought_storage()
    print(f"✅ Created thought storage: {type(thought_storage).__name__}")

    # Create and save a thought
    thought = Thought(
        prompt="Write about the benefits of flexible storage",
        text="Flexible storage allows applications to adapt to different deployment environments...",
    )

    thought_storage.save_thought(thought)
    print(f"✅ Saved thought: {thought.id}")

    # Retrieve the thought
    retrieved = thought_storage.get_thought(thought.id)
    if retrieved:
        print(f"✅ Retrieved thought: {retrieved.id}")
        print(f"   Prompt: {retrieved.prompt[:50]}...")
        print(f"   Text length: {len(retrieved.text or '')} characters")
    else:
        print("❌ Failed to retrieve thought")

    # Get storage statistics
    stats = thought_storage.get_stats()
    print(f"✅ Storage statistics available: {list(stats.keys())}")

    # Test other storage components
    checkpoint_storage = storage.get_checkpoint_storage()
    metrics_storage = storage.get_metrics_storage()
    print(f"✅ Created checkpoint storage: {type(checkpoint_storage).__name__}")
    print(f"✅ Created metrics storage: {type(metrics_storage).__name__}")


def test_storage_performance(storage: SifakaStorage):
    """Test and time storage performance across different tiers."""
    print("\n⚡ Storage Performance Testing")
    print("=" * 50)

    # Get the underlying storage to test individual tiers
    thought_storage = storage.get_thought_storage()
    cached_storage = thought_storage.storage

    # Create test thoughts
    test_thoughts = []
    for i in range(5):
        thought = Thought(
            prompt=f"Performance test prompt {i+1}",
            text=f"This is test thought number {i+1} for performance testing. " * 10,
        )
        test_thoughts.append(thought)

    print(f"📝 Created {len(test_thoughts)} test thoughts")

    # Test 1: Save thoughts and measure tier performance
    print("\n🔄 Testing Save Performance:")
    print("-" * 30)

    for i, thought in enumerate(test_thoughts):
        start_time = time.perf_counter()
        thought_storage.save_thought(thought)
        save_time = (time.perf_counter() - start_time) * 1000
        print(f"   Thought {i+1}: {save_time:.2f}ms")

    # Give async operations time to complete
    time.sleep(0.5)

    # Test 2: Retrieve from different tiers and measure performance
    print("\n📊 Testing Retrieval Performance by Tier:")
    print("-" * 50)

    for i, thought in enumerate(test_thoughts):
        thought_id = thought.id
        key = f"thought:{thought_id}"

        print(f"\n🔍 Thought {i+1} ({thought_id[:8]}...):")

        # Test Memory (L1) retrieval
        if cached_storage.memory:
            start_time = time.perf_counter()
            memory_result = cached_storage.memory.get(key)
            memory_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            status = "✅ HIT" if memory_result else "❌ MISS"
            print(f"   L1 Memory:     {memory_time:.1f}μs  {status}")
        else:
            print(f"   L1 Memory:     DISABLED")

        # Test Redis (L2) retrieval
        if cached_storage.cache:
            try:
                start_time = time.perf_counter()
                redis_result = cached_storage.cache.get(key)
                redis_time = (time.perf_counter() - start_time) * 1000  # milliseconds
                status = "✅ HIT" if redis_result else "❌ MISS"
                print(f"   L2 Redis:      {redis_time:.1f}ms   {status}")
            except Exception as e:
                print(f"   L2 Redis:      ERROR - {str(e)[:30]}...")
        else:
            print(f"   L2 Redis:      DISABLED")

        # Test Milvus (L3) retrieval
        if cached_storage.persistence:
            try:
                start_time = time.perf_counter()
                milvus_result = cached_storage.persistence.get(key)
                milvus_time = (time.perf_counter() - start_time) * 1000  # milliseconds
                status = "✅ HIT" if milvus_result else "❌ MISS"
                print(f"   L3 Milvus:     {milvus_time:.1f}ms   {status}")
            except Exception as e:
                print(f"   L3 Milvus:     ERROR - {str(e)[:30]}...")
        else:
            print(f"   L3 Milvus:     DISABLED")

        # Test unified retrieval (goes through all tiers)
        start_time = time.perf_counter()
        unified_result = thought_storage.get_thought(thought_id)
        unified_time = (time.perf_counter() - start_time) * 1000  # milliseconds
        status = "✅ SUCCESS" if unified_result else "❌ FAILED"
        print(f"   🎯 Unified:     {unified_time:.1f}ms   {status}")

    # Test 3: Cache warming demonstration
    print("\n🔥 Cache Warming Demonstration:")
    print("-" * 40)

    # Clear memory cache to simulate cold start
    if cached_storage.memory:
        cached_storage.memory.clear()
        print("   Cleared memory cache (simulating cold start)")

    # First retrieval (should be slower - cache miss)
    test_thought_id = test_thoughts[0].id
    try:
        start_time = time.perf_counter()
        first_retrieval = thought_storage.get_thought(test_thought_id)
        first_time = (time.perf_counter() - start_time) * 1000
        print(f"   1st retrieval (cold): {first_time:.2f}ms")

        # Second retrieval (should be faster - cache hit)
        start_time = time.perf_counter()
        second_retrieval = thought_storage.get_thought(test_thought_id)
        second_time = (time.perf_counter() - start_time) * 1000
        print(f"   2nd retrieval (warm): {second_time:.2f}ms")

        if first_time > 0 and second_time > 0:
            speedup = first_time / second_time
            print(f"   🚀 Speedup: {speedup:.1f}x faster")
    except Exception as e:
        print(f"   ⚠️ Cache warming test failed: {str(e)[:50]}...")
        print("   → This is expected when Redis/Milvus are unavailable")
        print("   → Memory-only storage doesn't need cache warming")

    # Test 4: Storage statistics
    print("\n📈 Storage Statistics:")
    print("-" * 25)

    stats = cached_storage.get_stats()
    for tier, tier_stats in stats.items():
        print(f"   {tier.capitalize()}:")
        if isinstance(tier_stats, dict):
            for key, value in tier_stats.items():
                if key in ["hits", "misses", "size", "max_size"]:
                    print(f"     {key}: {value}")
                elif key == "hit_rate":
                    print(f"     {key}: {value:.1%}")


def show_configuration_benefits():
    """Show the benefits of flexible configuration."""
    print("\n💡 Configuration Benefits")
    print("=" * 50)

    print("\n🚀 Development & Testing:")
    print("   • Memory-only: Fast setup, no dependencies")
    print("   • Instant feedback, easy debugging")

    print("\n🏭 Production Deployment:")
    print("   • Memory + Redis: Cross-process caching")
    print("   • Memory + Milvus: AI-powered semantic search")
    print("   • Full 3-tier: Maximum performance and features")

    print("\n🎯 Custom Requirements:")
    print("   • Explicit backend control for specific needs")
    print("   • Graceful degradation when backends unavailable")
    print("   • Easy migration between configurations")

    print("\n🔧 Operational Benefits:")
    print("   • No breaking changes to existing code")
    print("   • Automatic fallback to available backends")
    print("   • Clear logging of enabled/disabled backends")


def main():
    """Run the flexible storage demo."""
    try:
        # Demonstrate different configurations
        storage = demonstrate_storage_configurations()

        # Test basic storage operations
        test_storage_operations(storage)

        # Test performance across storage tiers
        test_storage_performance(storage)

        # Show benefits
        show_configuration_benefits()

        print("\n🎉 Flexible Storage Demo Completed!")
        print("\n✨ Key Takeaways:")
        print("   • Choose backends based on your needs")
        print("   • Graceful degradation when backends unavailable")
        print("   • Same API regardless of configuration")
        print("   • Memory tier provides microsecond access times")
        print("   • Redis tier provides millisecond cross-process caching")
        print("   • Milvus tier provides semantic vector search")
        print("   • Cache warming dramatically improves performance")
        print("   • Easy to migrate between configurations")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
