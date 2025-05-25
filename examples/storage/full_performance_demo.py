#!/usr/bin/env python3
"""
Full Storage Performance Demo with Real Milvus

This example demonstrates the complete 3-tier storage system with:
- Memory (microsecond performance)
- Redis (millisecond caching)
- Milvus (vector search and persistence)

Run with: python examples/storage/full_performance_demo.py
"""

import time
from sifaka.storage import SifakaStorage
from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType


def test_all_three_tiers():
    """Test the complete 3-tier storage system."""
    print("🚀 Complete 3-Tier Storage Performance Test")
    print("=" * 55)

    # Configure Redis MCP server
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    # Configure local Milvus MCP server
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv --directory ./mcp-server-milvus run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )

    try:
        # Create full 3-tier storage
        storage = SifakaStorage(redis_config, milvus_config)
        print(f"✅ Storage created with all tiers:")
        print(f"   Memory: {storage.enable_memory}")
        print(f"   Redis:  {storage.enable_redis}")
        print(f"   Milvus: {storage.enable_milvus}")

        # Get the storage components
        thought_storage = storage.get_thought_storage()
        cached_storage = thought_storage.storage

        # Create test thoughts
        thoughts = []
        for i in range(3):
            thought = Thought(
                prompt=f"Full tier test prompt {i+1}",
                text=f"This is a comprehensive test of all storage tiers for thought {i+1}. " * 25,
            )
            thoughts.append(thought)

        print(f"\n📝 Created {len(thoughts)} test thoughts")

        # Test save performance across all tiers
        print("\n💾 Save Performance (All Tiers):")
        for i, thought in enumerate(thoughts):
            start_time = time.perf_counter()
            thought_storage.save_thought(thought)
            save_time = (time.perf_counter() - start_time) * 1000
            print(f"   Thought {i+1}: {save_time:.2f}ms")

        # Give async operations time to complete
        print("   Waiting for async operations to complete...")
        time.sleep(2.0)

        # Test individual tier performance
        print("\n📊 Individual Tier Performance:")
        test_thought = thoughts[0]
        key = f"thought:{test_thought.id}"

        # Test Memory (L1)
        if cached_storage.memory:
            start_time = time.perf_counter()
            memory_result = cached_storage.memory.get(key)
            memory_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            status = "✅ HIT" if memory_result else "❌ MISS"
            print(f"   L1 Memory:  {memory_time:8.1f}μs  {status}")

        # Test Redis (L2)
        if cached_storage.cache:
            try:
                start_time = time.perf_counter()
                redis_result = cached_storage.cache.get(key)
                redis_time = (time.perf_counter() - start_time) * 1000  # milliseconds
                status = "✅ HIT" if redis_result else "❌ MISS"
                print(f"   L2 Redis:   {redis_time:8.1f}ms   {status}")
            except Exception as e:
                print(f"   L2 Redis:   ERROR - {str(e)[:40]}...")

        # Test Milvus (L3)
        if cached_storage.persistence:
            try:
                start_time = time.perf_counter()
                milvus_result = cached_storage.persistence.get(key)
                milvus_time = (time.perf_counter() - start_time) * 1000  # milliseconds
                status = "✅ HIT" if milvus_result else "❌ MISS"
                print(f"   L3 Milvus:  {milvus_time:8.1f}ms   {status}")
            except Exception as e:
                print(f"   L3 Milvus:  ERROR - {str(e)[:40]}...")

        # Test unified API performance
        start_time = time.perf_counter()
        unified_result = thought_storage.get_thought(test_thought.id)
        unified_time = (time.perf_counter() - start_time) * 1000000  # microseconds
        status = "✅ SUCCESS" if unified_result else "❌ FAILED"
        print(f"   🎯 Unified:  {unified_time:8.1f}μs  {status}")

        # Test cache warming effect
        print("\n🔥 Cache Warming Demonstration:")
        test_cache_warming(thought_storage, cached_storage, thoughts[1])

        # Show storage statistics
        print("\n📈 Storage Statistics:")
        stats = cached_storage.get_stats()
        for tier, tier_stats in stats.items():
            print(f"   {tier.capitalize()}:")
            if isinstance(tier_stats, dict):
                for key, value in tier_stats.items():
                    if key in ["hits", "misses", "size", "max_size"]:
                        print(f"     {key}: {value}")
                    elif key == "hit_rate":
                        print(f"     {key}: {value:.1%}")

        return storage

    except Exception as e:
        print(f"❌ Full 3-tier test failed: {e}")
        print("→ This might be due to Milvus or Redis connectivity issues")
        import traceback

        traceback.print_exc()
        return None


def test_cache_warming(thought_storage, cached_storage, test_thought):
    """Test the cache warming effect across tiers."""
    print("   Testing cache warming across all tiers...")

    # Clear memory to simulate cold start
    if cached_storage.memory:
        cached_storage.memory.clear()
        print("   → Cleared memory cache")

    # Cold retrieval (should hit Redis or Milvus)
    start_time = time.perf_counter()
    cold_result = thought_storage.get_thought(test_thought.id)
    cold_time = (time.perf_counter() - start_time) * 1000
    print(f"   → Cold retrieval: {cold_time:.2f}ms")

    # Warm retrieval (should hit memory)
    start_time = time.perf_counter()
    warm_result = thought_storage.get_thought(test_thought.id)
    warm_time = (time.perf_counter() - start_time) * 1000
    print(f"   → Warm retrieval: {warm_time:.2f}ms")

    if cold_time > 0 and warm_time > 0 and cold_result and warm_result:
        speedup = cold_time / warm_time
        print(f"   🚀 Cache warming speedup: {speedup:.1f}x faster")
    else:
        print("   ⚠️ Cache warming test inconclusive")


def test_fallback_scenarios():
    """Test graceful fallback when backends are unavailable."""
    print("\n🛡️ Testing Graceful Fallback Scenarios")
    print("=" * 45)

    # Test 1: Memory only
    print("\n1. Memory-only fallback:")
    memory_storage = SifakaStorage()
    print(
        f"   Backends: memory={memory_storage.enable_memory}, redis={memory_storage.enable_redis}, milvus={memory_storage.enable_milvus}"
    )

    # Test 2: Disable specific backends
    print("\n2. Explicit backend control:")
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    selective_storage = SifakaStorage(
        redis_config=redis_config, enable_redis=False  # Explicitly disable Redis
    )
    print(
        f"   Backends: memory={selective_storage.enable_memory}, redis={selective_storage.enable_redis}, milvus={selective_storage.enable_milvus}"
    )
    print("   → Redis config provided but explicitly disabled")


def show_performance_summary():
    """Show comprehensive performance insights."""
    print("\n💡 Performance Summary & Insights")
    print("=" * 40)

    print("\n🏆 Typical Performance Characteristics:")
    print("   • Memory:     1-50 microseconds (fastest)")
    print("   • Redis:      1-100 milliseconds (fast)")
    print("   • Milvus:     10-500 milliseconds (depends on data size)")
    print("   • Unified:    Automatically uses fastest available tier")

    print("\n🎯 Architecture Benefits:")
    print("   • Automatic tier optimization")
    print("   • Graceful degradation when backends fail")
    print("   • Cache warming for optimal performance")
    print("   • Flexible configuration for different use cases")

    print("\n🚀 Best Practices:")
    print("   • Start with memory-only for development")
    print("   • Add Redis for production caching")
    print("   • Add Milvus for vector search and long-term storage")
    print("   • Monitor cache hit rates for performance tuning")
    print("   • Use explicit backend control for specific requirements")


def main():
    """Run the complete storage performance demonstration."""
    print("⚡ Sifaka Complete Storage Performance Demo")
    print("=" * 60)
    print("Testing the full 3-tier storage architecture with real backends")

    try:
        # Test the complete 3-tier system
        full_storage = test_all_three_tiers()

        # Test fallback scenarios
        test_fallback_scenarios()

        # Show performance insights
        show_performance_summary()

        print("\n🎉 Complete Performance Demo Finished!")

        if full_storage and full_storage.enable_milvus:
            print("\n✨ Amazing! All three tiers are working:")
            print("   🧠 Memory: Ultra-fast microsecond access")
            print("   🔴 Redis: Fast millisecond caching")
            print("   🟢 Milvus: Vector search and persistence")
            print("   🎯 Unified API: Automatic optimization")
        else:
            print("\n✨ Flexible architecture demonstrated:")
            print("   🧠 Memory: Always available and fast")
            print("   🔴 Redis: Available when configured")
            print("   🟢 Milvus: Available when configured")
            print("   🛡️ Graceful fallback: System adapts automatically")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
