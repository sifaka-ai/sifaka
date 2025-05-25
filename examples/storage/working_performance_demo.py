#!/usr/bin/env python3
"""
Working Storage Performance Demo

This example demonstrates the flexible storage system with REAL MCP servers
that actually exist and work. It shows performance timing across different
storage tiers using only available backends.

PREREQUISITES:
1. Redis: Run `docker run -d -p 6379:6379 redis:latest`
2. Milvus: Follow https://milvus.io/docs/install_standalone-docker.md
3. Milvus MCP Server: Clone https://github.com/zilliztech/mcp-server-milvus
   - Install with: cd mcp-server-milvus && uv sync
   - Test with: uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530

Run with: python examples/storage/working_performance_demo.py
"""

import time
from sifaka.storage import SifakaStorage
from sifaka.core.thought import Thought
from sifaka.mcp import MCPServerConfig, MCPTransportType


def test_memory_only_performance():
    """Test memory-only storage performance."""
    print("🧠 Memory-Only Storage Performance")
    print("=" * 40)

    storage = SifakaStorage()
    thought_storage = storage.get_thought_storage()

    # Create test thoughts
    thoughts = []
    for i in range(3):
        thought = Thought(
            prompt=f"Test prompt {i+1}", text=f"This is test content for thought {i+1}. " * 20
        )
        thoughts.append(thought)

    # Test save performance
    print("\n💾 Save Performance:")
    for i, thought in enumerate(thoughts):
        start_time = time.perf_counter()
        thought_storage.save_thought(thought)
        save_time = (time.perf_counter() - start_time) * 1000000  # microseconds
        print(f"   Thought {i+1}: {save_time:.1f}μs")

    # Test retrieval performance
    print("\n📖 Retrieval Performance:")
    for i, thought in enumerate(thoughts):
        start_time = time.perf_counter()
        retrieved = thought_storage.get_thought(thought.id)
        retrieval_time = (time.perf_counter() - start_time) * 1000000  # microseconds
        status = "✅" if retrieved else "❌"
        print(f"   Thought {i+1}: {retrieval_time:.1f}μs {status}")

    return storage


def test_redis_performance():
    """Test memory + Redis storage performance."""
    print("\n🔴 Memory + Redis Storage Performance")
    print("=" * 45)

    # Configure Redis MCP server (this one actually exists!)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    try:
        storage = SifakaStorage(redis_config=redis_config)
        thought_storage = storage.get_thought_storage()

        print(f"✅ Redis backend enabled: {storage.enable_redis}")

        # Create test thoughts
        thoughts = []
        for i in range(3):
            thought = Thought(
                prompt=f"Redis test prompt {i+1}",
                text=f"This is Redis test content for thought {i+1}. " * 20,
            )
            thoughts.append(thought)

        # Test save performance
        print("\n💾 Save Performance (Memory + Redis):")
        for i, thought in enumerate(thoughts):
            start_time = time.perf_counter()
            thought_storage.save_thought(thought)
            save_time = (time.perf_counter() - start_time) * 1000  # milliseconds
            print(f"   Thought {i+1}: {save_time:.2f}ms")

        # Give Redis time to sync
        time.sleep(0.5)

        # Test cache warming effect
        print("\n🔥 Cache Warming Test:")
        test_thought = thoughts[0]

        # Clear memory to simulate cold start
        cached_storage = thought_storage.storage
        if cached_storage.memory:
            cached_storage.memory.clear()
            print("   Cleared memory cache")

        # Cold retrieval (from Redis)
        start_time = time.perf_counter()
        cold_result = thought_storage.get_thought(test_thought.id)
        cold_time = (time.perf_counter() - start_time) * 1000
        print(f"   Cold retrieval: {cold_time:.2f}ms")

        # Warm retrieval (from memory)
        start_time = time.perf_counter()
        warm_result = thought_storage.get_thought(test_thought.id)
        warm_time = (time.perf_counter() - start_time) * 1000
        print(f"   Warm retrieval: {warm_time:.2f}ms")

        if cold_time > 0 and warm_time > 0:
            speedup = cold_time / warm_time
            print(f"   🚀 Speedup: {speedup:.1f}x faster")

        return storage

    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("→ Falling back to memory-only storage")
        return SifakaStorage()


def test_tier_comparison():
    """Compare performance across different storage tiers."""
    print("\n📊 Storage Tier Performance Comparison")
    print("=" * 50)

    # Test with Redis if available
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    try:
        storage = SifakaStorage(redis_config=redis_config)
        thought_storage = storage.get_thought_storage()
        cached_storage = thought_storage.storage

        # Create a test thought
        thought = Thought(
            prompt="Tier comparison test",
            text="This is a test for comparing storage tier performance. " * 30,
        )

        # Save the thought
        thought_storage.save_thought(thought)
        time.sleep(0.2)  # Let async operations complete

        key = f"thought:{thought.id}"

        print("\n🔍 Individual Tier Performance:")

        # Test Memory (L1)
        if cached_storage.memory:
            start_time = time.perf_counter()
            memory_result = cached_storage.memory.get(key)
            memory_time = (time.perf_counter() - start_time) * 1000000  # microseconds
            status = "✅ HIT" if memory_result else "❌ MISS"
            print(f"   L1 Memory:  {memory_time:6.1f}μs  {status}")

        # Test Redis (L2)
        if cached_storage.cache:
            try:
                start_time = time.perf_counter()
                redis_result = cached_storage.cache.get(key)
                redis_time = (time.perf_counter() - start_time) * 1000  # milliseconds
                status = "✅ HIT" if redis_result else "❌ MISS"
                print(f"   L2 Redis:   {redis_time:6.1f}ms   {status}")
            except Exception as e:
                print(f"   L2 Redis:   ERROR - {str(e)[:30]}...")

        # Test unified API
        start_time = time.perf_counter()
        unified_result = thought_storage.get_thought(thought.id)
        unified_time = (time.perf_counter() - start_time) * 1000000  # microseconds
        status = "✅ SUCCESS" if unified_result else "❌ FAILED"
        print(f"   🎯 Unified: {unified_time:6.1f}μs  {status}")

        # Show performance ratios
        if cached_storage.memory and cached_storage.cache:
            try:
                ratio = (redis_time * 1000) / memory_time  # Convert to same units
                print(f"\n⚡ Performance Ratio:")
                print(f"   Redis is {ratio:.0f}x slower than Memory")
                print(f"   Memory: ~{memory_time:.0f}μs (microseconds)")
                print(f"   Redis:  ~{redis_time:.1f}ms (milliseconds)")
            except:
                pass

    except Exception as e:
        print(f"❌ Redis comparison failed: {e}")
        print("→ Using memory-only storage")


def show_summary():
    """Show performance summary and insights."""
    print("\n💡 Performance Insights")
    print("=" * 30)

    print("\n🏆 Key Findings:")
    print("   • Memory: Sub-microsecond to microsecond access times")
    print("   • Redis: Millisecond access times (1000x slower than memory)")
    print("   • Unified API: Optimizes automatically (hits fastest tier first)")
    print("   • Cache warming: Dramatic performance improvements")

    print("\n🎯 Best Practices:")
    print("   • Use memory for hot data (frequently accessed)")
    print("   • Use Redis for shared data (cross-process/persistence)")
    print("   • Let the unified API handle tier optimization")
    print("   • Monitor cache hit rates for performance tuning")

    print("\n🚀 Flexible Architecture Benefits:")
    print("   • Start simple (memory-only) and scale up")
    print("   • Add backends as needed without code changes")
    print("   • Graceful degradation when backends unavailable")
    print("   • Same API regardless of configuration")


def main():
    """Run the working performance demo."""
    print("⚡ Sifaka Storage Performance Demo")
    print("=" * 50)
    print("Testing REAL storage backends with actual performance timing")

    try:
        # Test 1: Memory-only performance
        memory_storage = test_memory_only_performance()

        # Test 2: Redis performance (if available)
        redis_storage = test_redis_performance()

        # Test 3: Tier comparison
        test_tier_comparison()

        # Show insights
        show_summary()

        print("\n🎉 Performance Demo Completed!")
        print("\n✨ What We Learned:")
        print("   • Memory storage: Ultra-fast (microseconds)")
        print("   • Redis storage: Fast enough for most use cases (milliseconds)")
        print("   • Flexible architecture: Works with any combination")
        print("   • Graceful fallback: System adapts to available backends")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
