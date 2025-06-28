"""Example of using Redis storage for real-time thought monitoring.

This shows the easiest way to store and monitor intermediate thoughts in Redis.
"""

import asyncio
import os
from sifaka import improve, Config
from sifaka.storage.redis import RedisStorage


async def main() -> None:
    """Demonstrate Redis storage for thoughts."""

    # Text to improve
    text = """
    Machine learning is when computers learn from data. They use algorithms
    to find patterns. Deep learning uses neural networks. It's very powerful.
    """

    print("🔴 Redis Thoughts Storage Example")
    print("=" * 50)

    # Check if Redis is available
    try:
        # Initialize Redis storage
        storage = RedisStorage(
            host="localhost",
            port=6379,
            prefix="sifaka:demo:",  # Use a prefix to organize keys
            ttl=3600,  # Keep data for 1 hour
        )

        print("✅ Connected to Redis")
        print(f"📍 Keys will be prefixed with: {storage.prefix}")
        print()

        # Run improvement with Redis storage
        print("🚀 Running improvement with Redis storage...")
        result = await improve(
            text,
            critics=["reflexion", "self_refine"],
            max_iterations=3,
            config=Config(
                temperature=0.7,
                critic_model="gpt-3.5-turbo",  # Use faster model for demo
            ),
            storage=storage,  # This stores everything in Redis!
        )

        print("\n✅ Improvement complete!")
        print(f"📝 Result ID: {result.result_id}")
        print(f"🔄 Iterations: {result.iteration}")
        print()

        # Show how to access thoughts from Redis
        print("📊 Retrieving thoughts from Redis...")
        thoughts = await storage.get_thoughts_stream(result.result_id)

        for thought in thoughts:
            print(f"\n[Iteration {thought['iteration']}] {thought['critic']}")
            print(f"Confidence: {thought['confidence']:.2f}")
            print(f"Feedback preview: {thought['feedback'][:150]}...")

        # Show real-time monitoring command
        print("\n💡 To monitor thoughts in real-time from another terminal:")
        print("   redis-cli")
        print(
            f"   > XREAD BLOCK 0 STREAMS {storage.prefix}thoughts:{result.result_id} $"
        )
        print()
        print("Or use the Python monitor:")
        print(
            f"   python -c \"import asyncio; from sifaka.storage.redis import monitor_thoughts; asyncio.run(monitor_thoughts('{result.result_id}'))\""
        )

        # List recent results
        print("\n📋 Recent results in Redis:")
        recent = await storage.list(limit=5)
        for r in recent:
            print(f"   - {r.result_id[:8]}... ({len(r.final_text)} chars)")

        await storage.cleanup()

    except ImportError:
        print("❌ Redis package not installed")
        print("   Install with: pip install redis")
    except ConnectionError:
        print("❌ Could not connect to Redis")
        print("   Make sure Redis is running: redis-server")
    except Exception as e:
        print(f"❌ Error: {e}")


async def demonstrate_live_monitoring():
    """Show how to monitor a running improvement in real-time."""

    print("\n\n🔍 Live Monitoring Demo")
    print("=" * 50)

    storage = RedisStorage(prefix="sifaka:live:")

    # Start improvement in background
    async def background_improvement():
        await improve(
            "Write a compelling introduction to Redis for developers.",
            critics=["self_refine", "constitutional", "meta_rewarding"],
            max_iterations=5,
            storage=storage,
        )

    # Start monitoring (in practice, this would be in another process)
    print("Starting improvement in background...")
    print("In production, run the monitor in a separate terminal")
    print()

    # Just show the concept
    task = asyncio.create_task(background_improvement())
    await asyncio.sleep(1)  # Let it start

    print("Monitor would show thoughts as they're generated...")
    await task


if __name__ == "__main__":
    print("Prerequisites:")
    print("1. Install Redis: brew install redis")
    print("2. Start Redis: redis-server")
    print("3. Install Python client: pip install redis")
    print()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        asyncio.run(main())
        # Uncomment to see live monitoring demo:
        # asyncio.run(demonstrate_live_monitoring())
