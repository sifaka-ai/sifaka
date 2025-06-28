"""Example of using Redis storage for persisting thoughts.

This shows the easiest way to store and retrieve intermediate thoughts in Redis.
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
        )

        print("\n✅ Improvement complete!")
        # Save to Redis and get the ID
        result_id = await storage.save(result)
        print(f"📝 Result ID: {result_id}")
        print(f"🔄 Iterations: {result.iteration}")
        print()

        # Show how to access thoughts from Redis
        print("📊 Retrieving thoughts from Redis...")
        thoughts = await storage.get_thoughts(result_id)

        for thought in thoughts:
            print(f"\n[Iteration {thought['iteration']}] {thought['critic']}")
            print(f"Confidence: {thought['confidence']:.2f}")
            print(f"Feedback preview: {thought['feedback'][:150]}...")

        # Show how to access stored data
        print("\n💡 To access thoughts from Redis CLI:")
        print("   redis-cli")
        print(f"   > GET {storage.prefix}thoughts:{result_id}")

        # List recent results
        print("\n📋 Recent results in Redis:")
        recent = await storage.list(limit=5)
        for i, r in enumerate(recent):
            print(f"   - Result {i+1}: {len(r.final_text)} chars")

        await storage.cleanup()

    except ImportError:
        print("❌ Redis package not installed")
        print("   Install with: pip install redis")
    except ConnectionError:
        print("❌ Could not connect to Redis")
        print("   Make sure Redis is running: redis-server")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Prerequisites:")
    print("1. Start Redis with Docker: docker run -d -p 6379:6379 redis")
    print("2. Install Python client: pip install redis")
    print()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        asyncio.run(main())
