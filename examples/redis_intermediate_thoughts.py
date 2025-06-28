#!/usr/bin/env python
"""Easiest way to store intermediate thoughts in Redis.

Just pass RedisStorage to the improve() function and all thoughts
are automatically stored.
"""

import asyncio
import os
from sifaka import improve, Config
from sifaka.storage.redis import RedisStorage


async def main():
    """Store Sifaka thoughts in Redis automatically."""

    # 1. Create Redis storage
    storage = RedisStorage(
        host="localhost",
        port=6379,
        namespace="sifaka_thoughts",  # All keys prefixed with this
    )

    # 2. Pass storage to improve() - that's it!
    result = await improve(
        "AI will replace all jobs by 2025",
        critics=["factual", "clarity"],
        max_iterations=2,
        storage=storage,  # ← This stores everything automatically
        config=Config(model="gpt-4o-mini"),
    )

    print(f"✅ Stored thought with ID: {result.id}")

    # 3. Retrieve thoughts later
    loaded = await storage.load(result.id)
    print(
        f"📝 Retrieved: {loaded.iteration} iterations, {len(loaded.generations)} generations"
    )

    # 4. Search thoughts
    matches = await storage.search("AI", limit=5)
    print(f"🔍 Found {len(matches)} thoughts mentioning 'AI'")

    # 5. List recent thoughts
    recent = await storage.list(limit=10)
    print(f"📋 {len(recent)} recent thoughts in Redis")

    await storage.close()


if __name__ == "__main__":
    # Check Redis is running
    try:
        import redis

        r = redis.Redis()
        r.ping()
    except:
        print("❌ Redis not running. Start with: redis-server")
        exit(1)

    # Check API key
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        exit(1)

    asyncio.run(main())
