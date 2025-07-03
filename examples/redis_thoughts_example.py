"""Example of using MultiStorage with Redis and FileStorage.

This demonstrates:
- Using MultiStorage to save to both Redis and filesystem
- Redis for fast search and real-time access
- FileStorage for permanent archival
- RediSearch capabilities when available
- Graceful fallback when Redis is not available
"""

import asyncio
import os
from datetime import datetime, timedelta
from sifaka import improve, Config
from sifaka.storage import MultiStorage, RedisStorage, FileStorage


async def main() -> None:
    """Demonstrate MultiStorage with Redis and FileStorage."""
    print("🏛️ MultiStorage Example (Redis + FileStorage)")
    print("=" * 50)
    
    try:
        # Part 1: Setup MultiStorage
        storage = await setup_multi_storage()
        
        # Part 2: Create multiple improvements
        await demo_multiple_improvements(storage)
        
        # Part 3: Demonstrate search capabilities
        await demo_search_capabilities(storage)
        
        # Part 4: Show storage status and benefits
        await demo_storage_benefits(storage)
        
        # Cleanup
        await storage.cleanup()
        
    except ImportError:
        print("❌ Redis package not installed")
        print("   Install with: pip install redis")
        print("   Note: FileStorage will still work without Redis")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def setup_multi_storage() -> MultiStorage:
    """Setup MultiStorage with Redis and FileStorage."""
    print("\n🏛️ Part 1: Setting up MultiStorage")
    print("=" * 50)
    
    # Create FileStorage (always works)
    file_storage = FileStorage(
        storage_dir="./thoughts"
    )
    print("✅ FileStorage configured: ./thoughts/")
    
    # Try to create Redis storage
    backends = [file_storage]  # FileStorage is always available
    
    try:
        redis_storage = RedisStorage(
            host="localhost",
            port=6379,
            prefix="sifaka:multi:",
            ttl=86400,  # Keep for 24 hours
            use_redisearch=True
        )
        
        # Test Redis connection
        await redis_storage._get_client()
        backends.append(redis_storage)
        
        print("✅ RedisStorage connected")
        
        # Check RediSearch
        stats = await redis_storage.get_search_stats()
        if stats["available"]:
            print("🔍 RediSearch enabled - advanced search available!")
        else:
            print("📌 Basic Redis search (install Redis Stack for more)")
            
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("📁 Using FileStorage only")
    
    # Create MultiStorage
    storage = MultiStorage(
        backends=backends,
        primary_backend=1 if len(backends) > 1 else 0  # Use Redis for reads if available
    )
    
    status = storage.get_backend_status()
    print(f"\n📦 MultiStorage configured with {len(status['backends'])} backends:")
    for backend in status['backends']:
        print(f"   - {backend}")
    
    return storage


async def demo_multiple_improvements(storage: MultiStorage) -> None:
    """Create multiple improvement results using MultiStorage."""
    print("\n🚀 Part 2: Creating Multiple Improvements")
    print("=" * 50)
    print("\nEach improvement will be saved to all configured backends.")
    
    # Different texts to improve with different critics
    test_cases = [
        {
            "text": """
Machine learning is when computers learn from data. They use algorithms
to find patterns. Deep learning uses neural networks. It's very powerful.
""",
            "critics": ["style", "self_refine"],
            "description": "Technical explanation"
        },
        {
            "text": """The implementation of the new feature should be done carefully. 
We need to consider all edge cases. Testing is important. Documentation too.""",
            "critics": ["self_refine", "style"],
            "description": "Software development"
        },
        {
            "text": """Our product revolutionizes how businesses operate. It provides 
amazing benefits. Customers love it. You should buy it now.""",
            "critics": ["style", "constitutional"],
            "description": "Marketing copy"
        },
        {
            "text": """The research shows interesting results. The data supports our 
hypothesis. More studies are needed. This could be significant.""",
            "critics": ["reflexion", "style"],
            "description": "Research summary"
        }
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Improving text {i}/4: {test_case['description']}")
        
        result = await improve(
            test_case["text"],
            critics=test_case["critics"],
            max_iterations=2,
            storage=storage,  # MultiStorage handles saving to all backends
            config=Config(
                temperature=0.7,
                model="gpt-4o-mini",  # Fast model for demo
            ),
        )
        
        print(f"   ✅ Saved to all backends")
        print(f"   Critics used: {', '.join(test_case['critics'])}")
        print(f"   Final confidence: {result.critiques[-1].confidence:.2f}")


async def demo_search_capabilities(storage: MultiStorage) -> None:
    """Demonstrate search capabilities with MultiStorage."""
    print("\n🔍 Part 3: Search Capabilities")
    print("=" * 50)
    print("\nMultiStorage will use the best available search backend.")
    
    # Test different search queries
    search_tests = [
        {
            "query": "machine learning",
            "description": "Full-text search"
        },
        {
            "query": "improve clarity",
            "description": "Search in feedback"
        },
        {
            "query": "@critic:{style}",
            "description": "Filter by critic (RediSearch only)"
        },
        {
            "query": "@confidence:[0.8 1.0]",
            "description": "High confidence results (RediSearch only)"
        },
        {
            "query": "@critic:{style} @confidence:[0.7 1.0]",
            "description": "Combined filters (RediSearch only)"
        }
    ]
    
    for test in search_tests:
        print(f"\n🔎 {test['description']}: '{test['query']}'")
        try:
            results = await storage.search(test["query"], limit=3)
            print(f"   Found {len(results)} results")
            
            for j, result in enumerate(results, 1):
                critics_used = list(set(crit.critic for crit in result.critiques))
                max_confidence = max(crit.confidence for crit in result.critiques)
                print(f"   {j}. Critics: {', '.join(critics_used)}, Confidence: {max_confidence:.2f}")
                print(f"      Preview: {result.original_text[:60].strip()}...")
        except Exception as e:
            print(f"   ⚠️  Query not supported without RediSearch: {e}")


async def demo_storage_benefits(storage: MultiStorage) -> None:
    """Demonstrate the benefits of MultiStorage."""
    print("\n🎆 Part 4: MultiStorage Benefits")
    print("=" * 50)
    
    # Show storage status
    status = storage.get_backend_status()
    print(f"\n📦 Storage Configuration:")
    print(f"   Backends: {', '.join(status['backends'])}")
    print(f"   Primary for reads: {status['primary_name'] or 'First available'}")
    
    # List recent results
    print("\n📋 Recent improvements:")
    results = await storage.list(limit=5)
    print(f"   Found {len(results)} recent results")
    
    # Demonstrate fallback
    print("\n🔄 Fallback demonstration:")
    if len(results) > 0:
        result_id = results[0].id if hasattr(results[0], 'id') else "test-id"
        loaded = await storage.load(result_id)
        if loaded:
            print(f"   ✅ Loaded result from primary backend")
            print(f"   Original text preview: {loaded.original_text[:50]}...")
    
    # Show benefits
    print("\n🌟 Benefits of MultiStorage:")
    print("   1. 📦 Redundancy - Data saved to multiple locations")
    print("   2. 🚀 Performance - Fast Redis reads, permanent file backup")
    print("   3. 🔄 Fallback - Continues working if one backend fails")
    print("   4. 🔍 Search - Uses best available search (Redis if present)")
    print("   5. 📁 Archival - FileStorage provides permanent records")
    
    # Check if Redis has advanced search
    for backend in storage.backends:
        if isinstance(backend, RedisStorage):
            stats = await backend.get_search_stats()
            if stats["available"]:
                print("\n🔍 Advanced Redis Search Available:")
                
                # Try advanced search
                try:
                    # Get the Redis backend directly for advanced search
                    results = await backend.search_advanced(
                        critics=["style"],
                        min_confidence=0.7,
                        limit=3
                    )
                    print(f"   Found {len(results)} high-confidence style improvements")
                except Exception as e:
                    print(f"   Advanced search demo failed: {e}")
            break


if __name__ == "__main__":
    print("Prerequisites:")
    print("1. Start Redis:")
    print("   - Basic: docker run -d -p 6379:6379 redis")
    print("   - With RediSearch: docker run -d -p 6379:6379 redis/redis-stack")
    print("2. Install Python client: pip install redis")
    print("3. Set environment variable: OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()
    print("ℹ️  This example works with both basic Redis and Redis Stack.")
    print("   RediSearch provides advanced search capabilities.")
    print()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        asyncio.run(main())
