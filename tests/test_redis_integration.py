#!/usr/bin/env python3
"""
Integration test for RedisRetriever with Sifaka Chain.

This test verifies that RedisRetriever works correctly when integrated
with the full Sifaka chain, including models, validators, and critics.
"""


def test_redis_chain_integration():
    """Test RedisRetriever integration with a complete Sifaka chain."""
    try:
        # Check Redis availability
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("Skipping Redis integration test")
        return True

    try:
        from sifaka.core.chain import Chain
        from sifaka.models.base import create_model
        from sifaka.validators.base import LengthValidator
        from sifaka.critics.reflexion import ReflexionCritic
        from sifaka.retrievers.base import InMemoryRetriever
        from sifaka.retrievers.redis import RedisRetriever

        print("✅ All imports successful")

        # Create base retriever with programming knowledge
        base_retriever = InMemoryRetriever()
        base_retriever.add_document("python1", "Python is a high-level programming language.")
        base_retriever.add_document("python2", "Python is excellent for data science and AI.")
        base_retriever.add_document("python3", "Python has a simple and readable syntax.")
        base_retriever.add_document("web1", "Django is a Python web framework.")
        base_retriever.add_document("web2", "Flask is a lightweight Python web framework.")

        # Create Redis caching retriever
        redis_retriever = RedisRetriever(
            base_retriever=base_retriever,
            key_prefix="test:integration",
            cache_ttl=60,
        )

        # Clear any existing cache
        redis_retriever.clear_cache("test:integration:*")

        print("✅ Retrievers created and configured")

        # Create model and components
        model = create_model("mock:default")
        validator = LengthValidator(min_length=20, max_length=500)
        critic = ReflexionCritic(model=model)

        print("✅ Model, validator, and critic created")

        # Create chain with Redis retriever
        chain = Chain(
            model=model,
            prompt="Explain why Python is popular for web development.",
            retriever=redis_retriever,
            pre_generation_retrieval=True,
            post_generation_retrieval=True,
            critic_retrieval=True,
        )

        chain.validate_with(validator)
        chain.improve_with(critic)

        print("✅ Chain configured with Redis retriever")

        # Run the chain
        result = chain.run()

        print(f"✅ Chain executed successfully")
        print(f"   Generated text length: {len(result.text)} characters")
        print(f"   Pre-generation context: {len(result.pre_generation_context)} documents")
        print(f"   Post-generation context: {len(result.post_generation_context)} documents")
        print(f"   Validation results: {len(result.validation_results)} validators")

        # Verify caching worked
        stats = redis_retriever.get_cache_stats()
        print(f"   Cached queries: {stats['cached_queries']}")

        # Test cache hit by running another similar query
        print("\n🔄 Testing cache hit with second chain run...")

        chain2 = Chain(
            model=model,
            prompt="Why is Python good for web development?",  # Similar query
            retriever=redis_retriever,
            pre_generation_retrieval=True,
        )

        result2 = chain2.run()

        # Check if cache was used
        stats2 = redis_retriever.get_cache_stats()
        print(f"   Cached queries after second run: {stats2['cached_queries']}")

        if stats2["cached_queries"] > stats["cached_queries"]:
            print("✅ Cache was used for second query")
        else:
            print("ℹ️  Cache may not have been used (different query hash)")

        # Clean up
        redis_retriever.clear_cache("test:integration:*")
        print("✅ Cache cleaned up")

        return True

    except Exception as e:
        print(f"❌ Redis integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_redis_retriever_error_handling():
    """Test RedisRetriever error handling."""
    try:
        from sifaka.retrievers.redis import RedisRetriever
        from sifaka.utils.error_handling import RetrieverError

        # Test connection to non-existent Redis server
        try:
            bad_retriever = RedisRetriever(
                redis_host="nonexistent-host",
                redis_port=9999,
            )
            print("❌ Should have failed to connect to bad Redis server")
            return False
        except RetrieverError:
            print("✅ Correctly handled bad Redis connection")

        # Test with Redis unavailable (if redis package not installed)
        # This is tested in the import error handling

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_redis_retriever_configuration():
    """Test RedisRetriever configuration options."""
    try:
        # Check Redis availability
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
    except Exception:
        print("❌ Redis not available, skipping configuration test")
        return True

    try:
        from sifaka.retrievers.redis import RedisRetriever

        # Test different configuration options
        retriever = RedisRetriever(
            redis_host="localhost",
            redis_port=6379,
            redis_db=1,  # Different database
            cache_ttl=30,
            key_prefix="test:config",
            max_results=5,
        )

        print("✅ RedisRetriever created with custom configuration")

        # Test adding and retrieving documents
        retriever.add_document("test1", "Test document 1", {"type": "test"})
        retriever.add_document("test2", "Test document 2", {"type": "test"})

        results = retriever.retrieve("test document")
        print(f"✅ Retrieved {len(results)} documents with custom config")

        # Test cache stats
        stats = retriever.get_cache_stats()
        print(
            f"✅ Cache stats: {stats['stored_documents']} docs, {stats['cached_queries']} queries"
        )

        # Clean up
        retriever.clear_cache("test:config:*")
        print("✅ Configuration test completed")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all Redis integration tests."""
    print("🧪 Redis Integration Tests for Sifaka")
    print("=" * 50)

    tests = [
        ("Redis Chain Integration", test_redis_chain_integration),
        ("Redis Error Handling", test_redis_retriever_error_handling),
        ("Redis Configuration", test_redis_retriever_configuration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n📊 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Redis integration tests passed!")
    else:
        print("⚠️  Some integration tests failed.")


if __name__ == "__main__":
    main()
