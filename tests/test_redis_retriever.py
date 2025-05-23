#!/usr/bin/env python3
"""
Test script for RedisRetriever.

This script tests the RedisRetriever implementation, including both standalone
operation and caching wrapper functionality.

Requirements:
- Redis server running on localhost:6379
- redis Python package installed
"""

import time
from typing import List


def test_redis_availability():
    """Test if Redis is available and can be imported."""
    try:
        import redis

        print("✅ Redis package is available")
        return True
    except ImportError:
        print("❌ Redis package not available. Install with: pip install redis>=5.0.0")
        return False


def test_redis_connection():
    """Test if Redis server is running and accessible."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        client.ping()
        print("✅ Redis server is running and accessible")
        return True
    except Exception as e:
        print(f"❌ Redis server not accessible: {e}")
        print("Make sure Redis is running on localhost:6379")
        return False


def test_redis_retriever_import():
    """Test if RedisRetriever can be imported."""
    try:
        from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever

        print("✅ RedisRetriever import successful")
        return True
    except ImportError as e:
        print(f"❌ RedisRetriever import failed: {e}")
        return False


def test_standalone_redis_retriever():
    """Test RedisRetriever in standalone mode."""
    try:
        from sifaka.retrievers.redis import RedisRetriever

        # Create standalone Redis retriever
        retriever = RedisRetriever(
            key_prefix="test:sifaka:retriever",
            cache_ttl=60,  # 1 minute for testing
        )

        print("✅ RedisRetriever created successfully")

        # Clear any existing test data
        retriever.clear_cache("test:sifaka:retriever:*")

        # Add some test documents
        retriever.add_document("doc1", "Artificial intelligence is transforming technology.")
        retriever.add_document("doc2", "Machine learning algorithms improve with data.")
        retriever.add_document(
            "doc3", "Natural language processing enables human-computer interaction."
        )
        retriever.add_document("doc4", "Deep learning uses neural networks for complex tasks.")

        print("✅ Test documents added to Redis")

        # Test retrieval
        results = retriever.retrieve("artificial intelligence machine learning")
        print(f"✅ Retrieved {len(results)} documents")

        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc[:50]}...")

        # Test cache stats
        stats = retriever.get_cache_stats()
        print(
            f"✅ Cache stats: {stats['stored_documents']} documents, {stats['cached_queries']} cached queries"
        )

        # Clean up
        retriever.clear_cache("test:sifaka:retriever:*")
        print("✅ Test data cleaned up")

        return True

    except Exception as e:
        print(f"❌ Standalone RedisRetriever test failed: {e}")
        return False


def test_caching_redis_retriever():
    """Test RedisRetriever as a caching wrapper."""
    try:
        from sifaka.retrievers.redis import RedisRetriever
        from sifaka.retrievers.base import InMemoryRetriever

        # Create base retriever
        base_retriever = InMemoryRetriever()
        base_retriever.add_document("doc1", "Quantum computing uses quantum bits.")
        base_retriever.add_document("doc2", "Superposition allows multiple states simultaneously.")
        base_retriever.add_document("doc3", "Quantum entanglement connects qubits.")

        # Create caching Redis retriever
        cached_retriever = RedisRetriever(
            base_retriever=base_retriever,
            key_prefix="test:cache:retriever",
            cache_ttl=60,
        )

        print("✅ Caching RedisRetriever created successfully")

        # Clear cache
        cached_retriever.clear_cache("test:cache:retriever:*")

        # First retrieval (should hit base retriever and cache result)
        start_time = time.time()
        results1 = cached_retriever.retrieve("quantum computing")
        first_time = time.time() - start_time

        print(f"✅ First retrieval: {len(results1)} documents in {first_time:.4f}s")

        # Second retrieval (should hit cache)
        start_time = time.time()
        results2 = cached_retriever.retrieve("quantum computing")
        second_time = time.time() - start_time

        print(f"✅ Second retrieval: {len(results2)} documents in {second_time:.4f}s")

        # Verify results are the same
        if results1 == results2:
            print("✅ Cached results match original results")
        else:
            print("❌ Cached results don't match original results")
            return False

        # Check cache stats
        stats = cached_retriever.get_cache_stats()
        print(f"✅ Cache stats: {stats['cached_queries']} cached queries")

        # Clean up
        cached_retriever.clear_cache("test:cache:retriever:*")
        print("✅ Cache cleaned up")

        return True

    except Exception as e:
        print(f"❌ Caching RedisRetriever test failed: {e}")
        return False


def test_redis_retriever_with_thought():
    """Test RedisRetriever with Thought container."""
    try:
        from sifaka.retrievers.redis import RedisRetriever
        from sifaka.core.thought import Thought

        # Create retriever
        retriever = RedisRetriever(
            key_prefix="test:thought:retriever",
            cache_ttl=60,
        )

        # Clear cache
        retriever.clear_cache("test:thought:retriever:*")

        # Add documents
        retriever.add_document("doc1", "Python is a programming language.")
        retriever.add_document("doc2", "Machine learning frameworks use Python.")
        retriever.add_document("doc3", "Data science relies on Python libraries.")

        # Create thought
        thought = Thought(prompt="Tell me about Python programming")

        # Test pre-generation retrieval
        updated_thought = retriever.retrieve_for_thought(thought, is_pre_generation=True)

        print(
            f"✅ Pre-generation retrieval: {len(updated_thought.pre_generation_context)} documents"
        )

        # Add some generated text
        updated_thought = updated_thought.set_text("Python is a versatile programming language.")

        # Test post-generation retrieval
        final_thought = retriever.retrieve_for_thought(updated_thought, is_pre_generation=False)

        print(
            f"✅ Post-generation retrieval: {len(final_thought.post_generation_context)} documents"
        )

        # Clean up
        retriever.clear_cache("test:thought:retriever:*")
        print("✅ Thought test completed and cleaned up")

        return True

    except Exception as e:
        print(f"❌ RedisRetriever with Thought test failed: {e}")
        return False


def test_factory_function():
    """Test the create_redis_retriever factory function."""
    try:
        from sifaka.retrievers.redis import create_redis_retriever
        from sifaka.retrievers.base import MockRetriever

        # Test without base retriever
        retriever1 = create_redis_retriever(
            key_prefix="test:factory1",
            cache_ttl=30,
        )
        print("✅ Factory function created standalone retriever")

        # Test with base retriever
        base = MockRetriever()
        retriever2 = create_redis_retriever(
            base_retriever=base,
            key_prefix="test:factory2",
            cache_ttl=30,
        )
        print("✅ Factory function created caching retriever")

        # Clean up
        retriever1.clear_cache("test:factory1:*")
        retriever2.clear_cache("test:factory2:*")

        return True

    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False


def main():
    """Run all Redis retriever tests."""
    print("🧪 Testing RedisRetriever Implementation")
    print("=" * 50)

    # Check prerequisites
    if not test_redis_availability():
        return

    if not test_redis_connection():
        return

    if not test_redis_retriever_import():
        return

    print("\n📋 Running RedisRetriever Tests")
    print("-" * 30)

    tests = [
        ("Standalone RedisRetriever", test_standalone_redis_retriever),
        ("Caching RedisRetriever", test_caching_redis_retriever),
        ("RedisRetriever with Thought", test_redis_retriever_with_thought),
        ("Factory Function", test_factory_function),
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

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All RedisRetriever tests passed!")
    else:
        print("⚠️  Some tests failed. Check Redis connection and dependencies.")


if __name__ == "__main__":
    main()
