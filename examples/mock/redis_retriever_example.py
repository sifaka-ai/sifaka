#!/usr/bin/env python3
"""
Redis Retriever Example for Sifaka.

This example demonstrates how to use the RedisRetriever for caching and
performance optimization in Sifaka chains.

Requirements:
- Redis server running on localhost:6379
- redis Python package installed

The example shows:
1. Using RedisRetriever as a standalone document store
2. Using RedisRetriever as a caching wrapper around other retrievers
3. Performance benefits of caching
4. Integration with Sifaka chains
"""

import time
from sifaka.chain import Chain

from sifaka.models.base import create_model
from sifaka.retrievers.base import InMemoryRetriever, MCPServerConfig, MCPTransportType
from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.utils.logging import configure_logging


def check_redis_availability():
    """Check if Redis is available and running."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
        print("✅ Redis is available and running")
        return True
    except ImportError:
        print("❌ Redis package not installed. Install with: pip install redis>=5.0.0")
        return False
    except Exception as e:
        print(f"❌ Redis server not accessible: {e}")
        print("Make sure Redis is running on localhost:6379")
        return False


def example_standalone_redis_retriever():
    """Example 1: Using RedisRetriever as a standalone document store."""
    print("\n📚 Example 1: Standalone Redis Document Store")
    print("-" * 50)

    # Create MCP configuration for Redis
    mcp_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis",
    )

    # Create a standalone Redis retriever
    retriever = RedisRetriever(
        mcp_config=mcp_config,
        key_prefix="example:docs",
        cache_ttl=300,  # 5 minutes
        max_results=5,
    )

    # Clear any existing data
    retriever.clear_cache("example:docs:*")

    # Add documents about AI and programming
    documents = [
        (
            "ai_basics",
            "Artificial intelligence is the simulation of human intelligence in machines.",
        ),
        (
            "ml_intro",
            "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        ),
        (
            "dl_overview",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
        ),
        (
            "nlp_intro",
            "Natural language processing enables computers to understand and generate human language.",
        ),
        (
            "python_ai",
            "Python is the most popular programming language for AI and machine learning projects.",
        ),
        (
            "tensorflow",
            "TensorFlow is an open-source machine learning framework developed by Google.",
        ),
        (
            "pytorch",
            "PyTorch is a machine learning library that provides tensor computation and neural networks.",
        ),
    ]

    print("Adding documents to Redis...")
    for doc_id, text in documents:
        retriever.add_document(doc_id, text, metadata={"category": "ai_programming"})

    # Test retrieval
    query = "machine learning neural networks"
    print(f"\nQuerying: '{query}'")

    results = retriever.retrieve(query)
    print(f"Found {len(results)} relevant documents:")

    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc}")

    # Show cache stats
    stats = retriever.get_cache_stats()
    print(f"\nCache stats: {stats['stored_documents']} documents stored")

    return retriever


def example_caching_wrapper():
    """Example 2: Using RedisRetriever as a caching wrapper."""
    print("\n⚡ Example 2: Redis as Caching Wrapper")
    print("-" * 50)

    # Create a base retriever with some documents
    base_retriever = InMemoryRetriever()

    # Add documents about web development
    web_docs = {
        "react_intro": "React is a JavaScript library for building user interfaces.",
        "vue_basics": "Vue.js is a progressive framework for building user interfaces.",
        "angular_overview": "Angular is a platform for building mobile and desktop web applications.",
        "nodejs_intro": "Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine.",
        "express_framework": "Express.js is a minimal web application framework for Node.js.",
        "mongodb_basics": "MongoDB is a document-oriented NoSQL database program.",
        "postgresql_intro": "PostgreSQL is a powerful, open source object-relational database system.",
    }

    for doc_id, text in web_docs.items():
        base_retriever.add_document(doc_id, text, {"category": "web_development"})

    # Create MCP configuration for Redis
    mcp_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis",
    )

    # Create caching Redis retriever
    cached_retriever = RedisRetriever(
        mcp_config=mcp_config,
        base_retriever=base_retriever,
        key_prefix="example:cache",
        cache_ttl=180,  # 3 minutes
    )

    # Clear cache
    cached_retriever.clear_cache("example:cache:*")

    query = "javascript web development framework"
    print(f"Querying: '{query}'")

    # First query (cache miss)
    print("\n1st query (cache miss):")
    start_time = time.time()
    results1 = cached_retriever.retrieve(query)
    time1 = time.time() - start_time
    print(f"   Time: {time1:.4f}s, Results: {len(results1)}")

    # Second query (cache hit)
    print("\n2nd query (cache hit):")
    start_time = time.time()
    results2 = cached_retriever.retrieve(query)
    time2 = time.time() - start_time
    print(f"   Time: {time2:.4f}s, Results: {len(results2)}")

    # Show performance improvement
    if time1 > 0:
        speedup = time1 / time2 if time2 > 0 else float("inf")
        print(f"\n⚡ Cache speedup: {speedup:.1f}x faster")

    # Show results
    print(f"\nRetrieved documents:")
    for i, doc in enumerate(results2, 1):
        print(f"  {i}. {doc[:60]}...")

    return cached_retriever


def example_chain_integration():
    """Example 3: Using RedisRetriever in a Sifaka chain."""
    print("\n🔗 Example 3: Chain Integration with Redis Caching")
    print("-" * 50)

    # Create base retriever with programming knowledge
    base_retriever = InMemoryRetriever()

    programming_docs = {
        "python_basics": "Python is an interpreted, high-level programming language with dynamic semantics.",
        "python_data": "Python has excellent libraries for data science including pandas, numpy, and matplotlib.",
        "python_web": "Python web frameworks like Django and Flask make web development efficient.",
        "python_ai": "Python is the leading language for artificial intelligence and machine learning.",
        "best_practices": "Python code should follow PEP 8 style guidelines for readability.",
        "testing": "Python testing can be done with unittest, pytest, or other testing frameworks.",
        "deployment": "Python applications can be deployed using Docker, cloud platforms, or traditional servers.",
    }

    for doc_id, text in programming_docs.items():
        base_retriever.add_document(doc_id, text)

    # Create MCP configuration for Redis
    mcp_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis",
    )

    # Create cached retriever
    cached_retriever = create_redis_retriever(
        mcp_config=mcp_config,
        base_retriever=base_retriever,
        key_prefix="example:chain",
        cache_ttl=240,  # 4 minutes
    )

    # Clear cache
    cached_retriever.clear_cache("example:chain:*")

    # Create model and components
    model = create_model("mock:default")
    validator = LengthValidator(min_length=50, max_length=500)
    critic = ReflexionCritic(model=model)

    # Create chain with Redis retriever
    chain = Chain(
        model=model,
        prompt="Write a beginner's guide to Python programming, focusing on its strengths and use cases.",
        retrievers=[cached_retriever],  # New API uses list of retrievers
    )

    chain.validate_with(validator)
    chain.improve_with(critic)

    print("Running chain with Redis-cached retrieval...")

    # Run the chain
    result = chain.run()

    print(f"\n📝 Generated text ({len(result.text)} chars):")
    print(f"   {result.text[:100]}...")

    print(f"\n📚 Pre-generation context: {len(result.pre_generation_context)} documents")
    print(f"📚 Post-generation context: {len(result.post_generation_context)} documents")

    # Show cache effectiveness
    stats = cached_retriever.get_cache_stats()
    print(f"\n📊 Cache stats: {stats['cached_queries']} queries cached")

    return result


def example_performance_comparison():
    """Example 4: Performance comparison with and without caching."""
    print("\n📊 Example 4: Performance Comparison")
    print("-" * 50)

    # Create base retriever with many documents
    base_retriever = InMemoryRetriever()

    # Add many documents to simulate a larger dataset
    for i in range(100):
        doc_id = f"doc_{i:03d}"
        text = f"This is document {i} about various topics including technology, science, and programming."
        base_retriever.add_document(doc_id, text)

    # Test without caching
    print("Testing without caching...")
    queries = [
        "technology programming",
        "science research",
        "development software",
        "technology programming",  # Repeat query
        "science research",  # Repeat query
    ]

    start_time = time.time()
    for query in queries:
        base_retriever.retrieve(query)
    no_cache_time = time.time() - start_time

    print(f"   Time without caching: {no_cache_time:.4f}s")

    # Test with caching
    print("Testing with Redis caching...")
    mcp_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.WEBSOCKET,
        url="ws://localhost:8080/mcp/redis",
    )

    cached_retriever = RedisRetriever(
        mcp_config=mcp_config,
        base_retriever=base_retriever,
        key_prefix="example:perf",
        cache_ttl=120,
    )

    # Clear cache
    cached_retriever.clear_cache("example:perf:*")

    start_time = time.time()
    for query in queries:
        cached_retriever.retrieve(query)
    cache_time = time.time() - start_time

    print(f"   Time with caching: {cache_time:.4f}s")

    # Show improvement
    if no_cache_time > 0 and cache_time > 0:
        improvement = ((no_cache_time - cache_time) / no_cache_time) * 100
        print(f"\n⚡ Performance improvement: {improvement:.1f}%")

    # Clean up
    cached_retriever.clear_cache("example:perf:*")


def main():
    """Run all Redis retriever examples."""
    print("🚀 Redis Retriever Examples for Sifaka")
    print("=" * 50)

    # Configure logging
    configure_logging(level="INFO")

    # Check Redis availability
    if not check_redis_availability():
        print("\n❌ Redis is not available. Please install and start Redis server.")
        print("   Installation: https://redis.io/download")
        print("   Docker: docker run -d -p 6379:6379 redis:latest")
        return

    try:
        # Run examples
        example_standalone_redis_retriever()
        example_caching_wrapper()
        example_chain_integration()
        example_performance_comparison()

        print("\n🎉 All Redis retriever examples completed successfully!")
        print("\n💡 Key Benefits of Redis Caching:")
        print("   • Faster retrieval for repeated queries")
        print("   • Reduced computation overhead")
        print("   • Scalable caching across multiple processes")
        print("   • Configurable TTL for cache management")
        print("   • Can wrap any existing retriever")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure Redis is running and accessible.")


if __name__ == "__main__":
    main()
