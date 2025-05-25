#!/usr/bin/env python3
"""
Unified Storage Example for Sifaka - Replacing Redis Retriever.

This example demonstrates how to use the new unified storage architecture
with CachedRetriever for caching and performance optimization in Sifaka chains.

Setup Instructions:
1. Start Docker Redis:
   docker run -d -p 6379:6379 redis:latest

2. Install Redis MCP Server:
   npm install -g @modelcontextprotocol/server-redis

3. Install Milvus MCP Server:
   npm install -g @milvus-io/mcp-server-milvus

4. Run this example:
   python examples/mock/redis_retriever_example.py

The example shows:
1. Using unified storage architecture with 3-tier caching
2. CachedRetriever replacing the old dual-mode RedisRetriever
3. Performance benefits of memory → cache → persistence pattern
4. Integration with Sifaka chains
"""

import time
from sifaka.core.chain import Chain

from sifaka.models.base import create_model
from sifaka.retrievers import InMemoryRetriever
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage import SifakaStorage
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


def example_standalone_cached_retriever():
    """Example 1: Using CachedRetriever with unified storage architecture."""
    print("\n📚 Example 1: Unified Storage with In-Memory Base Retriever")
    print("-" * 50)

    # Create MCP configurations for unified storage
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    # Create unified storage manager
    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        memory_size=100,
        cache_ttl=300,  # 5 minutes
    )

    # Create base retriever and wrap with caching
    base_retriever = InMemoryRetriever()
    retriever = storage.get_retriever_cache(base_retriever)

    # Clear any existing cache
    retriever.clear_cache()

    # Add documents about AI and programming to base retriever
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

    print("Adding documents to base retriever...")
    for doc_id, text in documents:
        base_retriever.add_document(doc_id, text, metadata={"category": "ai_programming"})

    # Test retrieval with 3-tier caching
    query = "machine learning neural networks"
    print(f"\nQuerying: '{query}'")

    results = retriever.retrieve(query)
    print(f"Found {len(results)} relevant documents:")

    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc}")

    # Show cache stats
    stats = retriever.get_cache_stats()
    print(f"\nUnified storage stats:")
    print(f"  Memory hits: {stats['memory']['hits']}")
    print(f"  Cache hits: {stats['cache_performance']['hits']}")
    print(f"  Total requests: {stats['cache_performance']['total_requests']}")

    return retriever


def example_caching_wrapper():
    """Example 2: Using CachedRetriever with unified storage."""
    print("\n⚡ Example 2: Unified Storage Caching Performance")
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

    # Create MCP configurations for unified storage
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    # Create unified storage and cached retriever
    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        cache_ttl=180,  # 3 minutes
    )

    cached_retriever = storage.get_retriever_cache(base_retriever)

    # Clear cache
    cached_retriever.clear_cache()

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
    """Example 3: Using CachedRetriever in a Sifaka chain."""
    print("\n🔗 Example 3: Chain Integration with Unified Storage")
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

    # Create MCP configurations for unified storage
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    # Create unified storage and cached retriever
    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        cache_ttl=240,  # 4 minutes
    )

    cached_retriever = storage.get_retriever_cache(base_retriever)

    # Clear cache
    cached_retriever.clear_cache()

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
    cached_queries = cached_retriever.get_cached_queries()
    print(f"\n📊 Cache stats: {len(cached_queries)} queries cached")
    print(f"📊 Cache hit rate: {stats['cache_performance']['hit_rate']:.1%}")

    return result


class SlowRetriever:
    """A deliberately slow retriever to simulate expensive operations."""

    def __init__(self, delay_seconds=0.1):
        self.delay_seconds = delay_seconds
        self.documents = {}

    def add_document(self, doc_id, text, metadata=None):
        self.documents[doc_id] = text

    def retrieve(self, query):
        # Simulate expensive computation (e.g., vector similarity, API calls)
        time.sleep(self.delay_seconds)

        # Simple keyword matching
        query_terms = query.lower().split()
        results = []

        for doc_id, text in self.documents.items():
            if any(term in text.lower() for term in query_terms):
                results.append(text)

        return results[:3]  # Return top 3


def example_performance_comparison():
    """Example 4: Performance comparison showing when Redis caching helps."""
    print("\n📊 Example 4: When Redis Caching Actually Helps")
    print("-" * 50)
    print("💡 Redis caching is beneficial when the base retrieval is expensive")
    print("   (e.g., vector search, API calls, complex computations)")

    # Create a slow retriever to simulate expensive operations
    slow_retriever = SlowRetriever(delay_seconds=0.1)  # 100ms delay per query

    # Add documents
    for i in range(20):
        doc_id = f"doc_{i:03d}"
        text = f"Document {i} about technology, science, programming, and research topics."
        slow_retriever.add_document(doc_id, text)

    # Test queries with repeated patterns (realistic scenario)
    queries = [
        "technology programming",
        "science research",
        "programming development",
        "technology programming",  # Repeat - cache hit
        "science research",  # Repeat - cache hit
        "programming development",  # Repeat - cache hit
        "new unique query",  # New query - cache miss
        "technology programming",  # Repeat - cache hit
    ]

    print(f"\nTesting with {len(queries)} queries ({len(set(queries))} unique)")
    print("Each query simulates 100ms of expensive computation...")

    # Test without caching (direct slow retriever)
    print("\n1. Without caching (direct expensive retrieval):")
    start_time = time.time()
    for i, query in enumerate(queries, 1):
        results = slow_retriever.retrieve(query)
        print(f"   Query {i}: {len(results)} results (100ms)")
    no_cache_time = time.time() - start_time
    print(f"   Total time: {no_cache_time:.3f}s")

    # Test with unified storage caching
    print("\n2. With unified storage caching:")
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        cache_ttl=120,
    )

    cached_retriever = storage.get_retriever_cache(slow_retriever)

    # Clear cache to start fresh
    cached_retriever.clear_cache()

    start_time = time.time()
    for i, query in enumerate(queries, 1):
        query_start = time.time()
        results = cached_retriever.retrieve(query)
        query_time = time.time() - query_start

        # Determine if this was likely a cache hit or miss
        cache_status = "miss" if query_time > 0.05 else "hit"
        print(
            f"   Query {i}: {len(results)} results ({query_time*1000:.0f}ms - cache {cache_status})"
        )

    cache_time = time.time() - start_time
    print(f"   Total time: {cache_time:.3f}s")

    # Show the real benefit
    if no_cache_time > 0 and cache_time > 0:
        speedup = no_cache_time / cache_time
        time_saved = no_cache_time - cache_time
        print(f"\n⚡ Results:")
        print(f"   • Speedup: {speedup:.1f}x faster")
        print(f"   • Time saved: {time_saved:.3f}s ({time_saved/no_cache_time*100:.1f}%)")
        print(
            f"   • Cache hits avoid expensive {slow_retriever.delay_seconds*1000:.0f}ms computations"
        )

    print(f"\n💡 Key Insight:")
    print(f"   Redis caching trades small network overhead (~1-5ms)")
    print(f"   for avoiding expensive operations ({slow_retriever.delay_seconds*1000:.0f}ms)")
    print(f"   The more expensive your base retrieval, the more beneficial caching becomes!")

    # Clean up
    cached_retriever.clear_cache()


def main():
    """Run all unified storage examples."""
    print("🚀 Unified Storage Examples for Sifaka")
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
        example_standalone_cached_retriever()
        example_caching_wrapper()
        example_chain_integration()
        example_performance_comparison()

        print("\n🎉 All unified storage examples completed successfully!")
        print("\n💡 Key Benefits of Unified Storage Architecture:")
        print("   • Consistent 3-tier caching (memory → Redis → Milvus)")
        print("   • Predictable performance characteristics")
        print("   • Vector similarity search capabilities")
        print("   • Automatic persistence without blocking")
        print("   • Clean separation of concerns")
        print("   • Easy testing and debugging")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure Redis and Milvus MCP servers are available.")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
