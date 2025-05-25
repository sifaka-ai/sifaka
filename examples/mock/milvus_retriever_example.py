#!/usr/bin/env python3
"""Unified Storage Example for Sifaka - Replacing Milvus Retriever.

This example demonstrates how to use the new unified storage architecture
with vector search capabilities using the official Milvus MCP server.

Setup Instructions:
1. Start Docker Redis:
   docker run -d -p 6379:6379 redis:latest

2. Install Redis MCP Server:
   npm install -g @modelcontextprotocol/server-redis

3. Install Milvus MCP Server:
   npm install -g @milvus-io/mcp-server-milvus

4. Run this example:
   python examples/mock/milvus_retriever_example.py

The example demonstrates:
1. Using unified storage architecture with vector search capabilities
2. Connecting to real MCP infrastructure (Redis + Milvus)
3. Semantic search with 3-tier caching (memory → Redis → Milvus)
4. Integration with Sifaka chains for context-aware generation
"""

from typing import Dict

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics import ReflexionCritic
from sifaka.retrievers import InMemoryRetriever
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.storage import SifakaStorage

VECTOR_DB_AVAILABLE = True  # Always available with unified storage


def create_sample_documents() -> Dict[str, str]:
    """Create a collection of sample documents about AI and technology."""
    return {
        "ai_overview": """
        Artificial Intelligence (AI) is a branch of computer science that aims to create
        intelligent machines that can perform tasks that typically require human intelligence.
        These tasks include learning, reasoning, problem-solving, perception, and language
        understanding. AI has applications in various fields including healthcare, finance,
        transportation, and entertainment.
        """,
        "machine_learning": """
        Machine Learning is a subset of artificial intelligence that enables computers to
        learn and improve from experience without being explicitly programmed. It uses
        algorithms and statistical models to analyze and draw inferences from patterns
        in data. Common types include supervised learning, unsupervised learning, and
        reinforcement learning.
        """,
        "deep_learning": """
        Deep Learning is a specialized subset of machine learning that uses artificial
        neural networks with multiple layers to model and understand complex patterns
        in data. It has been particularly successful in areas like image recognition,
        natural language processing, and speech recognition. Deep learning models can
        automatically discover representations from raw data.
        """,
        "nlp": """
        Natural Language Processing (NLP) is a field of artificial intelligence that
        focuses on the interaction between computers and human language. It involves
        developing algorithms and models that can understand, interpret, and generate
        human language in a valuable way. Applications include machine translation,
        sentiment analysis, and chatbots.
        """,
        "computer_vision": """
        Computer Vision is a field of artificial intelligence that trains computers to
        interpret and understand the visual world. Using digital images from cameras
        and videos and deep learning models, machines can accurately identify and
        classify objects and react to what they see. Applications include autonomous
        vehicles, medical imaging, and facial recognition.
        """,
        "robotics": """
        Robotics is an interdisciplinary field that integrates computer science and
        engineering to design, construct, and operate robots. Modern robotics combines
        artificial intelligence, machine learning, and computer vision to create
        autonomous systems that can perform complex tasks in various environments,
        from manufacturing to space exploration.
        """,
        "quantum_computing": """
        Quantum Computing is a revolutionary computing paradigm that uses quantum
        mechanical phenomena like superposition and entanglement to process information.
        Unlike classical computers that use bits, quantum computers use quantum bits
        (qubits) that can exist in multiple states simultaneously, potentially solving
        certain problems exponentially faster than classical computers.
        """,
        "blockchain": """
        Blockchain is a distributed ledger technology that maintains a continuously
        growing list of records, called blocks, which are linked and secured using
        cryptography. It provides transparency, security, and decentralization, making
        it useful for applications like cryptocurrencies, supply chain management,
        and digital identity verification.
        """,
    }


def demonstrate_basic_retrieval():
    """Demonstrate basic document storage and retrieval with unified storage."""
    print("=== Unified Storage Vector Retrieval Demo ===")

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

    print("Using unified storage with 3-tier architecture")
    print("Memory → Redis → Milvus for optimal performance")

    # Create unified storage manager
    storage = SifakaStorage(
        redis_config=redis_config,
        milvus_config=milvus_config,
        memory_size=50,
        cache_ttl=300,  # 5 minutes
    )

    # Create base retriever and wrap with caching
    base_retriever = InMemoryRetriever()
    retriever = storage.get_retriever_cache(base_retriever)

    print(f"Created unified storage with vector search capabilities")

    # Clear any existing cache
    retriever.clear_cache()

    # Add sample documents
    documents = create_sample_documents()
    print(f"\nAdding {len(documents)} documents to vector database...")

    for doc_id, text in documents.items():
        retriever.add_document(
            doc_id=doc_id, text=text.strip(), metadata={"category": "technology", "source": "demo"}
        )

    # Get collection statistics
    stats = retriever.get_collection_stats()
    print(f"Collection stats: {stats}")

    # Add a small delay to ensure documents are indexed
    import time

    print("Waiting for documents to be indexed...")
    time.sleep(2)

    # Test different types of queries
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Tell me about computer vision applications",
        "What are the benefits of quantum computing?",
        "How does blockchain provide security?",
    ]

    print("\n=== Search Results ===")
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)
        for i, doc_text in enumerate(results, 1):
            # Truncate for display (results are now strings, not Document objects)
            preview = doc_text[:100].replace("\n", " ").strip()
            print(f"  {i}. {preview}...")

        if not results:
            print("  No results found.")

    # Show cache stats
    cache_stats = retriever.get_cache_stats()
    print(f"\nCache performance:")
    print(f"  Memory hits: {cache_stats['memory']['hits']}")
    print(f"  Cache hits: {cache_stats['cache_performance']['hits']}")
    print(f"  Hit rate: {cache_stats['cache_performance']['hit_rate']:.2f}")

    print("\nBasic retrieval demo completed!")


def demonstrate_chain_integration():
    """Demonstrate using unified storage in a Sifaka chain."""
    print("\n=== Chain Integration Demo ===")

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
        memory_size=30,
        cache_ttl=240,  # 4 minutes
    )

    base_retriever = InMemoryRetriever()
    retriever = storage.get_retriever_cache(base_retriever)

    # Clear and populate with documents
    retriever.clear_cache()
    documents = create_sample_documents()

    for doc_id, text in documents.items():
        base_retriever.add_document(
            doc_id=doc_id, text=text.strip(), metadata={"category": "technology", "source": "demo"}
        )

    # Create model and components
    model = create_model("mock:default")
    validator = LengthValidator(min_length=100, max_length=800)
    critic = ReflexionCritic(model=model)

    # Create chain with vector retriever
    chain = Chain(
        model=model,
        prompt="Write a comprehensive explanation of how artificial intelligence and machine learning are transforming modern technology. Include specific examples and applications.",
        retrievers=[retriever],  # New API uses list of retrievers
    )

    # Add validator and critic
    chain.validate_with(validator)
    chain.improve_with(critic)

    print("Running chain with unified storage retrieval...")
    result = chain.run()

    print(f"\nFinal result (iteration {result.iteration}):")
    print(f"Text: {result.text[:200]}...")

    print(f"\nPre-generation context documents: {len(result.pre_generation_context)}")
    for i, doc in enumerate(result.pre_generation_context[:2], 1):
        preview = doc.text[:80].replace("\n", " ").strip()
        print(f"  {i}. {preview}...")

    if result.post_generation_context:
        print(f"\nPost-generation context documents: {len(result.post_generation_context)}")
        for i, doc in enumerate(result.post_generation_context[:2], 1):
            preview = doc.text[:80].replace("\n", " ").strip()
            print(f"  {i}. {preview}...")

    # Show cache stats
    cache_stats = retriever.get_cache_stats()
    print(f"\nCache performance:")
    print(f"  Memory hits: {cache_stats['memory']['hits']}")
    print(f"  Cache hits: {cache_stats['cache_performance']['hits']}")
    print(f"  Hit rate: {cache_stats['cache_performance']['hit_rate']:.2f}")

    print("\nChain integration demo completed!")


def main():
    """Run the unified storage examples."""
    # Enable debug logging
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Unified Storage Example for Sifaka")
    print("=" * 60)
    print("This example demonstrates using the unified storage architecture")
    print("with 3-tier caching (memory → Redis → Milvus) for optimal performance.")
    print()

    try:
        # Run basic retrieval demo
        demonstrate_basic_retrieval()

        # Run chain integration demo
        demonstrate_chain_integration()

        print("\n🎉 All unified storage examples completed successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   • Consistent 3-tier caching across all components")
        print("   • Vector similarity search with automatic persistence")
        print("   • Predictable performance characteristics")
        print("   • Clean separation of concerns")
        print("   • Easy integration with Sifaka chains")

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Unified storage example completed!")
    print()
    print("Next steps:")
    print("1. Make sure Redis is running: docker run -d -p 6379:6379 redis:latest")
    print(
        "2. Install MCP servers: npm install -g @modelcontextprotocol/server-redis @milvus-io/mcp-server-milvus"
    )
    print("3. Explore the unified storage architecture in your own chains")


if __name__ == "__main__":
    main()
