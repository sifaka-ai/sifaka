#!/usr/bin/env python3
"""Example demonstrating MilvusRetriever for semantic search.

This example shows how to use the MilvusRetriever to store documents
in a Milvus vector database and perform semantic similarity search.

The example demonstrates:
1. Creating a MilvusRetriever with Milvus Lite (embedded)
2. Adding documents to the vector database
3. Performing semantic search queries
4. Using the retriever in a Sifaka chain for context-aware generation
5. Comparing results with different query types

Requirements:
    - pymilvus>=2.4.0 (automatically installed with Sifaka)
    - The example uses Milvus Lite which doesn't require a separate server

Usage:
    python examples/mock/vector_db_retriever_example.py
"""

from typing import Dict, Any

from sifaka.chain import Chain
from sifaka.models.base import create_model
from sifaka.validators.base import LengthValidator
from sifaka.critics import ReflexionCritic

try:
    from sifaka.retrievers.vector_db_base import create_vector_db_retriever, create_milvus_retriever
    from sifaka.retrievers.milvus import MilvusRetriever

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    print(
        "Vector DB retrievers not available. Please install pymilvus: pip install 'pymilvus[model]'"
    )


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
    """Demonstrate basic document storage and retrieval."""
    print("=== Basic VectorDB Retrieval Demo ===")

    if not VECTOR_DB_AVAILABLE:
        print("Skipping VectorDB demo - pymilvus not available")
        return

    # Create retriever with Milvus Lite (embedded database)
    retriever = MilvusRetriever(
        collection_name="demo_collection",
        embedding_model="BAAI/bge-m3",  # BGE-M3 model for embeddings
        dimension=1024,  # BGE-M3 dense embedding dimension
        max_results=3,
    )

    print(f"Created MilvusRetriever with collection: {retriever.collection_name}")

    # Clear any existing data
    retriever.clear_collection()

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

    # Test different types of queries
    queries = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Tell me about computer vision applications",
        "What are the benefits of quantum computing?",
        "How does blockchain provide security?",
    ]

    print("\n=== Semantic Search Results ===")
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)

        for i, doc in enumerate(results, 1):
            # Truncate for display
            preview = doc[:100].replace("\n", " ").strip()
            print(f"  {i}. {preview}...")

    # Cleanup
    retriever.disconnect()
    print("\nBasic retrieval demo completed!")


def demonstrate_chain_integration():
    """Demonstrate using MilvusRetriever in a Sifaka chain."""
    print("\n=== Chain Integration Demo ===")

    if not VECTOR_DB_AVAILABLE:
        print("Skipping chain integration demo - pymilvus not available")
        return

    # Create retriever and populate with documents
    retriever = MilvusRetriever(
        collection_name="chain_demo_collection",
        max_results=2,  # Fewer results for cleaner output
    )

    # Clear and populate
    retriever.clear_collection()
    documents = create_sample_documents()

    for doc_id, text in documents.items():
        retriever.add_document(doc_id, text.strip())

    # Create model and components
    model = create_model("mock:default")
    validator = LengthValidator(min_length=100, max_length=800)
    critic = ReflexionCritic(model=model)

    # Create chain with vector retriever
    chain = Chain(
        model=model,
        prompt="Write a comprehensive explanation of how artificial intelligence and machine learning are transforming modern technology. Include specific examples and applications.",
        retriever=retriever,
        pre_generation_retrieval=True,
        post_generation_retrieval=True,
        critic_retrieval=True,
    )

    # Add validator and critic
    chain.validate_with(validator)
    chain.improve_with(critic)

    print("Running chain with VectorDB retrieval...")
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

    # Cleanup
    retriever.disconnect()
    print("\nChain integration demo completed!")


def main():
    """Run the VectorDB retriever examples."""
    print("VectorDB Retriever Example")
    print("=" * 50)

    try:
        # Run basic retrieval demo
        demonstrate_basic_retrieval()

        # Run chain integration demo
        demonstrate_chain_integration()

    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("VectorDB Retriever example completed!")


if __name__ == "__main__":
    main()
