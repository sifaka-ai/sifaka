"""Vector Database Retriever Example for Sifaka.

This example demonstrates how to use vector database retrievers with the Sifaka framework.
It shows how to set up and use different vector database retrievers for context retrieval.

The example uses mock components for simplicity, but the same approach can be used
with real vector databases like Milvus, Pinecone, or Weaviate.
"""

import logging

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought, Document
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.models.base import create_model
from sifaka.retrievers import InMemoryRetriever
from sifaka.utils.logging import configure_logging
from sifaka.validators.base import LengthValidator


def print_thought_details(thought: Thought) -> None:
    """Print detailed information about a thought."""
    print(f"\n📝 Final Result:")
    print(f"   Prompt: {thought.prompt}")
    print(f"   Generated Text: {thought.text}")
    print(f"   Iteration: {thought.iteration}")
    print(f"   Chain ID: {thought.chain_id}")

    # Print context information
    if thought.pre_generation_context:
        print(f"\n🔍 Pre-generation Context ({len(thought.pre_generation_context)} documents):")
        for i, doc in enumerate(thought.pre_generation_context[:3]):  # Show first 3
            print(f"   {i+1}. {doc.text[:100]}...")

    if thought.post_generation_context:
        print(f"\n🔍 Post-generation Context ({len(thought.post_generation_context)} documents):")
        for i, doc in enumerate(thought.post_generation_context[:3]):  # Show first 3
            print(f"   {i+1}. {doc.text[:100]}...")

    # Print validation results
    if thought.validation_results:
        print(f"\n✅ Validation Results:")
        for validator_name, result in thought.validation_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {validator_name}: {status}")
            if result.message:
                print(f"      Message: {result.message}")

    # Print critic feedback
    if thought.critic_feedback:
        print(f"\n🎯 Critic Feedback:")
        for feedback in thought.critic_feedback:
            print(f"   {feedback.critic_name}:")
            print(f"      Confidence: {feedback.confidence}")
            if feedback.suggestions:
                print(f"      Suggestions: {feedback.suggestions}")


def create_mock_vector_retriever() -> InMemoryRetriever:
    """Create a mock vector database retriever with sample documents."""
    retriever = InMemoryRetriever()

    # Add some sample documents that would typically come from a vector database
    sample_docs = [
        Document(
            text="Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently.",
            metadata={"source": "vector_db_guide", "type": "definition"},
        ),
        Document(
            text="Popular vector databases include Milvus, Pinecone, Weaviate, and Qdrant, each with different strengths.",
            metadata={"source": "vector_db_comparison", "type": "overview"},
        ),
        Document(
            text="Vector similarity search uses metrics like cosine similarity, Euclidean distance, or dot product.",
            metadata={"source": "similarity_metrics", "type": "technical"},
        ),
        Document(
            text="Embedding models convert text into dense vector representations for semantic search.",
            metadata={"source": "embeddings_guide", "type": "concept"},
        ),
        Document(
            text="Vector databases enable semantic search, recommendation systems, and RAG applications.",
            metadata={"source": "use_cases", "type": "applications"},
        ),
    ]

    # Add documents to the mock retriever
    for i, doc in enumerate(sample_docs):
        retriever.add_document(f"doc_{i}", doc.text, doc.metadata)

    return retriever


def main() -> None:
    """Run the vector database retriever example."""
    print("🚀 Vector Database Retriever Example - Sifaka Framework")
    print("=" * 65)

    # Configure logging
    configure_logging(level=logging.INFO)

    # Create components
    print("\n📦 Setting up components...")

    # Create a mock model
    model = create_model("mock:vector-model")

    # Create a mock vector database retriever
    vector_retriever = create_mock_vector_retriever()
    print("✅ Created mock vector database retriever with sample documents")

    # Create validators
    length_validator = LengthValidator(min_length=100, max_length=1000)

    # Create a critic
    critic = ReflexionCritic(model_name="mock:vector-critic")

    print("✅ Components created successfully!")

    # Define the prompt
    prompt = "Explain what vector databases are and how they work, including their main use cases and benefits."

    # Create and configure the chain
    print("\n🔗 Creating and configuring chain...")

    chain = Chain(
        model=model,
        prompt=prompt,
        model_retrievers=[vector_retriever],  # Use vector database retriever for model context
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
    )

    # Add validators and critics using the fluent API
    chain.validate_with(length_validator).improve_with(critic)

    print("✅ Chain configured with vector database retriever!")

    # Run the chain
    print("\n🏃 Running the chain...")
    result = chain.run()

    print("✅ Chain execution completed!")

    # Print the result
    print_thought_details(result)

    # Print performance summary
    print("\n📊 Performance Summary:")
    performance = chain.get_performance_summary()
    print(f"   Total Operations: {performance['total_operations']}")
    print(f"   Total Time: {performance['total_time']:.3f}s")
    print(f"   Average Time per Operation: {performance['avg_time_per_operation']:.3f}s")

    # Show retrieval operations specifically
    operations = performance.get("operations", {})
    retrieval_ops = {k: v for k, v in operations.items() if "retrieval" in k}
    if retrieval_ops:
        print(f"\n🔍 Retrieval Operations:")
        for op_name, metrics in retrieval_ops.items():
            print(
                f"   {op_name}: {metrics['call_count']} calls, {metrics['total_time']:.3f}s total"
            )

    print(f"\n🎉 Vector database retriever example completed successfully!")


if __name__ == "__main__":
    main()
