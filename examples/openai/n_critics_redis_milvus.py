#!/usr/bin/env python3
"""OpenAI N-Critics with Redis and Milvus Retrieval Example.

This example demonstrates:
- OpenAI model with Redis retrieval for context
- N-Critics ensemble with Milvus retrieval for comprehensive feedback
- No validators (validation bypassed)
- Exactly 1 retry iteration

The chain will generate content about machine learning with Redis providing
context to the model and Milvus providing specialized knowledge to the critics.
"""

import os

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.models.openai import OpenAIModel
from sifaka.retrievers.simple import InMemoryRetriever
from sifaka.storage.milvus import MilvusStorage
from sifaka.storage.redis import RedisStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_redis_retriever():
    """Set up Redis retriever with ML context documents."""

    # Create Redis MCP configuration (using official Redis MCP server)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory ../../mcp/mcp-redis src/main.py",
    )

    # Create Redis storage for model context
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:ml_context")

    # Create in-memory retriever and populate with ML context
    retriever = InMemoryRetriever()

    # Add machine learning context documents
    ml_documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Supervised learning uses labeled training data to learn a mapping function from inputs to outputs, commonly used for classification and regression tasks.",
        "Unsupervised learning finds hidden patterns in data without labeled examples, including clustering, dimensionality reduction, and association rules.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in large amounts of data.",
        "Reinforcement learning trains agents to make decisions by learning from rewards and penalties in an environment.",
        "Feature engineering involves selecting, modifying, or creating new features from raw data to improve model performance.",
        "Cross-validation is a technique to assess model performance by partitioning data into training and testing sets multiple times.",
        "Overfitting occurs when a model learns the training data too well and fails to generalize to new, unseen data.",
    ]

    for i, doc in enumerate(ml_documents):
        retriever.add_document(f"ml_doc_{i}", doc)

    return retriever


def setup_milvus_retriever():
    """Set up Milvus retriever with specialized critic knowledge."""

    # Create Milvus MCP configuration (using official Milvus MCP server)
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory ../../mcp/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )

    # Create Milvus storage for critic context
    milvus_storage = MilvusStorage(mcp_config=milvus_config, collection_name="critic_knowledge")

    # Create in-memory retriever and populate with critic knowledge
    retriever = InMemoryRetriever()

    # Add specialized knowledge for critics
    critic_documents = [
        "Technical accuracy requires verifying mathematical concepts, algorithmic descriptions, and implementation details for correctness.",
        "Clarity and readability involve using clear language, logical structure, and appropriate technical terminology for the target audience.",
        "Completeness means covering all essential aspects of a topic without significant gaps or omissions in the explanation.",
        "Practical relevance focuses on real-world applications, use cases, and actionable insights rather than purely theoretical concepts.",
        "Ethical considerations in AI include bias, fairness, transparency, accountability, and potential societal impacts of technology.",
        "Best practices in machine learning include proper data preprocessing, model validation, performance metrics, and deployment considerations.",
    ]

    for i, doc in enumerate(critic_documents):
        retriever.add_document(f"critic_doc_{i}", doc)

    return retriever


def main():
    """Run the OpenAI N-Critics with Redis and Milvus example."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    logger.info("Creating OpenAI N-Critics with Redis and Milvus example")

    # Create OpenAI model
    model = OpenAIModel(model_name="gpt-4", temperature=0.8, max_tokens=600)

    # Set up retrievers
    redis_retriever = setup_redis_retriever()
    milvus_retriever = setup_milvus_retriever()

    # Create N-Critics ensemble with specialized focus areas
    critic = NCriticsCritic(
        model=model,
        num_critics=3,
        focus_areas=["technical_accuracy", "clarity_and_structure", "practical_applications"],
        name="ML Expertise Ensemble",
    )

    # Create the chain with Redis for model and Milvus for critic
    chain = Chain(
        model=model,
        prompt="Explain the key differences between supervised, unsupervised, and reinforcement learning, including when to use each approach and provide practical examples.",
        model_retrievers=[redis_retriever],  # Redis context for model
        critic_retrievers=[milvus_retriever],  # Milvus context for critics
        max_improvement_iterations=1,  # Exactly 1 retry
        apply_improvers_on_validation_failure=False,  # No validators
        always_apply_critics=True,
    )

    # Add critic (no validators as specified)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with N-Critics and dual retrieval...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("OPENAI N-CRITICS WITH REDIS AND MILVUS RETRIEVAL EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nIterations: {result.iteration}")
    print(f"Max Iterations: 1 (as specified)")
    print(f"Chain ID: {result.chain_id}")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        print(f"\nModel Context from Redis ({len(result.pre_generation_context)} documents):")
        for i, doc in enumerate(result.pre_generation_context[:3], 1):  # Show first 3
            print(f"  {i}. {doc.content[:100]}...")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nN-Critics Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:200]}...")

    print(f"\nValidation: BYPASSED (no validators used)")
    print("\n" + "=" * 80)
    logger.info("N-Critics with dual retrieval example completed successfully")


if __name__ == "__main__":
    main()
