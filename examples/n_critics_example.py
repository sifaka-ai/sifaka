#!/usr/bin/env python3
"""PydanticAI N-Critics with In-Memory Retrieval Example.

This example demonstrates:
- PydanticAI agent for generation with in-memory retrieval for context
- N-Critics ensemble using unified PydanticAI models with specialized retrieval
- No validators (validation bypassed)
- Exactly 1 retry iteration
- Unified model architecture for both generation and criticism
- File storage for thought persistence

The chain will generate content about machine learning with in-memory retrievers
providing context to the agent and specialized knowledge to the critics.
"""

import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.n_critics import NCriticsCritic
from sifaka.models.base import create_model
from sifaka.retrievers.memory import InMemoryRetriever
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_model_retriever():
    """Set up in-memory retriever with ML context documents for the model."""

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


def setup_critic_retriever():
    """Set up in-memory retriever with specialized critic knowledge."""

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
    """Run the PydanticAI N-Critics with In-Memory Retrieval example."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    logger.info("Creating PydanticAI N-Critics with In-Memory Retrieval example")

    # Create PydanticAI agent for generation
    agent = Agent(
        "openai:gpt-4",
        system_prompt="You are an expert machine learning educator who provides clear, comprehensive explanations with practical examples.",
    )

    # Set up in-memory retrievers
    model_retriever = setup_model_retriever()
    critic_retriever = setup_critic_retriever()

    # Set up file storage for thoughts
    storage = FileStorage(file_path="./thoughts/n_critics_thoughts.json", overwrite=True)
    logger.info("Using file storage for thoughts")

    # Create unified model for critics using PydanticAI
    critic_model = create_model("openai:gpt-4", temperature=0.8, max_tokens=600)

    # Create extremely demanding N-Critics ensemble with specialized focus areas
    critic = NCriticsCritic(
        model=critic_model,
        num_critics=3,
        improvement_threshold=9.9,  # Extremely demanding threshold - almost always trigger improvements
        critic_roles=[
            "Technical Accuracy Expert: Focus on mathematical concepts, algorithmic correctness, and factual precision",
            "Clarity and Structure Specialist: Focus on writing clarity, logical flow, and educational effectiveness",
            "Practical Applications Analyst: Focus on real-world examples, use cases, and actionable insights",
        ],
    )

    # Create the PydanticAI chain with in-memory retrievers
    chain = create_pydantic_chain(
        agent=agent,
        model_retrievers=[model_retriever],  # In-memory context for model
        critic_retrievers=[critic_retriever],  # In-memory context for critics
        critics=[critic],  # N-Critics ensemble
        max_improvement_iterations=1,  # Exactly 1 retry
        always_apply_critics=True,
        storage=storage,  # File storage for thoughts
    )

    # Run the chain with the prompt
    prompt = "Explain the key differences between supervised, unsupervised, and reinforcement learning, including when to use each approach and provide practical examples."
    logger.info("Running PydanticAI chain with N-Critics and in-memory retrieval...")
    result = chain.run_sync(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI N-CRITICS WITH IN-MEMORY RETRIEVAL EXAMPLE")
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
        print(
            f"\nModel Context from In-Memory Retriever ({len(result.pre_generation_context)} documents):"
        )
        for i, doc in enumerate(result.pre_generation_context[:3], 1):  # Show first 3
            # Handle both content and text attributes
            text = getattr(doc, "content", None) or getattr(doc, "text", str(doc))
            print(f"  {i}. {text[:100]}...")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nN-Critics Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:200]}...")

    print(f"\nValidation: BYPASSED (no validators used)")
    print(f"\nArchitecture: PydanticAI Agent + Unified Models + In-Memory Retrieval")
    print(f"\nStorage: File storage (./thoughts)")
    print("\n" + "=" * 80)
    logger.info("PydanticAI N-Critics with in-memory retrieval example completed successfully")


if __name__ == "__main__":
    main()
