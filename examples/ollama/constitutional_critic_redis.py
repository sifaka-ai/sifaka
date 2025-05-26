#!/usr/bin/env python3
"""Ollama Constitutional Critic with Redis Retrieval Example.

This example demonstrates:
- Ollama local model with Redis retrieval for context
- Constitutional critic with Redis retrieval for principled evaluation
- Default retry behavior
- Local processing with external knowledge

The chain will generate content about digital privacy rights using Redis
for both model context and constitutional principles evaluation.
"""

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.models.ollama import OllamaModel
from sifaka.retrievers.simple import InMemoryRetriever
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_privacy_redis_retriever():
    """Set up retriever with digital privacy context documents.

    This example demonstrates Redis integration concept. In production,
    you would configure Redis MCP server and use RedisStorage.
    For now, we use in-memory storage with Redis-like structure.
    """

    # Create in-memory retriever (Redis fallback for this demo)
    retriever = InMemoryRetriever()

    # Add digital privacy context documents
    privacy_documents = [
        "Digital privacy is the right of individuals to control how their personal information is collected, used, and shared in digital environments.",
        "Data minimization principle states that organizations should only collect personal data that is necessary for specific, legitimate purposes.",
        "The General Data Protection Regulation (GDPR) gives individuals rights including access, rectification, erasure, and data portability.",
        "End-to-end encryption ensures that only communicating users can read messages, protecting against surveillance and data breaches.",
        "Privacy by design integrates privacy considerations into system development from the earliest stages rather than as an afterthought.",
        "Consent must be freely given, specific, informed, and unambiguous for lawful processing of personal data under privacy regulations.",
        "Data controllers are responsible for implementing appropriate technical and organizational measures to ensure data security.",
        "Anonymization and pseudonymization techniques help protect individual privacy while enabling data analysis and research.",
        "Cross-border data transfers require adequate protection levels or appropriate safeguards to maintain privacy rights.",
        "Privacy impact assessments help organizations identify and mitigate privacy risks before implementing new systems or processes.",
    ]

    # Store documents in retriever (simulating Redis storage structure)
    logger.info("Setting up privacy context documents...")
    for i, doc in enumerate(privacy_documents):
        doc_id = f"privacy_doc_{i}"
        retriever.add_document(doc_id, doc)

    logger.info(f"Loaded {len(privacy_documents)} privacy documents into retriever")

    # Return retriever and None for redis_storage (indicating fallback mode)
    return retriever, None


def setup_constitutional_principles():
    """Define constitutional principles for digital privacy evaluation."""

    principles = [
        "Respect individual autonomy and the right to privacy in digital spaces",
        "Ensure transparency in data collection, processing, and sharing practices",
        "Minimize data collection to what is necessary and proportionate for stated purposes",
        "Provide individuals with meaningful control over their personal information",
        "Implement strong security measures to protect personal data from unauthorized access",
        "Be honest about privacy risks and limitations of protection measures",
        "Avoid discriminatory practices in data processing and algorithmic decision-making",
        "Support democratic values and human rights in digital technology development",
        "Promote accountability and responsibility in data handling practices",
        "Balance privacy rights with legitimate interests of society and innovation",
    ]

    return principles


def main():
    """Run the Ollama Constitutional Critic with Redis example."""

    logger.info("Creating Ollama Constitutional Critic with Redis retrieval example")

    # Create Ollama model
    model = OllamaModel(
        model_name="mistral:latest",  # Using available model
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=700,
    )

    # Test if Ollama is available
    try:
        if not model.connection.health_check():
            raise Exception("Ollama server health check failed")
        logger.info("Ollama service is available")
    except Exception as e:
        logger.error(f"Ollama service not available: {e}")
        print("Error: Ollama service is not running. Please start Ollama and try again.")
        return

    # Set up retriever with privacy context (Redis demo mode)
    privacy_retriever, _ = setup_privacy_redis_retriever()

    # Get constitutional principles
    constitutional_principles = setup_constitutional_principles()

    # Create constitutional critic with Redis retrieval
    critic = ConstitutionalCritic(
        model=model,
        principles=constitutional_principles,
        retriever=privacy_retriever,  # Constitutional critic uses Redis for principled evaluation
        name="Digital Privacy Constitutional Critic",
    )

    # Create the chain with Redis for both model and critic
    chain = Chain(
        model=model,
        prompt="Write a comprehensive analysis of digital privacy rights in the modern internet age, covering the challenges individuals face, the role of technology companies, government regulations, and practical steps people can take to protect their privacy online.",
        model_retrievers=[privacy_retriever],  # Redis context for model
        critic_retrievers=[privacy_retriever],  # Same Redis context for critic
        max_improvement_iterations=3,  # Default retry behavior
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add constitutional critic (no validators specified)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with constitutional critic and Redis retrieval...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("OLLAMA CONSTITUTIONAL CRITIC WITH REDIS RETRIEVAL EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: Local Ollama")
    storage_status = "In-Memory (Redis demo mode)"
    print(f"  Storage: {storage_status}")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        context_source = "In-Memory (Redis demo mode)"
        print(
            f"\nModel Context from {context_source} ({len(result.pre_generation_context)} documents):"
        )
        for i, doc in enumerate(result.pre_generation_context[:3], 1):  # Show first 3
            print(f"  {i}. {doc.text[:100]}...")

    # Show constitutional critic feedback
    if result.critic_feedback:
        print(f"\nConstitutional Critic Evaluation:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Constitutional Assessment: {feedback.suggestions[:300]}...")

            # Show which principles were evaluated
            if hasattr(feedback, "principles_evaluated"):
                print(
                    f"     Principles Evaluated: {len(constitutional_principles)} constitutional principles"
                )

    print(f"\nConstitutional Principles Applied:")
    for i, principle in enumerate(constitutional_principles[:5], 1):  # Show first 5
        print(f"  {i}. {principle}")
    if len(constitutional_principles) > 5:
        print(f"  ... and {len(constitutional_principles) - 5} more principles")

    print(f"\nSystem Features:")
    print(f"  - Local Ollama processing")
    storage_feature = "In-Memory retrieval (Redis demo mode)"
    print(f"  - {storage_feature}")
    print(f"  - Constitutional principles evaluation")
    print(f"  - Privacy-focused content generation")

    print("\n" + "=" * 80)
    logger.info("Constitutional critic with Redis retrieval example completed successfully")


if __name__ == "__main__":
    main()
