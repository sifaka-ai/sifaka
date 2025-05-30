#!/usr/bin/env python3
"""Constitutional Critic Example (PydanticAI).

This example demonstrates:
- PydanticAI agent with Gemini model and in-memory retrieval for context
- Constitutional critic with in-memory retrieval for principled evaluation
- Modern agent-based workflow with hybrid Chain-Agent architecture
- Cloud processing with external knowledge

The PydanticAI chain will generate content about digital privacy rights using
in-memory retrieval for both model context and constitutional principles evaluation.
"""

import logging
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.retrievers.memory import InMemoryRetriever
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)

# Enable debug logging to see what's happening with the critic and chain
logging.getLogger("sifaka.critics.constitutional").setLevel(logging.DEBUG)
logging.getLogger("sifaka.core.chain.executor").setLevel(logging.DEBUG)
logging.getLogger("sifaka.agents.chain").setLevel(logging.DEBUG)


def setup_privacy_retriever():
    """Set up in-memory retriever with digital privacy context documents."""

    # Create in-memory retriever and populate with privacy context
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

    # Store documents in retriever
    logger.info("Setting up privacy context documents in memory...")
    for i, doc in enumerate(privacy_documents):
        doc_id = f"privacy_doc_{i}"
        retriever.add_document(doc_id, doc)

    logger.info(f"Loaded {len(privacy_documents)} privacy documents into in-memory retriever")

    # Create simple file storage for thoughts
    storage = FileStorage(
        "./thoughts/constitutional_critic_thoughts.json",
        overwrite=False,  # Append to preserve all iterations with critic feedback
    )

    return retriever, storage


def setup_constitutional_principles():
    """Define constitutional principles for digital privacy evaluation (trimmed to 5 core principles)."""

    principles = [
        "Respect individual autonomy and the right to privacy in digital spaces",
        "Ensure transparency in data collection, processing, and sharing practices",
        "Minimize data collection to what is necessary and proportionate for stated purposes",
    ]

    return principles


async def main():
    """Run the Constitutional Critic example using PydanticAI (async version)."""

    # Ensure API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    logger.info(
        "Creating PydanticAI Constitutional Critic with Gemini and in-memory retrieval example"
    )

    # Use Gemini model directly for the critic
    logger.info("Using Gemini model for constitutional critic")

    # Set up in-memory retriever with privacy context and file storage
    privacy_retriever, storage = setup_privacy_retriever()

    # Get constitutional principles
    constitutional_principles = setup_constitutional_principles()

    # Create constitutional critic with in-memory retrieval using Gemini
    critic = ConstitutionalCritic(
        model="google-gla:gemini-1.5-flash",
        principles=constitutional_principles,
        retriever=privacy_retriever,  # Constitutional critic uses in-memory retrieval for principled evaluation
        name="Digital Privacy Constitutional Critic",
    )

    # Create PydanticAI agent with Gemini model
    agent = Agent(
        model="google-gla:gemini-1.5-flash",
        system_prompt=(
            "You are a digital privacy expert and policy analyst. Provide comprehensive, "
            "well-researched analysis of digital privacy rights, data protection regulations, "
            "and practical privacy solutions. Use evidence-based reasoning and consider "
            "multiple perspectives including individual rights, technological capabilities, "
            "and regulatory frameworks. Be thorough and educational in your responses."
        ),
    )

    # Define the prompt
    prompt = "Write a comprehensive analysis of digital privacy rights in the modern internet age, covering the challenges individuals face, the role of technology companies, government regulations, and practical steps people can take to protect their privacy online."

    # Create PydanticAI chain with constitutional critic and in-memory retrieval
    chain = create_pydantic_chain(
        agent=agent,
        critics=[critic],  # Constitutional critic with in-memory retrieval
        model_retrievers=[privacy_retriever],  # In-memory context for pre-generation
        critic_retrievers=[privacy_retriever],  # Same in-memory context for critic evaluation
        max_improvement_iterations=3,  # Default retry behavior
        always_apply_critics=True,
        storage=storage,  # Use file storage for thoughts
    )

    print(f"DEBUG: Created PydanticAI chain with {len([critic])} critics")

    # Run the chain asynchronously
    logger.info("Running PydanticAI chain with constitutional critic and in-memory retrieval...")
    result = await chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI CONSTITUTIONAL CRITIC EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: Gemini Flash (via PydanticAI)")
    storage_status = "File Storage"
    print(f"  Storage: {storage_status}")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        context_source = "In-Memory"
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

    print(f"\nConstitutional Principles Applied (5 core principles):")
    for i, principle in enumerate(constitutional_principles, 1):
        print(f"  {i}. {principle}")

    print(f"\nSystem Features:")
    print(f"  - Gemini Flash processing via PydanticAI")
    storage_feature = "In-memory retrieval with file storage"
    print(f"  - {storage_feature}")
    print(f"  - Constitutional principles evaluation (5 core principles)")
    print(f"  - Privacy-focused content generation")
    print(f"  - Modern async agent-based workflow")

    print("\n" + "=" * 80)
    logger.info(
        "PydanticAI Constitutional critic with in-memory retrieval example completed successfully"
    )

    return result


if __name__ == "__main__":
    import asyncio

    async def run_example():
        """Wrapper to run the example and handle the result properly."""
        result = await main()
        return result

    # Run the example
    result = asyncio.run(run_example())
    # Result is available for further processing if needed
