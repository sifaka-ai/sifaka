#!/usr/bin/env python3
"""Ollama Constitutional Critic with Redis Retrieval Example (PydanticAI).

This example demonstrates:
- PydanticAI agent with Ollama local model and Redis retrieval for context
- Constitutional critic with Redis retrieval for principled evaluation
- Modern agent-based workflow with hybrid Chain-Agent architecture
- Local processing with external knowledge

The PydanticAI chain will generate content about digital privacy rights using Redis
for both model context and constitutional principles evaluation.
"""

import logging

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from sifaka.agents import create_pydantic_chain
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.models.ollama import OllamaModel
from sifaka.retrievers.simple import InMemoryRetriever
from sifaka.storage import FileStorage
from sifaka.storage.cached import CachedStorage
from sifaka.storage.redis import RedisStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)

# Enable debug logging to see what's happening with the critic
logging.getLogger("sifaka.critics.constitutional").setLevel(logging.DEBUG)
logging.getLogger("sifaka.core.chain.executor").setLevel(logging.DEBUG)


def setup_privacy_redis_retriever():
    """Set up Redis retriever with digital privacy context documents."""

    # Create Redis MCP configuration (using local server)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-redis src/main.py",
    )

    # Create Redis storage for thoughts with timestamp-based key pattern
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix=f"{timestamp}_privacy")

    # Create file storage as backup/secondary storage
    file_storage = FileStorage(
        f"./thoughts/constitutional_critic_redis_thoughts.json",
        overwrite=False,  # Append to preserve all iterations with critic feedback
    )

    # Create cached storage with file as cache and Redis as persistence
    dual_storage = CachedStorage(
        cache=file_storage,  # Fast local file cache
        persistence=redis_storage,  # Redis for persistence and sharing
    )

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
    logger.info("Setting up privacy context documents in Redis...")
    for i, doc in enumerate(privacy_documents):
        doc_id = f"privacy_doc_{i}"
        retriever.add_document(doc_id, doc)

    logger.info(f"Loaded {len(privacy_documents)} privacy documents into Redis retriever")

    return retriever, dual_storage


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
    """Run the Ollama Constitutional Critic with Redis example using PydanticAI."""

    logger.info("Creating PydanticAI Ollama Constitutional Critic with Redis retrieval example")

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

    # Set up Redis retriever with privacy context and dual storage
    privacy_retriever, dual_storage = setup_privacy_redis_retriever()

    # Get constitutional principles
    constitutional_principles = setup_constitutional_principles()

    # Create constitutional critic with Redis retrieval
    critic = ConstitutionalCritic(
        model=model,
        principles=constitutional_principles,
        retriever=privacy_retriever,  # Constitutional critic uses Redis for principled evaluation
        name="Digital Privacy Constitutional Critic",
    )

    # Create PydanticAI agent with Ollama model (using OpenAI-compatible API)
    ollama_model = OpenAIModel(
        model_name="mistral",  # Model name in Ollama
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1"
        ),  # Ollama OpenAI-compatible endpoint
    )

    agent = Agent(
        model=ollama_model,
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

    # Create PydanticAI chain with constitutional critic and Redis retrieval
    chain = create_pydantic_chain(
        agent=agent,
        critics=[critic],  # Constitutional critic with Redis retrieval
        model_retrievers=[privacy_retriever],  # Redis context for pre-generation
        critic_retrievers=[privacy_retriever],  # Same Redis context for critic evaluation
        max_improvement_iterations=3,  # Default retry behavior
        always_apply_critics=True,
        storage=dual_storage,  # Use Redis + File dual storage for thoughts
    )

    print(f"DEBUG: Created PydanticAI chain with {len([critic])} critics")

    # Run the chain
    logger.info("Running PydanticAI chain with constitutional critic and Redis retrieval...")
    result = chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI OLLAMA CONSTITUTIONAL CRITIC WITH REDIS RETRIEVAL EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: Local Ollama (via PydanticAI)")
    storage_status = "Redis + File Dual Storage"
    print(f"  Storage: {storage_status}")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        context_source = "Redis"
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
    print(f"  - Local Ollama processing via PydanticAI")
    storage_feature = "Redis retrieval with dual storage"
    print(f"  - {storage_feature}")
    print(f"  - Constitutional principles evaluation")
    print(f"  - Privacy-focused content generation")
    print(f"  - Modern agent-based workflow")

    print("\n" + "=" * 80)
    logger.info(
        "PydanticAI Constitutional critic with Redis retrieval example completed successfully"
    )


if __name__ == "__main__":
    main()
