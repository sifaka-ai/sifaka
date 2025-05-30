#!/usr/bin/env python3
"""Anthropic Self-RAG with Redis + File Dual Storage and Length Validator Example (PydanticAI).

This example demonstrates:
- PydanticAI Agent with Anthropic Claude model and Redis retrieval for enhanced context
- Self-RAG critic for retrieval-augmented generation feedback
- Length validator to ensure appropriate response length
- Dual storage: Redis (primary) + File storage (backup/fallback)
- Default retry behavior

The chain will generate content about climate science with Redis providing
scientific context and Self-RAG ensuring factual accuracy through retrieval.
Thoughts are stored in both Redis and local files for redundancy.
"""

import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.models import create_model
from sifaka.retrievers.simple import InMemoryRetriever
from sifaka.storage import FileStorage
from sifaka.storage.cached import CachedStorage
from sifaka.storage.redis import RedisStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_climate_redis_retriever():
    """Set up Redis retriever with climate science documents."""

    # Create Redis MCP configuration (using local server)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-redis src/main.py",
    )

    # Create Redis storage for thoughts with timestamp-based key pattern
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix=f"{timestamp}_climate")

    # Create file storage as backup/secondary storage
    file_storage = FileStorage(
        f"./thoughts/self_rag_redis_length_thoughts.json",
        overwrite=True,  # Overwrite existing file instead of appending
    )

    # Create cached storage with file as cache and Redis as persistence
    dual_storage = CachedStorage(
        cache=file_storage,  # Fast local file cache
        persistence=redis_storage,  # Redis for persistence and sharing
    )

    # Create in-memory retriever and populate with climate science context
    retriever = InMemoryRetriever()

    # Add climate science documents
    climate_documents = [
        "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, keeping the planet warm enough to support life.",
        "Carbon dioxide (CO2) concentrations have increased by over 40% since pre-industrial times, primarily due to fossil fuel combustion and deforestation.",
        "Global average temperatures have risen by approximately 1.1°C (2°F) since the late 19th century, with most warming occurring in the past 40 years.",
        "Climate feedback loops can amplify warming, such as melting Arctic ice reducing surface reflectivity and permafrost thaw releasing stored carbon.",
        "Renewable energy sources like solar, wind, and hydroelectric power produce electricity with minimal greenhouse gas emissions compared to fossil fuels.",
        "Climate adaptation strategies include building sea walls, developing drought-resistant crops, and improving urban heat island management.",
        "The Paris Agreement aims to limit global warming to well below 2°C above pre-industrial levels, with efforts to limit it to 1.5°C.",
        "Ocean acidification occurs when seawater absorbs CO2 from the atmosphere, forming carbonic acid and lowering ocean pH levels.",
        "Extreme weather events like hurricanes, droughts, and heatwaves are becoming more frequent and intense due to climate change.",
        "Carbon capture and storage technologies aim to remove CO2 from the atmosphere or prevent its release from industrial sources.",
    ]

    for i, doc in enumerate(climate_documents):
        retriever.add_document(f"climate_doc_{i}", doc)

    return retriever, dual_storage


def main():
    """Run the Anthropic Self-RAG with Redis and Length Validator example using PydanticAI."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating PydanticAI Anthropic Self-RAG with Redis retrieval example")

    # Create PydanticAI agent with Anthropic Claude model
    agent = Agent(
        "anthropic:claude-3-5-haiku-latest",
        system_prompt=(
            "You are a climate science expert assistant. Provide accurate, well-researched "
            "information about climate change, greenhouse gases, and environmental solutions. "
            "Use scientific evidence and cite relevant data when available. Be comprehensive "
            "but concise in your explanations."
        ),
    )

    # Set up Redis retriever with climate science context and dual storage
    redis_retriever, dual_storage = setup_climate_redis_retriever()

    # Create Self-RAG critic for fact-checking and retrieval-augmented feedback
    # Use Haiku model for the critic as per preferences
    critic_model = create_model("anthropic:claude-3-5-haiku-latest")

    critic = SelfRAGCritic(
        model=critic_model,
        retriever=redis_retriever,  # Self-RAG uses retrieval for fact-checking
        name="Climate Science Self-RAG Critic",
    )

    # Create length validator to ensure comprehensive but concise responses
    # Set bounds to likely fail initially but succeed after critic feedback
    length_validator = LengthValidator(
        min_length=1200,  # Minimum 1200 characters (higher to trigger initial failure)
        max_length=1600,  # Maximum 1600 characters (tighter window to require refinement)
    )

    # Create the PydanticAI chain with Redis retrieval for the model
    chain = create_pydantic_chain(
        agent=agent,
        model_retrievers=[redis_retriever],  # Redis context for model
        validators=[length_validator],
        critics=[critic],
        max_improvement_iterations=3,  # Default retry behavior
        always_apply_critics=False,  # Set to False per user preferences
        storage=dual_storage,  # Use Redis + File dual storage for thoughts
    )

    # Define the prompt
    prompt = "Explain the relationship between greenhouse gas emissions and global temperature rise, including the main sources of emissions and potential solutions for mitigation."

    # Run the chain
    logger.info("Running PydanticAI chain with Self-RAG critic and Redis retrieval...")
    result = chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI ANTHROPIC SELF-RAG WITH REDIS + FILE DUAL STORAGE AND LENGTH VALIDATOR")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nIterations: {result.iteration}")
    print(f"Chain ID: {result.chain_id}")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for i, (validator_name, validation_result) in enumerate(
            result.validation_results.items(), 1
        ):
            print(
                f"  {i}. {validator_name}: {'✓ PASSED' if validation_result.passed else '✗ FAILED'}"
            )
            if validation_result.passed:
                print(
                    f"     Text length: {len(result.text)} characters (within {length_validator.min_length}-{length_validator.max_length} range)"
                )
            else:
                print(f"     Error: {validation_result.message}")
                if validation_result.issues:
                    print(f"     Issues: {', '.join(validation_result.issues)}")
                if validation_result.suggestions:
                    print(f"     Suggestions: {', '.join(validation_result.suggestions)}")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        print(f"\nModel Context from Redis ({len(result.pre_generation_context)} documents):")
        for i, doc in enumerate(result.pre_generation_context[:3], 1):  # Show first 3
            print(f"  {i}. {doc.text[:120]}...")

    # Show Self-RAG critic feedback
    if result.critic_feedback:
        print(f"\nSelf-RAG Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:250]}...")
            if hasattr(feedback, "retrieval_score"):
                print(f"     Retrieval Relevance Score: {feedback.retrieval_score}")

    print("\n" + "=" * 80)
    logger.info("PydanticAI Self-RAG with Redis retrieval example completed successfully")


if __name__ == "__main__":
    main()
