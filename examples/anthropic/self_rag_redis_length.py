#!/usr/bin/env python3
"""Anthropic Self-RAG with Redis Retrieval and Length Validator Example.

This example demonstrates:
- Anthropic Claude model with Redis retrieval for enhanced context
- Self-RAG critic for retrieval-augmented generation feedback
- Length validator to ensure appropriate response length
- Default retry behavior

The chain will generate content about climate science with Redis providing
scientific context and Self-RAG ensuring factual accuracy through retrieval.
"""

import os

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.models.anthropic import AnthropicModel
from sifaka.retrievers.simple import InMemoryRetriever
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
        url="cd mcp/mcp-redis && python -m main.py",
    )

    # Create Redis storage for climate context
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:climate")

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

    return retriever


def main():
    """Run the Anthropic Self-RAG with Redis and Length Validator example."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating Anthropic Self-RAG with Redis retrieval example")

    # Create Anthropic model
    model = AnthropicModel(model_name="claude-3-sonnet-20240229", max_tokens=800, temperature=0.6)

    # Set up Redis retriever with climate science context
    redis_retriever = setup_climate_redis_retriever()

    # Create Self-RAG critic for fact-checking and retrieval-augmented feedback
    critic = SelfRAGCritic(
        model=model,
        retriever=redis_retriever,  # Self-RAG uses retrieval for fact-checking
        name="Climate Science Self-RAG Critic",
    )

    # Create length validator to ensure comprehensive but concise responses
    length_validator = LengthValidator(
        min_length=300,  # Minimum 300 characters for comprehensive coverage
        max_length=1200,  # Maximum 1200 characters to stay focused
    )

    # Create the chain with Redis retrieval for the model
    chain = Chain(
        model=model,
        prompt="Explain the relationship between greenhouse gas emissions and global temperature rise, including the main sources of emissions and potential solutions for mitigation.",
        model_retrievers=[redis_retriever],  # Redis context for model
        max_improvement_iterations=3,  # Default retry behavior
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add validator and critic
    chain.validate_with(length_validator)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with Self-RAG critic and Redis retrieval...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("ANTHROPIC SELF-RAG WITH REDIS RETRIEVAL AND LENGTH VALIDATOR")
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
    logger.info("Self-RAG with Redis retrieval example completed successfully")


if __name__ == "__main__":
    main()
