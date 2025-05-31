#!/usr/bin/env python3
"""Self-RAG with Length Validator Example (PydanticAI).

This example demonstrates:
- PydanticAI Agent with Anthropic Claude model and in-memory retrieval for enhanced context
- Self-RAG critic for retrieval-augmented generation feedback
- Length validator to ensure appropriate response length
- File storage for thought persistence
- Default retry behavior

The chain will generate content about climate science with in-memory retrieval providing
scientific context and Self-RAG ensuring factual accuracy through retrieval.
Thoughts are stored in local files for persistence.
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_rag import SelfRAGCritic
from sifaka.models import create_model
from sifaka.retrievers import InMemoryRetriever
from sifaka.storage import FileStorage
from sifaka.utils.logging import get_logger
from sifaka.validators import LengthValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_climate_retriever():
    """Set up in-memory retriever with climate science documents."""

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


async def main():
    """Run the Self-RAG with Length Validator example using PydanticAI."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    logger.info("Creating PydanticAI Self-RAG with in-memory retrieval example")

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

    # Set up in-memory retriever with climate science context
    retriever = setup_climate_retriever()

    # Create file storage for thoughts
    file_storage = FileStorage(
        "./thoughts/self_rag_example_thoughts.json",
        overwrite=True,  # Overwrite existing file instead of appending
    )

    # Create Self-RAG critic using create_model for the critic model
    critic_model = create_model("anthropic:claude-3-5-haiku-latest")

    critic = SelfRAGCritic(
        model=critic_model,
        retriever=retriever,  # Self-RAG uses retrieval for fact-checking
        name="Climate Science Self-RAG Critic",
    )

    # Create length validator to ensure comprehensive but concise responses
    # Set bounds to likely fail initially but succeed after critic feedback
    length_validator = LengthValidator(
        min_length=1200,  # Minimum 1200 characters (higher to trigger initial failure)
        max_length=1600,  # Maximum 1600 characters (tighter window to require refinement)
    )

    # Create the PydanticAI chain with in-memory retrieval for the model
    chain = create_pydantic_chain(
        agent=agent,
        model_retrievers=[retriever],  # In-memory context for model
        validators=[length_validator],
        critics=[critic],
        max_improvement_iterations=3,  # Default retry behavior
        always_apply_critics=True,  # Set to True per user preferences
        analytics_storage=file_storage,  # Use file storage for thoughts
    )

    # Define the prompt
    prompt = "Explain the relationship between greenhouse gas emissions and global temperature rise, including the main sources of emissions and potential solutions for mitigation."

    # Run the chain
    logger.info("Running PydanticAI chain with Self-RAG critic and in-memory retrieval...")
    result = await chain.run(prompt)

    # Display results
    print("\n" + "=" * 80)
    print("PYDANTIC AI SELF-RAG WITH IN-MEMORY RETRIEVAL AND LENGTH VALIDATOR")
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
        print(
            f"\nModel Context from In-Memory Retriever ({len(result.pre_generation_context)} documents):"
        )
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
    logger.info("PydanticAI Self-RAG with in-memory retrieval example completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
