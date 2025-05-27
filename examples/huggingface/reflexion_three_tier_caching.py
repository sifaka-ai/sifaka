#!/usr/bin/env python3
"""HuggingFace Reflexion with Three-Tiered Caching Example.

This example demonstrates:
- HuggingFace model with retrievers and three-tiered caching
- Enhanced Reflexion critic with trial-based learning and task feedback
- Memory → Redis → Milvus caching architecture
- Episodic memory for learning from past reflections
- Default retry behavior

The chain will generate content about sustainable energy with full three-tier
caching providing comprehensive context management and storage. The Reflexion
critic tracks trial numbers and can incorporate external task feedback
for more authentic trial-and-error learning.
"""

import os

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.models.huggingface import HuggingFaceModel
from sifaka.retrievers.simple import InMemoryRetriever
from sifaka.storage.cached import CachedStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.milvus import MilvusStorage
from sifaka.storage.redis import RedisStorage
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def setup_three_tier_storage():
    """Set up three-tiered caching: Memory → Redis → Milvus."""

    # Layer 1: Memory storage (fastest)
    memory_storage = MemoryStorage()

    # Layer 2: Redis storage (medium speed, persistent)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-redis src/main.py",
    )
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:energy")

    # Layer 3: Milvus storage (slowest, vector search)
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="uv run --directory mcp/mcp-server-milvus src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530",
    )
    milvus_storage = MilvusStorage(mcp_config=milvus_config, collection_name="sustainable_energy")

    # Create three-tier cached storage: Memory → Redis → Milvus
    cached_storage = CachedStorage(
        cache=memory_storage,  # L1: Memory (fastest)
        persistence=CachedStorage(
            cache=redis_storage,  # L2: Redis (persistent cache)
            persistence=milvus_storage,  # L3: Milvus (vector search)
        ),
    )

    return cached_storage


def setup_energy_retrievers():
    """Set up retrievers with sustainable energy context."""

    # Create in-memory retriever for model context
    model_retriever = InMemoryRetriever()

    # Add sustainable energy documents for model context
    energy_documents = [
        "Solar photovoltaic technology converts sunlight directly into electricity using semiconductor materials that exhibit the photovoltaic effect.",
        "Wind turbines harness kinetic energy from moving air masses to generate electricity through aerodynamic rotor blades and electrical generators.",
        "Hydroelectric power generates electricity by harnessing the gravitational force of flowing or falling water through turbines and generators.",
        "Geothermal energy utilizes heat from the Earth's core to generate electricity or provide direct heating for buildings and industrial processes.",
        "Energy storage systems like lithium-ion batteries and pumped hydro storage help balance supply and demand in renewable energy grids.",
        "Smart grids use digital technology to optimize electricity distribution, integrate renewable sources, and improve energy efficiency.",
        "Carbon capture and storage technologies aim to reduce greenhouse gas emissions from fossil fuel power plants and industrial facilities.",
        "Energy efficiency measures in buildings include improved insulation, LED lighting, smart thermostats, and high-efficiency appliances.",
    ]

    for i, doc in enumerate(energy_documents):
        model_retriever.add_document(f"energy_doc_{i}", doc)

    # Create in-memory retriever for critic context
    critic_retriever = InMemoryRetriever()

    # Add reflection and improvement guidance for critics
    reflection_documents = [
        "Effective reflection involves analyzing the accuracy, completeness, and clarity of generated content against established knowledge.",
        "Technical accuracy in energy discussions requires verification of scientific principles, efficiency ratings, and implementation details.",
        "Comprehensive coverage should address environmental impacts, economic considerations, and social implications of energy technologies.",
        "Clear explanations use appropriate technical terminology while remaining accessible to the intended audience level.",
        "Balanced perspectives acknowledge both benefits and limitations of different energy technologies and approaches.",
        "Current relevance ensures information reflects recent technological developments and policy changes in the energy sector.",
    ]

    for i, doc in enumerate(reflection_documents):
        critic_retriever.add_document(f"reflection_doc_{i}", doc)

    return model_retriever, critic_retriever


def main():
    """Run the HuggingFace Reflexion with Three-Tiered Caching example."""

    # Ensure API key is available
    if not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError("HUGGINGFACE_API_KEY environment variable is required")

    logger.info("Creating HuggingFace Reflexion with three-tiered caching example")

    # Create HuggingFace model using Microsoft Phi-4
    model = HuggingFaceModel(
        model_name="microsoft/phi-4",  # Microsoft Phi-4 model
        api_token=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0.7,
        max_tokens=900,
        use_inference_api=True,
    )

    # Test if HuggingFace API is available
    try:
        _ = model.generate("Test", max_tokens=5)
        logger.info("HuggingFace API is available")
    except Exception as e:
        logger.error(f"HuggingFace API not available: {e}")
        print("Error: HuggingFace API is not accessible. Please check your API key and try again.")
        return

    # Set up three-tiered caching storage
    cached_storage = setup_three_tier_storage()

    # Set up retrievers with energy context
    model_retriever, critic_retriever = setup_energy_retrievers()

    # Create Reflexion critic for iterative improvement
    critic = ReflexionCritic(
        model=model,
        max_memory_size=10,  # Keep memory of past reflections
    )

    # Create the chain with three-tiered caching
    chain = Chain(
        model=model,
        prompt="Provide a comprehensive analysis of sustainable energy technologies, comparing solar, wind, hydroelectric, and geothermal power in terms of efficiency, environmental impact, cost-effectiveness, and scalability for meeting global energy needs.",
        model_retrievers=[model_retriever],  # Energy context for model
        critic_retrievers=[critic_retriever],  # Reflection guidance for critic
        storage=cached_storage,  # Three-tiered caching
        max_improvement_iterations=2,  # Allow for improvement iterations
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add Reflexion critic (no validators specified)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with Reflexion critic and three-tiered caching...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("HUGGINGFACE REFLEXION WITH THREE-TIERED CACHING EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Model: HuggingFace Phi-4 (remote)")
    print(f"  Storage: Three-tiered caching")

    # Show caching information
    print(f"\nThree-Tiered Caching Architecture:")
    print(f"  Layer 1: Memory Storage (fastest)")
    print(f"  Layer 2: Redis Storage (persistent)")
    print(f"  Layer 3: Milvus Storage (vector search)")
    print(f"  Cache TTL: 1 hour")

    # Show retrieval context
    if hasattr(result, "pre_generation_context") and result.pre_generation_context:
        print(f"\nModel Context ({len(result.pre_generation_context)} documents):")
        for i, doc in enumerate(result.pre_generation_context[:3], 1):  # Show first 3
            print(f"  {i}. {doc.text[:120]}...")

    # Show Reflexion critic feedback
    if result.critic_feedback:
        print(f"\nReflexion Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Reflection Analysis: {feedback.suggestions[:300]}...")

            # Show Reflexion specific metrics if available
            if feedback.metadata:
                if "trial_number" in feedback.metadata:
                    print(f"     Trial Number: {feedback.metadata['trial_number']}")
                if "memory_size" in feedback.metadata:
                    print(f"     Memory Size: {feedback.metadata['memory_size']}")
                if "task_feedback" in feedback.metadata and feedback.metadata["task_feedback"]:
                    task_fb = feedback.metadata["task_feedback"]
                    print(f"     Task Success: {task_fb.get('success', 'N/A')}")
                    if task_fb.get("score"):
                        print(f"     Task Score: {task_fb['score']:.2f}")

            # Legacy compatibility
            if hasattr(feedback, "reflection_depth"):
                print(f"     Reflection Depth: {feedback.reflection_depth}")
            if hasattr(feedback, "improvement_score"):
                print(f"     Improvement Score: {feedback.improvement_score}")

    # Analyze energy technology coverage
    print(f"\nEnergy Technology Coverage Analysis:")
    energy_techs = ["solar", "wind", "hydroelectric", "geothermal"]
    covered_techs = [tech for tech in energy_techs if tech.lower() in result.text.lower()]
    print(f"  Technologies covered: {len(covered_techs)}/{len(energy_techs)}")
    for tech in covered_techs:
        print(f"    ✓ {tech.capitalize()}")

    print(f"\nSystem Features:")
    print(f"  - HuggingFace Inference API")
    print(f"  - Microsoft Phi-4 model")
    print(f"  - Reflexion-based improvement")
    print(f"  - Comprehensive energy analysis")
    print(f"  - Multi-layer storage optimization")

    print("\n" + "=" * 80)
    logger.info("Reflexion with three-tiered caching example completed successfully")


if __name__ == "__main__":
    main()
