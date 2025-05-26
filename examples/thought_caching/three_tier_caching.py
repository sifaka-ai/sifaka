#!/usr/bin/env python3
"""Three-Tiered Thought Caching Example.

This example demonstrates:
- Memory → Redis → Milvus three-tier caching for thoughts
- Automatic tier management and fallback
- Performance optimization through layered storage
- Comprehensive thought persistence across all tiers

The chain will generate content about distributed systems and demonstrate
how thoughts are cached across memory, Redis, and Milvus storage layers.
"""

import os
import time
from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.base import MockModel
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.validators.base import LengthValidator
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.redis import RedisStorage
from sifaka.storage.milvus import MilvusStorage
from sifaka.storage.cached import CachedStorage
from sifaka.mcp import MCPServerConfig, MCPTransportType
from sifaka.utils.logging import get_logger

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_three_tier_storage():
    """Create three-tiered caching storage system."""

    # Layer 1: Memory storage (fastest, temporary)
    memory_storage = MemoryStorage()
    logger.info("Created memory storage layer (Layer 1)")

    # Layer 2: Redis storage (fast, persistent)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="cd mcp/mcp-redis && python -m main.py",
    )
    redis_storage = RedisStorage(mcp_config=redis_config, key_prefix="sifaka:thoughts")
    logger.info("Created Redis storage layer (Layer 2)")

    # Layer 3: Milvus storage (slower, vector search, long-term)
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="cd mcp/mcp-server-milvus && python -m mcp_server_milvus",
    )
    milvus_storage = MilvusStorage(mcp_config=milvus_config, collection_name="thought_cache")
    logger.info("Created Milvus storage layer (Layer 3)")

    # Create three-tier cached storage
    cached_storage = CachedStorage(
        memory_storage=memory_storage,
        redis_storage=redis_storage,
        milvus_storage=milvus_storage,
        cache_ttl=3600,  # 1 hour cache TTL
        enable_write_through=True,  # Write to all layers
        enable_read_through=True,  # Read from fastest available layer
    )

    logger.info("Created three-tier cached storage system")
    return cached_storage, memory_storage, redis_storage, milvus_storage


def demonstrate_tier_access(cached_storage, memory_storage, redis_storage, milvus_storage, thought):
    """Demonstrate how thoughts are accessed across different tiers."""

    print(f"\nThree-Tier Access Demonstration:")

    # Check memory tier
    start_time = time.time()
    memory_result = memory_storage.get_thought(thought.chain_id, thought.iteration)
    memory_time = (time.time() - start_time) * 1000
    print(f"  Memory Tier: {'✓ Found' if memory_result else '✗ Not found'} ({memory_time:.2f}ms)")

    # Check Redis tier
    try:
        start_time = time.time()
        redis_result = redis_storage.get_thought(thought.chain_id, thought.iteration)
        redis_time = (time.time() - start_time) * 1000
        print(f"  Redis Tier: {'✓ Found' if redis_result else '✗ Not found'} ({redis_time:.2f}ms)")
    except Exception as e:
        print(f"  Redis Tier: ✗ Error ({e})")

    # Check Milvus tier
    try:
        start_time = time.time()
        milvus_result = milvus_storage.get_thought(thought.chain_id, thought.iteration)
        milvus_time = (time.time() - start_time) * 1000
        print(
            f"  Milvus Tier: {'✓ Found' if milvus_result else '✗ Not found'} ({milvus_time:.2f}ms)"
        )
    except Exception as e:
        print(f"  Milvus Tier: ✗ Error ({e})")


def main():
    """Run the Three-Tiered Thought Caching example."""

    logger.info("Creating three-tiered thought caching example")

    # Create three-tier storage system
    cached_storage, memory_storage, redis_storage, milvus_storage = create_three_tier_storage()

    # Create mock model with distributed systems responses
    model = MockModel(
        name="Distributed Systems Model",
        responses=[
            "Distributed systems are collections of independent computers that appear to users as a single coherent system, working together to achieve common goals.",
            "Distributed systems are sophisticated collections of independent computers that appear to users as a single, coherent system, working together through network communication to achieve common computational goals while handling challenges like network failures, data consistency, and load distribution.",
            "Distributed systems represent sophisticated architectures of independent computers that appear to users as a single, coherent system, working together through network communication to achieve common computational goals while elegantly handling complex challenges like network failures, data consistency, load distribution, and fault tolerance through advanced algorithms and protocols.",
        ],
    )

    # Create Self-Refine critic for improvement
    critic = SelfRefineCritic(
        model=model, max_refinements=2, name="Distributed Systems Self-Refine Critic"
    )

    # Create length validator
    length_validator = LengthValidator(
        min_length=150, max_length=800, name="Technical Content Length Validator"
    )

    # Create the chain with three-tier caching
    chain = Chain(
        model=model,
        prompt="Explain what distributed systems are, their key characteristics, and the main challenges involved in building reliable distributed systems.",
        storage=cached_storage,  # Three-tier caching
        max_improvement_iterations=2,
        apply_improvers_on_validation_failure=True,
        always_apply_critics=True,
    )

    # Add validator and critic
    chain.validate_with(length_validator)
    chain.improve_with(critic)

    # Run the chain
    logger.info("Running chain with three-tier thought caching...")
    result = chain.run()

    # Display results
    print("\n" + "=" * 80)
    print("THREE-TIERED THOUGHT CACHING EXAMPLE")
    print("=" * 80)
    print(f"\nPrompt: {result.prompt}")
    print(f"\nFinal Text ({len(result.text)} characters):")
    print("-" * 50)
    print(result.text)

    print(f"\nProcessing Details:")
    print(f"  Iterations: {result.iteration}")
    print(f"  Chain ID: {result.chain_id}")
    print(f"  Storage: Three-tier caching")

    # Show validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        for validation_result in result.validation_results:
            status = "✓ PASSED" if validation_result.is_valid else "✗ FAILED"
            print(f"  {validation_result.validator_name}: {status}")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nSelf-Refine Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:250]}...")

    # Demonstrate tier access
    demonstrate_tier_access(cached_storage, memory_storage, redis_storage, milvus_storage, result)

    # Show caching architecture details
    print(f"\nThree-Tier Caching Architecture:")
    print(f"  ┌─ Layer 1: Memory Storage")
    print(f"  │  ├─ Speed: Fastest (~0.1ms)")
    print(f"  │  ├─ Persistence: Temporary (process lifetime)")
    print(f"  │  └─ Capacity: Limited by RAM")
    print(f"  │")
    print(f"  ├─ Layer 2: Redis Storage")
    print(f"  │  ├─ Speed: Fast (~1-10ms)")
    print(f"  │  ├─ Persistence: Configurable TTL (2 hours)")
    print(f"  │  └─ Capacity: High (disk-backed)")
    print(f"  │")
    print(f"  └─ Layer 3: Milvus Storage")
    print(f"     ├─ Speed: Moderate (~10-100ms)")
    print(f"     ├─ Persistence: Long-term")
    print(f"     └─ Features: Vector search, semantic similarity")

    # Show thought history across tiers
    print(f"\nThought History Across Tiers ({len(result.history)} iterations):")
    for i, historical_thought in enumerate(result.history, 1):
        print(f"  Iteration {i}:")
        print(f"    Text length: {len(historical_thought.text)} characters")
        print(f"    Cached in: Memory + Redis + Milvus")
        if i <= 2:  # Show first 2 iterations in detail
            print(f"    Preview: {historical_thought.text[:80]}...")

    # Performance and benefits analysis
    print(f"\nCaching Benefits:")
    print(f"  ✓ Fast access through memory layer")
    print(f"  ✓ Persistent storage through Redis")
    print(f"  ✓ Long-term archival through Milvus")
    print(f"  ✓ Automatic tier management")
    print(f"  ✓ Fault tolerance through redundancy")
    print(f"  ✓ Semantic search capabilities")

    print(f"\nUse Cases for Three-Tier Caching:")
    print(f"  - High-performance thought retrieval")
    print(f"  - Fault-tolerant thought persistence")
    print(f"  - Semantic similarity search")
    print(f"  - Long-term thought archival")
    print(f"  - Multi-level cache optimization")
    print(f"  - Distributed system resilience")

    print("\n" + "=" * 80)
    logger.info("Three-tiered thought caching example completed successfully")


if __name__ == "__main__":
    main()
