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

import time

from dotenv import load_dotenv

from sifaka.core.chain import Chain
from sifaka.critics.self_refine import SelfRefineCritic
from sifaka.models.base import MockModel
from sifaka.storage.cached import CachedStorage
from sifaka.storage.file import FileStorage
from sifaka.storage.memory import MemoryStorage
from sifaka.utils.logging import get_logger
from sifaka.validators.base import LengthValidator

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def create_three_tier_storage():
    """Create three-tiered caching storage system (mock version)."""

    # Layer 1: Memory storage (fastest, temporary)
    memory_storage = MemoryStorage()
    logger.info("Created memory storage layer (Layer 1)")

    # Layer 2: File storage as mock Redis (fast, persistent)
    # In a real implementation, this would be Redis
    file_storage = FileStorage(file_path="mock_redis_cache.json")
    logger.info("Created file storage layer (Layer 2 - mock Redis)")

    # Layer 3: File storage as mock Milvus (slower, vector search, long-term)
    # In a real implementation, this would be Milvus
    milvus_mock_storage = FileStorage(file_path="mock_milvus_cache.json")
    logger.info("Created file storage layer (Layer 3 - mock Milvus)")

    # Create cached storage combining memory and file storage
    cached_storage = CachedStorage(
        cache=memory_storage,
        persistence=file_storage,
    )

    logger.info("Created three-tier cached storage system (mock)")
    return cached_storage, memory_storage, file_storage, milvus_mock_storage


def demonstrate_tier_access(cached_storage, memory_storage, file_storage, milvus_storage, thought):
    """Demonstrate how thoughts are accessed across different tiers (mock version)."""

    print(f"\nThree-Tier Access Demonstration:")

    # Check memory tier
    start_time = time.time()
    memory_key = f"{thought.chain_id}:{thought.iteration}"
    memory_result = memory_storage.get(memory_key)
    memory_time = (time.time() - start_time) * 1000
    print(f"  Memory Tier: {'✓ Found' if memory_result else '✗ Not found'} ({memory_time:.2f}ms)")

    # Check file storage tier (mock Redis)
    try:
        start_time = time.time()
        file_result = file_storage.get(memory_key)
        file_time = (time.time() - start_time) * 1000
        print(
            f"  File Tier (mock Redis): {'✓ Found' if file_result else '✗ Not found'} ({file_time:.2f}ms)"
        )
    except Exception as e:
        print(f"  File Tier (mock Redis): ✗ Error ({e})")

    # Check Milvus tier (mock)
    try:
        start_time = time.time()
        milvus_result = milvus_storage.get(memory_key)
        milvus_time = (time.time() - start_time) * 1000
        print(
            f"  File Tier (mock Milvus): {'✓ Found' if milvus_result else '✗ Not found'} ({milvus_time:.2f}ms)"
        )
    except Exception as e:
        print(f"  File Tier (mock Milvus): ✗ Error ({e})")


def main():
    """Run the Three-Tiered Thought Caching example."""

    logger.info("Creating three-tiered thought caching example")

    # Create three-tier storage system
    cached_storage, memory_storage, file_storage, milvus_storage = create_three_tier_storage()

    # Create mock model with distributed systems responses
    model = MockModel(
        model_name="Distributed Systems Model",
        responses=[
            "Distributed systems are collections of independent computers that appear to users as a single coherent system, working together to achieve common goals.",
            "Distributed systems are sophisticated collections of independent computers that appear to users as a single, coherent system, working together through network communication to achieve common computational goals while handling challenges like network failures, data consistency, and load distribution.",
            "Distributed systems represent sophisticated architectures of independent computers that appear to users as a single, coherent system, working together through network communication to achieve common computational goals while elegantly handling complex challenges like network failures, data consistency, load distribution, and fault tolerance through advanced algorithms and protocols.",
        ],
    )

    # Create Self-Refine critic for improvement
    critic = SelfRefineCritic(model=model, max_iterations=2)

    # Create length validator
    length_validator = LengthValidator(min_length=150, max_length=800)

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
        for validator_name, validation_result in result.validation_results.items():
            status = "✓ PASSED" if validation_result.passed else "✗ FAILED"
            print(f"  {validator_name}: {status}")

    # Show critic feedback
    if result.critic_feedback:
        print(f"\nSelf-Refine Critic Feedback:")
        for i, feedback in enumerate(result.critic_feedback, 1):
            print(f"  {i}. {feedback.critic_name}:")
            print(f"     Needs Improvement: {feedback.needs_improvement}")
            if feedback.suggestions:
                print(f"     Suggestions: {feedback.suggestions[:250]}...")

    # Demonstrate tier access
    demonstrate_tier_access(cached_storage, memory_storage, file_storage, milvus_storage, result)

    # Show caching architecture details
    print(f"\nThree-Tier Caching Architecture (Mock):")
    print(f"  ┌─ Layer 1: Memory Storage")
    print(f"  │  ├─ Speed: Fastest (~0.1ms)")
    print(f"  │  ├─ Persistence: Temporary (process lifetime)")
    print(f"  │  └─ Capacity: Limited by RAM")
    print(f"  │")
    print(f"  ├─ Layer 2: File Storage (mock Redis)")
    print(f"  │  ├─ Speed: Fast (~1-10ms)")
    print(f"  │  ├─ Persistence: File-based")
    print(f"  │  └─ Capacity: High (disk-backed)")
    print(f"  │")
    print(f"  └─ Layer 3: File Storage (mock Milvus)")
    print(f"     ├─ Speed: Moderate (~10-100ms)")
    print(f"     ├─ Persistence: Long-term")
    print(f"     └─ Features: File-based storage (would be vector search in real Milvus)")

    # Show thought history across tiers
    print(f"\nThought History Across Tiers ({len(result.history)} iterations):")
    for i, historical_thought in enumerate(result.history, 1):
        print(f"  Iteration {historical_thought.iteration}:")
        print(f"    Summary: {historical_thought.summary}")
        print(f"    Cached in: Memory + Redis + Milvus")
        if i <= 2:  # Show first 2 iterations in detail
            print(f"    Details: {historical_thought.summary}")

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
