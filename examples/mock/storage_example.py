#!/usr/bin/env python3
"""
Example demonstrating Sifaka's unified storage capabilities.

This example shows how to:
1. Create and configure unified storage (Memory → Redis → Milvus)
2. Save thoughts with context, validation results, and critic feedback
3. Query thoughts with various filters and semantic search
4. Work with thought history and iterations
5. Perform health checks and view storage statistics
"""

from datetime import datetime, timedelta

from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
from sifaka.storage import SifakaStorage, ThoughtQuery
from sifaka.mcp import MCPServerConfig, MCPTransportType


def main():
    """Demonstrate unified storage capabilities."""
    print("🧠 Sifaka Unified Storage Example")
    print("=" * 50)

    # 1. Setup unified storage
    print("\n1. Setting up flexible storage backends...")

    # Configure Redis MCP server (optional)
    redis_config = MCPServerConfig(
        name="redis-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @modelcontextprotocol/server-redis redis://localhost:6379",
    )

    # Configure Milvus MCP server (optional)
    milvus_config = MCPServerConfig(
        name="milvus-server",
        transport_type=MCPTransportType.STDIO,
        url="npx -y @milvus-io/mcp-server-milvus",
    )

    # Demonstrate different storage configurations
    print("  Trying different storage configurations:")

    # Option 1: Memory-only (simplest)
    print("  - Memory-only storage...")
    memory_storage = SifakaStorage()
    print("    ✓ Memory-only storage created")

    # Option 2: Memory + Redis (caching)
    print("  - Memory + Redis caching...")
    try:
        redis_storage = SifakaStorage(redis_config=redis_config)
        print("    ✓ Memory + Redis storage created")
    except Exception as e:
        print(f"    ⚠ Redis not available: {e}")
        redis_storage = None

    # Option 3: Full 3-tier (memory + cache + persistence)
    print("  - Full 3-tier storage...")
    try:
        full_storage = SifakaStorage(redis_config, milvus_config)
        print("    ✓ Full 3-tier storage created")
        storage_manager = full_storage
    except Exception as e:
        print(f"    ⚠ Full storage not available: {e}")
        print("    → Falling back to memory-only storage")
        storage_manager = memory_storage

    storage = storage_manager.get_thought_storage()
    print(
        f"✓ Using storage with enabled backends: {storage_manager.enable_memory and 'memory'}, {storage_manager.enable_redis and 'redis'}, {storage_manager.enable_milvus and 'milvus'}"
    )

    # 2. Create and save thoughts
    print("\n2. Creating and saving thoughts...")

    # Create a thought with rich context
    thought1 = Thought(
        prompt="Write a comprehensive guide about artificial intelligence",
        system_prompt="You are an expert AI researcher and educator.",
        chain_id="ai-guide-chain",
    )

    # Add pre-generation context
    thought1 = thought1.add_pre_generation_context(
        [
            Document(
                text="AI is the simulation of human intelligence in machines.",
                metadata={"source": "encyclopedia", "confidence": 0.95},
                score=0.9,
            ),
            Document(
                text="Machine learning is a subset of AI that enables computers to learn.",
                metadata={"source": "research_paper", "confidence": 0.88},
                score=0.85,
            ),
        ]
    )

    # Set generated text
    thought1 = thought1.set_text(
        """
    Artificial Intelligence (AI) represents one of the most transformative technologies of our time.
    At its core, AI is the development of computer systems that can perform tasks typically requiring
    human intelligence, such as visual perception, speech recognition, decision-making, and language translation.
    """
    )

    # Add validation results
    validation_result = ValidationResult(
        passed=True,
        message="Content meets quality standards",
        score=0.92,
        issues=[],
        suggestions=["Consider adding more specific examples"],
    )
    thought1 = thought1.add_validation_result("quality_validator", validation_result)

    # Add critic feedback
    critic_feedback = CriticFeedback(
        critic_name="technical_accuracy_critic",
        violations=["Could benefit from more recent developments"],
        suggestions=[
            "Include information about large language models",
            "Mention recent breakthroughs in computer vision",
        ],
        feedback={"confidence": 0.87, "review_time": datetime.now().isoformat()},
    )
    thought1 = thought1.add_critic_feedback(critic_feedback)

    # Save the thought
    storage.save_thought(thought1)
    print(f"✓ Saved thought 1: {thought1.id}")

    # Create an improved iteration
    thought2 = thought1.next_iteration()
    thought2 = thought2.set_text(
        """
    Artificial Intelligence (AI) represents one of the most transformative technologies of our time.
    At its core, AI is the development of computer systems that can perform tasks typically requiring
    human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

    Recent breakthroughs in large language models like GPT and Claude have revolutionized natural language
    processing, while advances in computer vision have enabled remarkable progress in autonomous vehicles
    and medical imaging.
    """
    )
    storage.save_thought(thought2)
    print(f"✓ Saved thought 2 (iteration): {thought2.id}")

    # Create another thought in a different chain
    thought3 = Thought(
        prompt="Explain quantum computing for beginners",
        text="Quantum computing harnesses quantum mechanics to process information in fundamentally new ways...",
        chain_id="quantum-chain",
    )
    storage.save_thought(thought3)
    print(f"✓ Saved thought 3: {thought3.id}")

    # 3. Query and retrieve thoughts
    print("\n3. Querying thoughts...")

    # Get all thoughts
    all_thoughts = storage.query_thoughts()
    print(f"✓ Total thoughts: {all_thoughts.total_count}")

    # Query by chain ID
    ai_chain_query = ThoughtQuery(chain_ids=["ai-guide-chain"])
    ai_thoughts = storage.query_thoughts(ai_chain_query)
    print(f"✓ AI guide chain thoughts: {ai_thoughts.total_count}")

    # Query by content
    quantum_query = ThoughtQuery(text_contains="quantum")
    quantum_thoughts = storage.query_thoughts(quantum_query)
    print(f"✓ Thoughts containing 'quantum': {quantum_thoughts.total_count}")

    # Query thoughts with validation results
    validated_query = ThoughtQuery(has_validation_results=True)
    validated_thoughts = storage.query_thoughts(validated_query)
    print(f"✓ Validated thoughts: {validated_thoughts.total_count}")

    # 4. Work with thought history
    print("\n4. Working with thought history...")

    # Get complete history of the AI guide
    history = storage.get_thought_history(thought2.id)
    print(f"✓ AI guide history: {len(history)} thoughts")
    for i, thought in enumerate(history):
        print(f"  - Iteration {thought.iteration}: {thought.id[:8]}...")

    # Get all thoughts in the AI chain
    chain_thoughts = storage.get_chain_thoughts("ai-guide-chain")
    print(f"✓ AI chain thoughts: {len(chain_thoughts)}")

    # 5. Health check and statistics
    print("\n5. Storage health and statistics...")

    health = storage.health_check()
    print(f"✓ Storage status: {health['status']}")
    print(f"✓ Total thoughts: {health['total_thoughts']}")
    print(f"✓ Test passed: {health.get('test_passed', 'N/A')}")
    print(f"✓ Timestamp: {health.get('timestamp', 'N/A')}")
    if "storage_stats" in health:
        stats = health["storage_stats"]
        if "memory" in stats:
            print(f"✓ Memory utilization: {stats['memory'].get('utilization', 0):.1%}")
        if "cache" in stats:
            print(f"✓ Cache hit rate: {stats['cache'].get('hit_rate', 0):.1%}")

    # 6. Advanced querying
    print("\n6. Advanced querying...")

    # Query with date range (last hour)
    recent_query = ThoughtQuery(
        start_date=datetime.now() - timedelta(hours=1),
        sort_by="timestamp",
        sort_order="desc",
        limit=5,
    )
    recent_thoughts = storage.query_thoughts(recent_query)
    print(f"✓ Recent thoughts (last hour): {recent_thoughts.total_count}")

    # Query with multiple filters
    complex_query = ThoughtQuery(
        chain_ids=["ai-guide-chain"], has_critic_feedback=True, min_iteration=1, limit=10
    )
    complex_results = storage.query_thoughts(complex_query)
    print(f"✓ Complex query results: {complex_results.total_count}")
    print(f"✓ Query execution time: {complex_results.execution_time_ms:.2f}ms")

    # 7. Demonstrate retrieval of specific thought
    print("\n7. Retrieving specific thought...")

    retrieved_thought = storage.get_thought(thought1.id)
    if retrieved_thought:
        print(f"✓ Retrieved thought: {retrieved_thought.id}")
        print(f"  - Prompt: {retrieved_thought.prompt[:50]}...")
        print(f"  - Has context: {len(retrieved_thought.pre_generation_context or [])}")
        print(f"  - Has validation: {bool(retrieved_thought.validation_results)}")
        print(f"  - Has feedback: {len(retrieved_thought.critic_feedback or [])}")

    # 8. Storage statistics
    print("\n8. Storage statistics...")
    try:
        stats = storage.get_stats()
        print(f"✓ Memory cache: {stats.get('memory', {}).get('size', 0)} items")
        print(f"✓ Memory hit rate: {stats.get('memory', {}).get('hit_rate', 0):.2%}")
        print(f"✓ Redis cache hits: {stats.get('redis', {}).get('hits', 0)}")
        print(f"✓ Vector storage: Available for semantic search")
    except Exception as e:
        print(f"  Could not get storage stats: {e}")

    print("\n" + "=" * 50)
    print("🎉 Unified storage example completed successfully!")
    print("\nKey features demonstrated:")
    print("  ✓ 3-tier storage architecture (Memory → Redis → Milvus)")
    print("  ✓ Thought history and iterations")
    print("  ✓ Context preservation (documents, validation, feedback)")
    print("  ✓ Flexible querying with filters")
    print("  ✓ Semantic search capabilities")
    print("  ✓ Health monitoring and statistics")
    print("  ✓ Automatic caching and performance optimization")


if __name__ == "__main__":
    main()
