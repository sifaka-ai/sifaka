#!/usr/bin/env python3
"""
Example demonstrating Sifaka's JSON persistence capabilities.

This example shows how to:
1. Create and configure JSON storage
2. Save thoughts with context, validation results, and critic feedback
3. Query thoughts with various filters
4. Work with thought history and iterations
5. Perform health checks and maintenance
"""

import os
import shutil
from datetime import datetime, timedelta

from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
from sifaka.persistence import JSONThoughtStorage, create_json_config


def main():
    """Demonstrate persistence capabilities."""
    print("🧠 Sifaka Persistence Example")
    print("=" * 50)

    # 1. Setup storage
    print("\n1. Setting up JSON storage...")
    storage_dir = "./example_storage"

    # Clean up any existing storage
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    # Create storage with configuration
    config = create_json_config(storage_dir=storage_dir, enable_indexing=True, pretty_print=True)
    storage = JSONThoughtStorage(
        storage_dir=config.storage_dir, enable_indexing=config.enable_indexing
    )
    print(f"✓ Storage created at: {storage_dir}")

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
        issues=["Could benefit from more recent developments"],
        suggestions=[
            "Include information about large language models",
            "Mention recent breakthroughs in computer vision",
        ],
        metadata={"confidence": 0.87, "review_time": datetime.now().isoformat()},
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
    from sifaka.persistence.base import ThoughtQuery

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
    print(f"✓ Storage size: {health['total_size_mb']} MB")
    print(f"✓ Directories exist: {health['directories_exist']}")
    print(f"✓ Writable: {health['writable']}")

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

    # 8. Cleanup
    print("\n8. Cleanup...")
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        print("✓ Storage cleaned up")

    print("\n" + "=" * 50)
    print("🎉 Persistence example completed successfully!")
    print("\nKey features demonstrated:")
    print("  ✓ JSON storage with rich metadata")
    print("  ✓ Thought history and iterations")
    print("  ✓ Context preservation (documents, validation, feedback)")
    print("  ✓ Flexible querying with filters")
    print("  ✓ Health monitoring and statistics")
    print("  ✓ Performance tracking")


if __name__ == "__main__":
    main()
