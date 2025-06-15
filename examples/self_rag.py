#!/usr/bin/env python3
"""
Self-RAG with Simple Built-in Features

This example demonstrates Self-RAG (Retrieval-Augmented Generation) using
Sifaka's simple built-in features approach. It shows how to:

1. Use Self-RAG critic with simple configuration
2. Enable built-in logging, timing, and caching
3. Process different types of content that benefit from retrieval
4. Monitor performance with built-in statistics

The example maintains the power of Self-RAG while using a much simpler
configuration approach with built-in features.
"""

import asyncio
import os

# Simple imports - no complex dependencies needed
import sifaka
from sifaka import SifakaConfig, SifakaEngine


def create_simple_config():
    """Create a simple configuration with Self-RAG critic and built-in features."""

    # Use the simple configuration approach with built-in features
    config = (
        SifakaConfig.builder()
        .model("openai:gpt-4o-mini")  # Fast, reliable model
        .max_iterations(3)  # Allow multiple iterations for improvement
        .min_length(150)  # Minimum content length
        .max_length(1200)  # Maximum content length
        .critics(["self_rag"])  # Use Self-RAG critic
        .with_logging(log_level="INFO", log_content=False)  # Enable logging
        .with_timing()  # Enable performance timing
        .with_caching(cache_size=50)  # Enable caching for repeated runs
        .build()
    )

    return config


async def demo_self_rag_simple():
    """Demonstrate Self-RAG with simple built-in features."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🔍 Self-RAG with Simple Built-in Features")
    print("=" * 50)

    # Create simple configuration with Self-RAG
    config = create_simple_config()

    # Create engine with simple configuration
    engine = SifakaEngine(config=config)

    print("✅ Created Self-RAG engine with built-in features")
    print(f"   Model: {config.model}")
    print(f"   Critics: {config.critics}")
    print(
        f"   Built-in features: logging={config.enable_logging}, timing={config.enable_timing}, caching={config.enable_caching}"
    )

    # Test cases with different retrieval needs
    test_cases = [
        {
            "name": "Factual Claims Needing Verification",
            "prompt": "Write about recent developments in AI safety research and include specific examples and statistics.",
        },
        {
            "name": "Technical Explanation",
            "prompt": "Explain how transformer neural networks work, including key innovations and current applications.",
        },
        {
            "name": "Current Events Analysis",
            "prompt": "Analyze the current state of renewable energy adoption globally with recent data and trends.",
        },
    ]

    # Process each test case using the simple engine approach
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")

        print(f"\n📝 Prompt: {test_case['prompt']}")

        # Generate and critique with Self-RAG
        print(f"\n🔍 Running Self-RAG generation + critique workflow...")
        try:
            thought = await engine.think(test_case["prompt"], max_iterations=3)

            # Display results using simple built-in information
            print(f"\n✅ Self-RAG Results:")
            print(f"Final text: {len(thought.final_text or thought.current_text)} characters")
            print(f"Iterations: {thought.iteration}")
            print(f"Validation passed: {thought.validation_passed()}")
            print(f"Total critiques: {len(thought.critiques)}")

            # Show which critics were applied
            if thought.critiques:
                print(f"\nCritics Applied:")
                applied_critics = set(critique.critic for critique in thought.critiques)
                for critic in applied_critics:
                    count = sum(1 for c in thought.critiques if c.critic == critic)
                    print(f"  - {critic}: {count} times")

            # Show final text preview
            final_text = thought.final_text or thought.current_text
            print(f"\n📝 Generated Text Preview:")
            print(f"{final_text[:200]}..." if len(final_text) > 200 else final_text)

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print(f"💡 Make sure you have a valid OpenAI API key in your environment")

    # Show performance stats if timing is enabled
    timing_stats = engine.get_timing_stats()
    if timing_stats.get("total_requests", 0) > 0:
        print(f"\n⏱️ Performance Stats:")
        print(f"Duration: {timing_stats['avg_duration_seconds']:.2f}s")
        print(f"Iterations: {timing_stats['avg_iterations']:.1f}")

    # Show cache stats if caching is enabled
    cache_stats = engine.get_cache_stats()
    if cache_stats.get("cache_size", 0) >= 0:
        print(f"\n💾 Cache Stats:")
        print(f"Cache size: {cache_stats['cache_size']}")

    # Summary
    print(f"\n{'='*60}")
    print("📋 Demo Summary")
    print(f"{'='*60}")
    print(f"✅ Self-RAG with simple built-in features demonstrated")
    print(f"🎯 Retrieval-augmented generation with simple configuration")
    print(f"📊 Built-in performance monitoring and caching")

    print("\n✅ Self-RAG with Simple Built-in Features completed!")
    print(
        "Key Benefits: Retrieval-augmented generation, built-in performance monitoring, simple configuration"
    )


if __name__ == "__main__":
    asyncio.run(demo_self_rag_simple())
