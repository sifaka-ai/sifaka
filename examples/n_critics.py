#!/usr/bin/env python3
"""Example demonstrating N-Critics with Simple Built-in Features.

This example shows how to use multiple critics with the simple configuration
approach, featuring built-in logging, timing, and caching.
"""

import asyncio
import os

# Simple imports - no complex dependencies needed
import sifaka
from sifaka import SifakaConfig, SifakaEngine


def create_simple_config():
    """Create a simple configuration with multiple critics and built-in features."""

    # Use the simple configuration approach with built-in features
    config = (
        SifakaConfig.builder()
        .model("openai:gpt-4o-mini")  # Fast, cost-effective generator
        .max_iterations(3)  # Allow multiple iterations for improvement
        .min_length(200)  # Minimum content length
        .max_length(1000)  # Maximum content length
        .critics(["reflexion", "constitutional", "self_refine"])  # Multiple critics
        .with_logging(log_level="INFO", log_content=False)  # Enable logging
        .with_timing()  # Enable performance timing
        .with_caching(cache_size=100)  # Enable caching for repeated runs
        .build()
    )

    return config


async def main():
    """Demonstrate multiple critics with simple built-in features."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🤖 Multiple Critics with Simple Built-in Features")
    print("=" * 60)

    # The prompt we'll use for generation
    prompt = "Write a balanced analysis of AI's impact on healthcare, covering both benefits and potential risks."

    # Create simple configuration with multiple critics
    config = create_simple_config()

    # Create engine with simple configuration
    engine = SifakaEngine(config=config)

    # Generate and critique with multiple critics
    print("🔄 Running generation + critique workflow...")
    thought = await engine.think(prompt, max_iterations=3)

    # Display results using simple built-in information
    print("\n✅ Multiple Critics Results:")

    # Show simple overview
    print(f"Final text: {len(thought.final_text or thought.current_text)} characters")
    print(f"Iterations: {thought.iteration}")
    print(f"Validation passed: {thought.validation_passed()}")
    print(f"Total generations: {len(thought.generations)}")
    print(f"Total validations: {len(thought.validations)}")
    print(f"Total critiques: {len(thought.critiques)}")

    # Show which critics were applied
    if thought.critiques:
        print(f"\nCritics Applied:")
        applied_critics = set(critique.critic for critique in thought.critiques)
        for critic in applied_critics:
            count = sum(1 for c in thought.critiques if c.critic == critic)
            print(f"  - {critic}: {count} times")

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

    # Show final text preview
    final_text = thought.final_text or thought.current_text
    print(f"\n📝 Final Text Preview:")
    print(f"{final_text[:200]}..." if len(final_text) > 200 else final_text)

    print("\n✅ Multiple Critics with Simple Built-in Features completed!")
    print(
        "Key Benefits: Multiple perspectives, built-in performance monitoring, simple configuration"
    )


if __name__ == "__main__":
    asyncio.run(main())
