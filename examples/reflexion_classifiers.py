#!/usr/bin/env python3
"""Reflexion Critic with Simple Built-in Features.

This example demonstrates:
1. Using the Reflexion critic for self-reflection and improvement
2. Simple built-in validation with length and content requirements
3. Built-in logging, timing, and caching features
4. Clean, simple configuration without complex dependencies
5. Performance monitoring and result analysis

The example uses a prompt that might trigger validation failures to demonstrate
the reflexion process when validation constraints are not met.
"""

import asyncio
import logging
import os

# Simple imports - no complex dependencies needed
import sifaka
from sifaka import SifakaConfig, SifakaEngine

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_simple_config():
    """Create a simple configuration with built-in features and validation."""

    # Use the simple configuration approach with built-in features
    config = (
        SifakaConfig.builder()
        .model("openai:gpt-4o-mini")  # Fast, cost-effective generator
        .max_iterations(3)  # Allow multiple iterations for improvement
        .min_length(100)  # Minimum content length
        .max_length(800)  # Maximum content length
        .critics(["reflexion"])  # Use reflexion critic
        .with_logging(log_level="INFO", log_content=False)  # Enable logging
        .with_timing()  # Enable performance timing
        .with_caching(cache_size=50)  # Enable caching for repeated runs
        .build()
    )

    return config


async def main():
    """Run the Reflexion critic example with simple built-in features."""

    # Ensure API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    print("🚀 Starting Reflexion Critic with Simple Built-in Features")

    # Create simple configuration with built-in features
    config = create_simple_config()

    # Create the Sifaka engine with simple configuration
    engine = SifakaEngine(config=config)

    # Test prompt that might trigger validation failures
    # This prompt could potentially trigger length or content validation failures
    test_prompt = """
    Write a comprehensive explanation about how artificial intelligence technology
    will completely revolutionize the future of humanity. Discuss both the amazing
    benefits and the potential risks that could be catastrophic if we're not careful.
    Make sure to cover the technological aspects and future implications.
    """

    print("🤖 Running generation with potential validation challenges...")
    print("This prompt might trigger validation failures, causing Reflexion critic to activate")

    try:
        # Run the generation process
        result = await engine.think(
            prompt=test_prompt,
            max_iterations=3,  # Allow multiple iterations for improvement
        )

        # Display results
        print("\n" + "=" * 80)
        print("REFLEXION CRITIC WITH SIMPLE BUILT-IN FEATURES - RESULTS")
        print("=" * 80)

        print(f"\nValidation Passed: {result.validation_passed()}")
        print(f"Total Iterations: {result.iteration}")
        print(f"Thought ID: {result.id}")

        # Use final_text if available, otherwise current_text
        final_text = result.final_text or result.current_text or "No text generated"
        print(f"\nFinal Text ({len(final_text)} characters):")
        print("-" * 50)
        print(final_text)

        # Show validation results
        if result.validations:
            print(f"\nValidation Results:")
            print("-" * 30)
            for validation in result.validations:
                status = "✅ PASSED" if validation.passed else "❌ FAILED"
                print(f"  {validation.validator}: {status}")

        # Show critic results if any were applied
        if result.critiques:
            print(f"\nCritic Results:")
            print("-" * 30)
            for critique in result.critiques:
                print(f"  {critique.critic}: Applied (confidence: {critique.confidence})")

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

        print(f"\nThought Information:")
        print(f"  - Thought ID: {result.id}")
        print(f"  - Total generations: {len(result.generations)}")
        print(f"  - Total validations: {len(result.validations)}")
        print(f"  - Total critiques: {len(result.critiques)}")
        print(f"  - Techniques applied: {result.techniques_applied}")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise

    print("✅ Reflexion Critic with Simple Built-in Features completed!")


if __name__ == "__main__":
    asyncio.run(main())
