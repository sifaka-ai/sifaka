#!/usr/bin/env python3
"""Self-Refine Example with Simple Built-in Features.

This example demonstrates:
- Simple configuration with built-in logging and timing
- Self-Refine critic for iterative improvement
- Comprehensive set of validators for quality assurance
- Built-in performance monitoring and caching
- Clean, simple architecture

The engine will generate content about software engineering best practices
and use multiple validators to ensure high-quality, comprehensive output.
The Self-Refine critic provides iterative feedback for continuous improvement.
"""

import asyncio
import os
from datetime import datetime

# Simple imports - no complex dependencies needed
import sifaka
from sifaka import SifakaConfig, SifakaEngine


def create_simple_config():
    """Create a simple configuration with built-in features and validation."""

    # Use the simple configuration approach with built-in features
    config = (
        SifakaConfig.builder()
        .model("anthropic:claude-3-5-haiku-20241022")
        .max_iterations(3)
        .min_length(1200)  # Ensure substantial content
        .max_length(3000)  # Stay focused
        .critics(["self_refine"])  # Use self-refine critic
        .with_logging(log_level="INFO", log_content=False)  # Enable logging
        .with_timing()  # Enable performance timing
        .with_caching(cache_size=100)  # Enable caching for repeated runs
        .build()
    )

    return config


def show_self_refine_progression(thought):
    """Show how the text improved through Self-Refine iterations."""
    print("\n🔄 Self-Refine Progression:")

    # Show progression by iteration
    for iteration in range(thought.iteration + 1):
        # Get generations for this iteration
        iteration_generations = [g for g in thought.generations if g.iteration == iteration]
        if not iteration_generations:
            continue

        latest_gen = iteration_generations[-1]

        # Get validation status
        iteration_validations = [v for v in thought.validations if v.iteration == iteration]
        passed = sum(1 for v in iteration_validations if v.passed)
        total = len(iteration_validations)

        # Get Self-Refine feedback
        iteration_critiques = [
            c
            for c in thought.critiques
            if c.iteration == iteration and c.critic == "SelfRefineCritic"
        ]

        print(
            f"  Iteration {iteration}: {len(latest_gen.text)} chars, "
            f"validation {passed}/{total}, ",
            end="",
        )

        if iteration_critiques:
            critique = iteration_critiques[-1]
            print(f"confidence {critique.confidence:.2f}")
        else:
            print("no critique")


async def main():
    """Run the Self-Refine example using simple built-in features."""

    # Ensure API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    print("🚀 Self-Refine Example with Simple Built-in Features")

    # Create simple configuration with built-in features
    config = create_simple_config()

    # Create Sifaka engine with simple configuration
    engine = SifakaEngine(config=config)

    # Define the prompt designed to initially fail validation
    prompt = "Write about software engineering. Give me some tips."

    # Run the engine with multiple iterations for Self-Refine to work
    print(f"🤖 Self-Refine Example")
    print(f"Prompt: {prompt}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    thought = await engine.think(prompt, max_iterations=3)

    # Display results using simple built-in information
    print("\n✅ Self-Refine Results:")

    # Show simple overview
    print(f"Final text: {len(thought.final_text or thought.current_text)} characters")
    print(f"Iterations: {thought.iteration}")
    print(f"Validation passed: {thought.validation_passed()}")
    print(f"Total generations: {len(thought.generations)}")
    print(f"Total validations: {len(thought.validations)}")
    print(f"Total critiques: {len(thought.critiques)}")

    # Show Self-Refine progression
    show_self_refine_progression(thought)

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

    print("✅ Self-Refine example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
