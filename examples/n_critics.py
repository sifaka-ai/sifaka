#!/usr/bin/env python3
"""Example demonstrating multi-model N-Critics ensemble.

This example shows how to use the N-Critics implementation in true multi-model mode,
where each critical perspective uses a different model for specialized evaluation.
"""

import asyncio
from datetime import datetime
from dotenv import load_dotenv

from sifaka.critics.n_critics import NCriticsCritic
from sifaka.core.engine import SifakaEngine
from sifaka.graph.dependencies import SifakaDependencies
from sifaka.storage import SifakaFilePersistence
from sifaka.utils.thought_inspector import get_thought_overview

# Load environment variables
load_dotenv()


async def main():
    """Demonstrate multi-model N-Critics ensemble with full generation workflow."""

    print("🤖 Multi-Model N-Critics Ensemble with Full Generation")
    print("=" * 60)

    # The prompt we'll use for generation
    prompt = "Write a balanced analysis of AI's impact on healthcare, covering both benefits and potential risks."

    # Set up storage for thoughts
    storage = SifakaFilePersistence("thoughts", file_prefix="n_critics_")

    # Example 1: Single-model N-Critics with full generation workflow
    print("\n📝 Single-Model N-Critics with Full Generation")

    # Create dependencies with single-model N-Critics
    single_model_deps = SifakaDependencies(
        generator="openai:gpt-4o-mini",  # Generator model
        critics={"n_critics": "groq:llama-3.1-8b-instant"},  # Single model for all perspectives
    )

    # Create engine with single-model configuration
    single_engine = SifakaEngine(dependencies=single_model_deps, persistence=storage)

    # Generate and critique with single-model N-Critics
    print("🔄 Running generation + critique workflow...")
    thought_single = await single_engine.think(prompt, max_iterations=2)

    # Display single-model results
    single_overview = get_thought_overview(thought_single)
    print(
        f"✅ Single-Model Results: {single_overview['final_text_length']} chars, "
        f"{single_overview['total_critiques']} critiques, "
        f"validation: {single_overview['validation_passed']}"
    )

    # Example 2: Multi-model N-Critics with full generation workflow
    print("\n🎭 Multi-Model N-Critics with Full Generation")

    # Create custom N-Critics with different models for each perspective
    perspective_models = {
        "Clarity": "groq:llama-3.1-8b-instant",  # Fast model for clarity
        "Accuracy": "anthropic:claude-3-5-haiku-latest",  # Accurate model for facts
        "Completeness": "openai:gpt-4o-mini",  # Comprehensive model
        "Style": "groq:mixtral-8x7b-32768",  # Creative model for style
    }

    # Create multi-model N-Critics instance
    multi_model_critic = NCriticsCritic(perspective_models=perspective_models)

    # Create generator agent
    from pydantic_ai import Agent

    generator = Agent(
        "openai:gpt-4o-mini",
        system_prompt="Generate high-quality content using available tools when needed. Focus on accuracy, clarity, and helpfulness.",
    )

    # Create dependencies with multi-model N-Critics
    multi_model_deps = SifakaDependencies(
        generator=generator,
        critics={"n_critics": multi_model_critic},  # Use our custom multi-model critic
        validators=[],
        retrievers={},
    )

    # Create engine with multi-model configuration
    multi_engine = SifakaEngine(dependencies=multi_model_deps, persistence=storage)

    # Generate and critique with multi-model N-Critics
    print("🔄 Running generation + critique workflow...")
    thought_multi = await multi_engine.think(prompt, max_iterations=2)

    # Display multi-model results
    multi_overview = get_thought_overview(thought_multi)
    print(
        f"✅ Multi-Model Results: {multi_overview['final_text_length']} chars, "
        f"{multi_overview['total_critiques']} critiques, "
        f"validation: {multi_overview['validation_passed']}"
    )

    # Show key differences between approaches
    if thought_multi.critiques:
        latest_critique = thought_multi.critiques[-1]
        is_true_ensemble = latest_critique.critic_metadata.get("is_true_ensemble", False)
        agreement_ratio = latest_critique.critic_metadata.get("perspective_agreement", 0)
        print(f"🎯 True ensemble: {is_true_ensemble}, Agreement: {agreement_ratio:.2f}")

    # Compare the two approaches
    print(f"\n📊 Comparison:")
    print(
        f"Single-model: {len(thought_single.final_text)} chars, {thought_single.iteration} iterations"
    )
    print(
        f"Multi-model: {len(thought_multi.final_text)} chars, {thought_multi.iteration} iterations"
    )

    # Show storage information
    print(f"\n💾 Thoughts saved to: {storage.storage_dir}")
    print(f"Single-model ID: {thought_single.id}")
    print(f"Multi-model ID: {thought_multi.id}")

    # Demonstrate thought retrieval using model_dump
    print(f"\n🔍 Thought data available via:")
    print(f"• thought.model_dump() - Full serializable dict")
    print(f"• thought.model_dump_json() - JSON string")
    print(f"• Storage backend - Automatic persistence")

    print("\n✅ N-Critics demonstration complete!")
    print("Key Benefits: Specialized models per perspective, true ensemble diversity, reduced bias")


if __name__ == "__main__":
    asyncio.run(main())
