#!/usr/bin/env python3
"""Simple MetaRewardingCritic example using Gemini.

This example demonstrates the basic usage of MetaRewardingCritic with Google Gemini,
using no validators, no retrieval, and only one chain retry for simplicity.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Install google-generativeai: pip install google-generativeai

The MetaRewardingCritic implements a two-stage judgment process:
1. Initial judgment of the response quality
2. Meta-judgment that evaluates the quality of the initial judgment
This creates a feedback loop for improving both responses and judgment capabilities.
"""

import os
from sifaka import Chain
from sifaka.models import create_model
from sifaka.critics.meta_rewarding import MetaRewardingCritic


def main():
    """Run a simple MetaRewardingCritic example with Gemini."""

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return

    print("🤖 Simple MetaRewardingCritic Example with Gemini")
    print("=" * 50)

    try:
        # Create Gemini model
        print("📡 Creating Gemini model...")
        model = create_model("gemini:gemini-1.5-flash")
        print("✅ Model created successfully")

        # Create MetaRewardingCritic with simple configuration
        print("🔍 Creating MetaRewardingCritic...")
        critic = MetaRewardingCritic(
            model=model,
            # Use same model for meta-judgment (simplest setup)
            meta_judge_model=model,
            # Enable scoring for clearer feedback
            use_scoring=True,
            score_range=(1, 10),
        )
        print("✅ Critic created successfully")

        # Create a simple chain
        print("⛓️  Creating chain...")
        chain = Chain(
            model=model,
            prompt="Write a brief explanation of how artificial intelligence works, suitable for a general audience.",
            max_improvement_iterations=1,  # Only one retry
            always_apply_critics=True,  # Always apply the critic
        )

        # Add the critic to the chain
        chain.improve_with(critic)
        print("✅ Chain configured successfully")

        # Run the chain
        print("\n🚀 Running chain...")
        print("-" * 30)

        result = chain.run()

        # Display results
        print("\n📝 Results:")
        print("=" * 50)

        print(f"\n📊 Final iteration: {result.iteration}")
        print(f"📏 Text length: {len(result.text)} characters")

        print(f"\n📖 Generated text:")
        print("-" * 20)
        print(result.text)

        # Show critic feedback if available
        if result.critic_feedback:
            print(f"\n🔍 Critic feedback ({len(result.critic_feedback)} entries):")
            print("-" * 30)
            for i, feedback in enumerate(result.critic_feedback, 1):
                print(f"\nFeedback {i}:")
                print(f"  Critic: {feedback.critic_name}")
                print(f"  Needs improvement: {feedback.needs_improvement}")
                print(f"  Message: {feedback.message[:200]}...")  # Truncate for readability

        # Show improvement history if available
        if hasattr(result, "improvement_history") and result.improvement_history:
            print(f"\n📈 Improvement history ({len(result.improvement_history)} iterations):")
            print("-" * 30)
            for i, iteration in enumerate(result.improvement_history):
                print(f"  Iteration {i + 1}: {len(iteration)} characters")

        print("\n✅ Example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Installed google-generativeai package")
        print("3. Valid API key with sufficient quota")


if __name__ == "__main__":
    main()
