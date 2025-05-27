#!/usr/bin/env python3
"""Simple SelfConsistencyCritic example using Gemini.

This example demonstrates the basic usage of SelfConsistencyCritic with Google Gemini,
using no validators, no retrieval, and only one chain retry for simplicity.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Install google-generativeai: pip install google-generativeai

The SelfConsistencyCritic generates multiple critiques of the same text and uses
consensus to determine the most reliable feedback. This improves critique reliability
by reducing the impact of single inconsistent or low-quality critiques.
"""

import os
from sifaka import Chain
from sifaka.models import create_model
from sifaka.critics.self_consistency import SelfConsistencyCritic


def main():
    """Run a simple SelfConsistencyCritic example with Gemini."""

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return

    print("🤖 Simple SelfConsistencyCritic Example with Gemini")
    print("=" * 50)

    try:
        # Create Gemini model
        print("📡 Creating Gemini model...")
        model = create_model("gemini:gemini-1.5-flash")
        print("✅ Model created successfully")

        # Create SelfConsistencyCritic with simple configuration
        print("🔍 Creating SelfConsistencyCritic...")
        critic = SelfConsistencyCritic(
            model=model,
            # Generate 3 critiques for consensus (small number for simplicity)
            num_iterations=3,
            # Require 60% agreement for consensus
            consensus_threshold=0.6,
            # Use majority vote for aggregation
            aggregation_method="majority_vote",
            # Enable chain of thought for better reasoning
            use_chain_of_thought=True,
        )
        print("✅ Critic created successfully")

        # Create a simple chain
        print("⛓️  Creating chain...")
        chain = Chain(
            model=model,
            prompt="Explain the concept of machine learning and provide a simple example that anyone can understand.",
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

                # Show confidence if available (SelfConsistency specific)
                if hasattr(feedback, "confidence"):
                    print(f"  Confidence: {feedback.confidence:.2f}")

        # Show improvement history if available
        if hasattr(result, "improvement_history") and result.improvement_history:
            print(f"\n📈 Improvement history ({len(result.improvement_history)} iterations):")
            print("-" * 30)
            for i, iteration in enumerate(result.improvement_history):
                print(f"  Iteration {i + 1}: {len(iteration)} characters")

        print("\n✅ Example completed successfully!")
        print("\n💡 Note: SelfConsistencyCritic generated multiple critiques internally")
        print("   and used consensus to provide the most reliable feedback.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Installed google-generativeai package")
        print("3. Valid API key with sufficient quota")


if __name__ == "__main__":
    main()
