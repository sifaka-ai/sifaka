#!/usr/bin/env python3
"""Simple SelfConsistencyCritic example using Gemini (PydanticAI).

This example demonstrates the basic usage of SelfConsistencyCritic with Google Gemini,
using no validators, no retrieval, and only one chain retry for simplicity.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Install google-generativeai: pip install google-generativeai
- Install pydantic-ai: pip install pydantic-ai

The SelfConsistencyCritic generates multiple critiques of the same text and uses
consensus to determine the most reliable feedback. This improves critique reliability
by reducing the impact of single inconsistent or low-quality critiques.
"""

import os

from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models import create_model
from sifaka.storage import FileStorage


def main():
    """Run a simple SelfConsistencyCritic example with Gemini."""

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return

    print("🤖 Simple SelfConsistencyCritic Example with Gemini (PydanticAI)")
    print("=" * 60)

    try:
        # Create PydanticAI agent with Gemini model
        print("📡 Creating PydanticAI Gemini agent...")
        agent = Agent(
            "google-gla:gemini-2.0-flash",
            system_prompt=(
                "You are an AI education expert and science communicator. Provide clear, "
                "accurate explanations of artificial intelligence concepts that are accessible "
                "to general audiences. Use analogies, examples, and structured explanations "
                "to make complex topics understandable. Maintain an engaging and informative tone."
            ),
        )
        print("✅ Agent created successfully")

        # Create Sifaka model for the critic (critics still need Sifaka models)
        print("📡 Creating Sifaka model for critic...")
        critic_model = create_model("pydantic-ai:google-gla:gemini-2.0-flash")
        print("✅ Critic model created successfully")

        # Create SelfConsistencyCritic with simple configuration
        print("🔍 Creating SelfConsistencyCritic...")
        critic = SelfConsistencyCritic(
            model=critic_model,
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

        # Create a PydanticAI chain
        print("⛓️  Creating PydanticAI chain...")
        chain = create_pydantic_chain(
            agent=agent,
            critics=[critic],
            max_improvement_iterations=1,  # Only one retry
            always_apply_critics=True,  # Always apply the critic
            storage=FileStorage(
                "./thoughts/self_consistency_simple_thoughts.json",
                overwrite=True,  # Overwrite existing file instead of appending
            ),  # Save thoughts to single JSON file for debugging
        )
        print("✅ Chain configured successfully")

        # Run the chain
        print("\n🚀 Running chain...")
        print("-" * 30)

        # Define the prompt for the PydanticAI chain
        prompt = "Explain the concept of machine learning and provide a simple example that anyone can understand."
        result = chain.run(prompt)

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
                print(f"  Message: {feedback.feedback[:200]}...")  # Truncate for readability

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
        print("💡 This example now uses PydanticAI for modern agent-based workflows!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have:")
        print("1. Set GOOGLE_API_KEY environment variable")
        print("2. Installed google-generativeai package")
        print("3. Installed pydantic-ai package")
        print("4. Valid API key with sufficient quota")


if __name__ == "__main__":
    main()
