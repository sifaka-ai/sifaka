#!/usr/bin/env python3
"""Simple SelfConsistencyCritic example using PydanticAI with Groq.

This example demonstrates the basic usage of SelfConsistencyCritic with Groq,
using no validators, no retrieval, and only one chain retry for simplicity.

Requirements:
- Set GROQ_API_KEY environment variable
- Install pydantic-ai: pip install pydantic-ai

The SelfConsistencyCritic generates multiple critiques of the same text and uses
consensus to determine the most reliable feedback. This improves critique reliability
by reducing the impact of single inconsistent or low-quality critiques.
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent

# Load environment variables from .env file
load_dotenv()

from sifaka.agents import create_pydantic_chain
from sifaka.critics.self_consistency import SelfConsistencyCritic
from sifaka.models import create_model
from sifaka.storage import FileStorage


async def main():
    """Run a simple SelfConsistencyCritic example with Groq."""

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-api-key-here'")
        return

    print("🤖 Simple SelfConsistencyCritic Example with Groq (PydanticAI)")
    print("=" * 60)

    try:
        # Create PydanticAI agent with Groq model
        print("📡 Creating PydanticAI Groq agent...")
        agent = Agent(
            "groq:llama-3.3-70b-versatile",
            system_prompt=(
                "You are an AI education expert and science communicator. Provide clear, "
                "accurate explanations of artificial intelligence concepts that are accessible "
                "to general audiences. Use analogies, examples, and structured explanations "
                "to make complex topics understandable. Maintain an engaging and informative tone."
            ),
        )
        print("✅ Agent created successfully")

        # Create Sifaka model for the critic (using OpenAI since Groq not yet supported in Sifaka)
        print("📡 Creating Sifaka model for critic (using OpenAI since Groq not yet supported)...")
        critic_model = create_model("openai:gpt-4o-mini")
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
            max_improvement_iterations=2,  # Only one retry
            always_apply_critics=True,  # Always apply the critic
            analytics_storage=FileStorage(
                "./thoughts/self_consistency_example_groq_thoughts.json",
                overwrite=True,  # Overwrite existing file instead of appending
            ),  # Save thoughts to single JSON file for debugging
        )
        print("✅ Chain configured successfully")

        # Run the chain
        print("\n🚀 Running chain...")
        print("-" * 30)

        # Define the prompt for the PydanticAI chain
        prompt = "Explain the concept of machine learning and provide a simple example that anyone can understand."
        result = await chain.run(prompt)

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
                print(f"  Message: {feedback.feedback}")  # Show full feedback

                # Show confidence if available (SelfConsistency specific)
                if hasattr(feedback, "confidence"):
                    print(f"  Confidence: {feedback.confidence:.2f}")

                # Debug: Show raw individual critiques if available
                if hasattr(feedback, "metadata") and feedback.metadata:
                    individual_critiques = feedback.metadata.get("individual_critiques", [])
                    if individual_critiques:
                        print(
                            f"\n  🔍 DEBUG: Raw model responses ({len(individual_critiques)} critiques):"
                        )
                        for j, critique in enumerate(individual_critiques[:2], 1):  # Show first 2
                            print(f"    Raw Critique {j}:")
                            print(f"    {critique.get('message', 'No message')[:300]}...")
                            print(f"    Parsed Issues: {critique.get('issues', [])}")
                            print(f"    Parsed Suggestions: {critique.get('suggestions', [])}")
                            print()

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
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Installed pydantic-ai package")
        print("3. Valid API key with sufficient quota")


if __name__ == "__main__":
    asyncio.run(main())
