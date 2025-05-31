#!/usr/bin/env python3
"""Meta-Rewarding Critic Example.

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

from pydantic_ai import Agent

from sifaka.agents import create_pydantic_chain
from sifaka.critics.meta_rewarding import MetaRewardingCritic
from sifaka.models import create_model
from sifaka.storage import FileStorage


def main():
    """Run a simple MetaRewardingCritic example with Gemini."""

    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🤖 Meta-Rewarding Critic Example")
    print("=" * 50)

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

        # Create OpenAI model for the critic (use regular model, not PydanticAI model)
        print("🔍 Creating critic model...")
        critic_model = create_model("openai:gpt-4o-mini")

        # Create MetaRewardingCritic with simple configuration
        print("🔍 Creating MetaRewardingCritic...")
        critic = MetaRewardingCritic(
            model=critic_model,
            # Use same model for meta-judgment (simplest setup)
            meta_judge_model=critic_model,
            # Enable scoring for clearer feedback
            use_scoring=True,
            score_range=(1, 10),
        )
        print("✅ Critic created successfully")

        # Create a PydanticAI chain
        print("⛓️  Creating PydanticAI chain...")
        chain = create_pydantic_chain(
            agent=agent,
            critics=[critic],
            max_improvement_iterations=2,  # Only two retries
            always_apply_critics=True,  # Always apply the critic
            analytics_storage=FileStorage(
                "./thoughts/meta_rewarding_example_thoughts.json",
                overwrite=True,  # Overwrite existing file instead of appending
            ),  # Save thoughts to single JSON file for debugging
        )
        print("✅ Chain configured successfully")

        # Define the prompt
        prompt = "Write a brief explanation of how artificial intelligence works, suitable for a general audience."

        # Run the chain
        print("\n🚀 Running PydanticAI chain...")
        print("-" * 30)

        result = chain.run_sync(prompt)

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

                # Show issues and suggestions separately
                if hasattr(feedback, "issues") and feedback.issues:
                    print(f"  Issues ({len(feedback.issues)}):")
                    for j, issue in enumerate(feedback.issues, 1):
                        print(f"    {j}. {issue}")

                if hasattr(feedback, "suggestions") and feedback.suggestions:
                    print(f"  Suggestions ({len(feedback.suggestions)}):")
                    for j, suggestion in enumerate(feedback.suggestions, 1):
                        print(f"    {j}. {suggestion}")

                # Show full message if no structured feedback
                if not (hasattr(feedback, "issues") and feedback.issues) and not (
                    hasattr(feedback, "suggestions") and feedback.suggestions
                ):
                    print(
                        f"  Full message: {feedback.feedback[:500]}..."
                    )  # Show more of the message

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
        print("1. Set GOOGLE_API_KEY and OPENAI_API_KEY environment variables")
        print("2. Installed google-generativeai and openai packages")
        print("3. Valid API keys with sufficient quota")


if __name__ == "__main__":
    main()
