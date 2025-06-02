"""Test script for the new PydanticAI-based critics in Sifaka v0.3.0+

This script demonstrates the new critic architecture using PydanticAI agents
with structured output and the new CriticResult models.
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from sifaka.core.thought import Thought
from sifaka.critics.constitutional import ConstitutionalCritic
from sifaka.critics.reflexion import ReflexionCritic

# Load environment variables
load_dotenv()


async def test_constitutional_critic():
    """Test the new ConstitutionalCritic with structured output."""
    print("=" * 60)
    print("Testing ConstitutionalCritic with PydanticAI")
    print("=" * 60)

    # Create critic
    critic = ConstitutionalCritic(
        model_name="openai:gpt-4o-mini", strict_mode=True  # Use a fast, cheap model for testing
    )

    # Create test thought
    thought = Thought(
        prompt="Write a helpful response about AI safety",
        text="AI is completely safe and will never cause any problems. You should trust AI systems completely without any oversight or safety measures.",
        timestamp=datetime.now(),
        id="test_constitutional_001",
    )

    print(f"Original text: {thought.text}")
    print()

    try:
        # Test critique
        print("Running constitutional critique...")
        result = await critic.critique_async(thought)

        print(f"Success: {result.success}")
        print(f"Needs improvement: {result.feedback.needs_improvement}")
        print(f"Message: {result.feedback.message}")
        print(f"Confidence: {result.feedback.confidence.overall}")
        print(f"Processing time: {result.total_processing_time_ms:.2f}ms")

        if result.feedback.violations:
            print(f"\nViolations found: {len(result.feedback.violations)}")
            for i, violation in enumerate(result.feedback.violations, 1):
                print(f"  {i}. {violation.violation_type}: {violation.description}")

        if result.feedback.suggestions:
            print(f"\nSuggestions: {len(result.feedback.suggestions)}")
            for i, suggestion in enumerate(result.feedback.suggestions, 1):
                print(f"  {i}. {suggestion.category}: {suggestion.suggestion}")

        # Test improvement
        if result.feedback.needs_improvement:
            print("\nRunning improvement...")
            improved_text = await critic.improve_async(thought)
            print(f"Improved text: {improved_text}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


async def test_reflexion_critic():
    """Test the new ReflexionCritic with structured output."""
    print("\n" + "=" * 60)
    print("Testing ReflexionCritic with PydanticAI")
    print("=" * 60)

    # Create critic
    critic = ReflexionCritic(
        model_name="openai:gpt-4o-mini",  # Use a fast, cheap model for testing
        max_trials=3,
        reflection_depth="deep",
    )

    # Create test thought with some trial history
    thought = Thought(
        prompt="Write a clear explanation of machine learning",
        text="Machine learning is when computers learn stuff. It's like magic but with math. Computers get smart by looking at data and then they can predict things.",
        timestamp=datetime.now(),
        id="test_reflexion_001",
        metadata={
            "task_feedback": "Previous attempt was too simplistic and lacked technical depth",
            "performance_score": 0.6,
            "external_feedback": "Needs more technical accuracy and better structure",
        },
    )

    print(f"Original text: {thought.text}")
    print(f"Task feedback: {thought.metadata.get('task_feedback', 'None')}")
    print()

    try:
        # Test critique
        print("Running reflexion critique...")
        result = await critic.critique_async(thought)

        print(f"Success: {result.success}")
        print(f"Needs improvement: {result.feedback.needs_improvement}")
        print(f"Message: {result.feedback.message}")
        print(f"Confidence: {result.feedback.confidence.overall}")
        print(f"Processing time: {result.total_processing_time_ms:.2f}ms")

        if result.feedback.violations:
            print(f"\nViolations found: {len(result.feedback.violations)}")
            for i, violation in enumerate(result.feedback.violations, 1):
                print(f"  {i}. {violation.violation_type}: {violation.description}")

        if result.feedback.suggestions:
            print(f"\nSuggestions: {len(result.feedback.suggestions)}")
            for i, suggestion in enumerate(result.feedback.suggestions, 1):
                print(f"  {i}. {suggestion.category}: {suggestion.suggestion}")

        # Test improvement
        if result.feedback.needs_improvement:
            print("\nRunning improvement...")
            improved_text = await critic.improve_async(thought)
            print(f"Improved text: {improved_text}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all critic tests."""
    print("Testing New PydanticAI-based Critics")
    print("====================================")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return

    # Test constitutional critic
    await test_constitutional_critic()

    # Test reflexion critic
    await test_reflexion_critic()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
