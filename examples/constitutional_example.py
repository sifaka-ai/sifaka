"""Example of using Constitutional AI critic for principle-based evaluation.

Constitutional AI evaluates text against a set of principles for safety and quality.
"""

import asyncio
import os

from sifaka import Config, improve
from sifaka.storage.file import FileStorage


async def main() -> None:
    """Run Constitutional AI improvement example."""

    # Text that might need safety improvements
    text = """
    To get rid of pests in your garden, you should use the strongest chemicals
    available. Just spray everything heavily and don't worry about the instructions
    on the label - more is always better when dealing with bugs.
    """

    print("⚖️ Constitutional AI Example - Principle-based evaluation")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Constitutional critic using Anthropic
    result = await improve(
        text,
        critics=["constitutional"],
        max_iterations=3,
        config=Config(
            model="gpt-3.5-turbo",  # Fast Anthropic model
            critic_model="gpt-3.5-turbo",  # Same model for critics
            temperature=0.6,
        ),
        storage=FileStorage(),  # Enable thought logging
    )

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Uses ANTHROPIC_API_KEY for Claude models
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   This example uses Claude for constitutional evaluation")
    else:
        asyncio.run(main())
