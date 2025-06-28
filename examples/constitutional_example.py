"""Example of using Constitutional AI critic for principle-based evaluation.

Constitutional AI evaluates text against a set of principles for safety and quality.
"""

import asyncio
from sifaka import improve


async def main():
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

    # Run improvement with Constitutional critic
    result = await improve(text, critics=["constitutional"], max_iterations=3)

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
