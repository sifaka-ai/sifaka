"""Example of using Reflexion critic for iterative text improvement.

Reflexion uses self-reflection on previous attempts to identify and correct mistakes.
"""

import asyncio
from sifaka import improve


async def main():
    """Run Reflexion improvement example."""

    # Original text to improve
    text = """
    Climate change is a big problem. It's getting hotter and the ice is melting.
    We should probably do something about it soon or things will get worse.
    """

    print("🔍 Reflexion Example - Learning from iterations")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Reflexion critic
    result = await improve(text, critics=["reflexion"], max_iterations=3)

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
