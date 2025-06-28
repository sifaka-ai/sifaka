"""Example of using Self-Refine critic for general text improvement.

Self-Refine provides iterative refinement for overall quality enhancement.
"""

import asyncio
from sifaka import improve


async def main() -> None:
    """Run Self-Refine improvement example."""

    # Generic text that needs polish
    text = """
    Email marketing is good for businesses. You can send emails to customers
    and they might buy things. It's cheaper than other marketing. You should
    collect email addresses and send newsletters.
    """

    print("✨ Self-Refine Example - General improvement")
    print("=" * 50)
    print(f"Original text ({len(text.split())} words):")
    print(text.strip())
    print()

    # Run improvement with Self-Refine critic
    result = await improve(text, critics=["self_refine"], max_iterations=3)

    print(f"✅ Improved text ({len(result.final_text.split())} words):")
    print(result.final_text.strip())
    print(f"\n📊 Iterations: {result.iteration}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    asyncio.run(main())
